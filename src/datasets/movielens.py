import logging
from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import requests
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset, UserHistoryItem
from src.registry import register
from src.utils.io_utils import unzip_archive

logger = logging.getLogger(__name__)


class MovieLensVariant(StrEnum):
    ML_1M = "ml-1m"
    ML_20M = "ml-20m"


@dataclass
class MovieLensUser:
    user_id: int
    gender: str
    age: int
    occupation: int
    zip_code: int


@dataclass
class MovieLensMovie:
    movie_id: int
    title: str
    genres: list[str]


@dataclass
class MovieLensRating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass
class MovieLensItem:
    user_id: int
    movie_ids: list[int]        # chronological sequence kept for this user, len = L + 1
    ratings: list[int]          # parallel to movie_ids
    timestamps: list[int]       # parallel to movie_ids
    is_post_split: list[bool]   # parallel to movie_ids; True iff item is val-target eligible


@register("dataset")
class MovieLensDataset(BaseDataset):
    def __init__(
        self,
        variant: MovieLensVariant,
        partition: str = "train",
        data_path: str | None = None,
        rating_threshold: int | None = None,
        max_history_size: int = 200,
        min_history_size: int = 5,
        val_part: float = 0.15,
        split_strategy: str = "time",
    ) -> None:
        zip_path = self._download_dataset(variant, data_path)
        data_dir = unzip_archive(zip_path, zip_path.parent) / variant
        self._variant = variant
        logger.info(f"movielens dataset {self._variant} loaded to {data_dir}")

        self._min_history_size = min_history_size
        self._max_history_size = max_history_size
        self._partition = partition
        self._split_strategy = split_strategy

        self._ratings = self._read_data_file(
            data_dir, "ratings", names=["userId", "movieId", "rating", "timestamp"]
        )
        if rating_threshold:
            self._ratings = self._ratings[self._ratings["rating"] >= rating_threshold]
        self._ratings = self._ratings.sort_values(by="timestamp")
        self._split_timestamp = (
            val_part * self._ratings["timestamp"].max() +
            (1 - val_part) * self._ratings["timestamp"].min()
        )

        self._movies = self._read_data_file(
            data_dir, "movies", names=["movieId", "title", "genres"], encoding="latin-1"
        )
        rated_movie_ids = sorted(self._ratings["movieId"].astype(int).unique().tolist())
        self._movie_id_to_local_id = {movie_id: i + 1 for i, movie_id in enumerate(rated_movie_ids)}
        self._local_id_to_movie_id = {local_id: movie_id for movie_id, local_id in self._movie_id_to_local_id.items()}
        self._all_genres = sorted(set(self._movies["genres"].astype(str).str.split("|").explode()))
        self._genre_to_idx = {g: i for i, g in enumerate(self._all_genres)}
        genre_onehot = self._movies["genres"].astype(str).str.get_dummies(sep="|")[self._all_genres]
        raw_movie_features = dict(zip(
            self._movies["movieId"].astype(int),
            torch.from_numpy(genre_onehot.values).float(),
        ))
        self._movie_features = {
            local_id: raw_movie_features[movie_id]
            for movie_id, local_id in self._movie_id_to_local_id.items()
        }
        try:
            self._users = self._read_data_file(
                data_dir, "users", names=["userId", "gender", "age", "occupation", "zipCode"]
            )
        except FileNotFoundError:
            self._users = pd.DataFrame() 

        self._all_item_ids = list(range(1, len(self._movie_id_to_local_id) + 1))
        self._index = self._build_index()
        logger.info("Unique rated items: %s", len(self._all_item_ids))

    def _read_data_file(self, data_dir: Path, filename: str, names: list[str], encoding: str = "utf-8") -> pd.DataFrame:
        """Helper to read either modern .csv or legacy .dat movielens files."""
        csv_file = data_dir / f"{filename}.csv"
        dat_file = data_dir / f"{filename}.dat"

        if csv_file.exists():
            return pd.read_csv(csv_file, sep=",", header=0, names=names, encoding=encoding)
        elif dat_file.exists():
            return pd.read_csv(dat_file, sep="::", engine="python", header=None, names=names, encoding=encoding)
        else:
            raise FileNotFoundError(f"Missing {filename}.csv or {filename}.dat in {data_dir}")

    def _build_download_url(self, variant: MovieLensVariant) -> str:
        return f"https://files.grouplens.org/datasets/movielens/{variant}.zip"

    def _download_dataset(self, variant: MovieLensVariant, output_path: str | None) -> Path:
        if output_path is None:
            output_path = f"data/movielens/{variant}"
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        url = self._build_download_url(variant)
        zip_path = output_dir / f"{variant}.zip"
        if zip_path.exists():
            return zip_path

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return zip_path

    def _build_index(self) -> list[MovieLensItem]:
        index: list[MovieLensItem] = []
        for user_idx, user_df in tqdm(self._ratings.groupby("userId", sort=False), desc=f"building index for {self._variant}"):
            user_df = user_df.sort_values("timestamp")
            movie_ids = [
                self._movie_id_to_local_id[movie_id]
                for movie_id in user_df["movieId"].astype(int).tolist()
            ]
            timestamps = user_df["timestamp"].astype(int).tolist()
            ratings = user_df["rating"].astype(int).tolist()
            if self._split_strategy == "leave_one_out":
                if len(movie_ids) < self._min_history_size + 1:
                    continue
                if self._partition == "train":
                    movie_ids = movie_ids[:-1]
                    ratings = ratings[:-1]
                    timestamps = timestamps[:-1]
                    if len(movie_ids) < self._min_history_size + 1:
                        continue
                    is_post_split = [False] * len(movie_ids)
                else:
                    is_post_split = [False] * (len(movie_ids) - 1) + [True]
            else:
                is_post_split = [ts >= self._split_timestamp for ts in timestamps]

                if self._partition == "train":
                    keep = [i for i, post in enumerate(is_post_split) if not post]
                    if len(keep) < self._min_history_size + 1:
                        continue
                    movie_ids = [movie_ids[i] for i in keep]
                    ratings = [ratings[i] for i in keep]
                    timestamps = [timestamps[i] for i in keep]
                    is_post_split = [False] * len(movie_ids)
                else:
                    if not any(is_post_split):
                        continue
                    if len(movie_ids) < self._min_history_size + 1:
                        continue

            cap = self._max_history_size + 1
            index.append(
                MovieLensItem(
                    user_id=int(user_idx),
                    movie_ids=movie_ids[-cap:],
                    ratings=ratings[-cap:],
                    timestamps=timestamps[-cap:],
                    is_post_split=is_post_split[-cap:],
                )
            )
        return index

    def __getitem__(self, ind: int) -> UserHistoryItem:
        item = self._index[ind]
        ids = item.movie_ids
        if self._split_strategy == "leave_one_out":
            if self._partition == "train":
                historical_ids = ids[:-1]
                target_ids_list = ids[1:]
                historical_ratings_list = item.ratings[:-1]
                target_ratings_list = item.ratings[1:]
                historical_timestamps_list = item.timestamps[:-1]

                input_ids = torch.tensor(historical_ids, dtype=torch.long)
                history_ratings = torch.tensor(historical_ratings_list, dtype=torch.long)
                history_timestamps = torch.tensor(historical_timestamps_list, dtype=torch.long)
                history_features = torch.stack([self._movie_features[mid] for mid in historical_ids])
                target_ids = torch.tensor(target_ids_list, dtype=torch.long)
                target_feedback = torch.tensor(target_ratings_list, dtype=torch.long)
                loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
                timestamp = item.timestamps[-1]
            else:
                historical_ids = ids[:-1]
                target_id = ids[-1]
                historical_ratings_list = item.ratings[:-1]
                target_rating = item.ratings[-1]
                historical_timestamps_list = item.timestamps[:-1]
                target_timestamp = item.timestamps[-1]

                input_ids = torch.tensor(historical_ids, dtype=torch.long)
                history_ratings = torch.tensor(historical_ratings_list, dtype=torch.long)
                history_timestamps = torch.tensor(historical_timestamps_list, dtype=torch.long)
                history_features = torch.stack([self._movie_features[mid] for mid in historical_ids])

                seq_len = input_ids.size(0)
                target_ids = torch.zeros(seq_len, dtype=torch.long)
                target_ids[-1] = target_id
                target_feedback = torch.zeros(seq_len, dtype=torch.long)
                target_feedback[-1] = target_rating
                loss_mask = torch.zeros(seq_len, dtype=torch.bool)
                loss_mask[-1] = True
                timestamp = target_timestamp
        else:
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            history_ratings = torch.tensor(item.ratings[:-1], dtype=torch.long)
            history_timestamps = torch.tensor(item.timestamps[:-1], dtype=torch.long)
            target_ids = torch.tensor(ids[1:], dtype=torch.long)
            target_feedback = torch.tensor(item.ratings[1:], dtype=torch.long)
            history_features = torch.stack([self._movie_features[mid] for mid in ids[:-1]])
            loss_mask = torch.tensor(item.is_post_split[1:], dtype=torch.bool)
            if self._partition == "train":
                loss_mask = torch.ones_like(loss_mask)
            timestamp = item.timestamps[-1]
        return UserHistoryItem(
            user_id=item.user_id,
            history_ids=input_ids,
            history_ratings=history_ratings,
            history_timestamps=history_timestamps,
            history_features=history_features,
            target=target_ids,
            target_feedback=target_feedback,
            loss_mask=loss_mask,
            timestamp=timestamp,
        )

    @property
    def all_item_ids(self) -> list[int]:
        return self._all_item_ids

    def get_movie_name(self, local_item_id: int) -> str:
        movie_id = self._local_id_to_movie_id.get(local_item_id)
        if not movie_id: return "Unknown"
        return self._movies[self._movies["movieId"] == movie_id]["title"].iloc[0]

    def get_movie_genres(self, local_item_id: int) -> str:
        movie_id = self._local_id_to_movie_id.get(local_item_id)
        if not movie_id: return "Unknown"
        return self._movies[self._movies["movieId"] == movie_id]["genres"].iloc[0]
