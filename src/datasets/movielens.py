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
    ) -> None:
        zip_path = self._download_dataset(variant, data_path)
        data_dir = unzip_archive(zip_path, zip_path.parent) / variant
        self._variant = variant
        logger.info(f"movielens dataset {self._variant} loaded to {data_dir}")

        self._min_history_size = min_history_size
        self._max_history_size = max_history_size
        self._partition = partition

        self._ratings = self._read_table(
            data_dir,
            "ratings",
            dat_columns=["userId", "movieId", "rating", "timestamp"],
        )
        if rating_threshold:
            self._ratings = self._ratings[self._ratings["rating"] >= rating_threshold]
        self._ratings = self._ratings.sort_values(by="timestamp")
        self._split_timestamp = (
            val_part * self._ratings["timestamp"].max() +
            (1 - val_part) * self._ratings["timestamp"].min()
        )

        self._movies = self._read_table(
            data_dir,
            "movies",
            dat_columns=["movieId", "title", "genres"],
            dat_encoding="latin-1",
        )
        self._all_genres = sorted(set(self._movies["genres"].str.split("|").explode()))
        self._genre_to_idx = {g: i for i, g in enumerate(self._all_genres)}
        genre_onehot = self._movies["genres"].str.get_dummies(sep="|")[self._all_genres]
        self._movie_features = dict(zip(
            self._movies["movieId"].astype(int),
            torch.from_numpy(genre_onehot.values).float(),
        ))
        self._users = self._read_table(
            data_dir,
            "users",
            dat_columns=["userId", "gender", "age", "occupation", "zipCode"],
            required=False,
        )
        self._index = self._build_index()

    @property
    def item_count(self) -> int:
        return int(self._movies["movieId"].astype(int).max()) + 1

    def _read_table(
        self,
        data_dir: Path,
        name: str,
        dat_columns: list[str],
        dat_encoding: str = "utf-8",
        required: bool = True,
    ) -> pd.DataFrame | None:
        csv_path = data_dir / f"{name}.csv"
        dat_path = data_dir / f"{name}.dat"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        if dat_path.exists():
            return pd.read_csv(
                dat_path,
                sep="::",
                engine="python",
                names=dat_columns,
                encoding=dat_encoding,
            )
        if required:
            raise FileNotFoundError(
                f"Neither {csv_path} nor {dat_path} exists for variant {self._variant}"
            )
        return None

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
            movie_ids = user_df["movieId"].astype(int).tolist()
            timestamps = user_df["timestamp"].astype(int).tolist()
            ratings = user_df["rating"].astype(int).tolist()
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
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        target_feedback = torch.tensor(item.ratings[1:], dtype=torch.long)
        history_features = torch.stack([self._movie_features[mid] for mid in ids[:-1]])
        loss_mask = torch.tensor(item.is_post_split[1:], dtype=torch.bool)
        if self._partition == "train":
            loss_mask = torch.ones_like(loss_mask)
        return UserHistoryItem(
            user_id=item.user_id,
            history_ids=input_ids,
            history_features=history_features,
            target=target_ids,
            target_feedback=target_feedback,
            loss_mask=loss_mask,
            timestamp=item.timestamps[-1],
        )
