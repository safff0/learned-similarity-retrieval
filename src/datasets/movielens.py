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
    history: list[MovieLensRating]
    target_movie: int | None = None
    target_feedback: int | None = None
    timestamp: int | None = None


@register("dataset")
class MovieLensDataset(BaseDataset):
    def __init__(
        self,
        variant: MovieLensVariant,
        partition: str = "train",
        data_path: str | None = None,
        rating_threshold: int | None = None,
        max_history_size: int = 64,
        min_history_size: int = 2,
        val_part: float = 0.15,
    ) -> None:
        zip_path = self._download_dataset(variant, data_path)
        data_dir = unzip_archive(zip_path, zip_path.parent) / variant
        self._variant = variant
        logger.info(f"movielens dataset {self._variant} loaded to {data_dir}")

        self._min_history_size = min_history_size
        self._max_history_size = max_history_size
        self._partition = partition

        self._ratings = pd.read_csv(
            data_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )
        if rating_threshold:
            self._ratings = self._ratings[self._ratings["rating"] >= rating_threshold]
        self._ratings = self._ratings.sort_values(by="timestamp")
        split_timestamp = (
            val_part * self._ratings["timestamp"].max() +
            (1 - val_part) * self._ratings["timestamp"].min()
        )
        if self._partition == "train":
            self._ratings = self._ratings[self._ratings["timestamp"] < split_timestamp]
        elif self._partition == "val":
            self._ratings = self._ratings[self._ratings["timestamp"] >= split_timestamp]

        self._movies = pd.read_csv(
            data_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        self._all_genres = sorted(set(self._movies["genres"].str.split("|").explode()))
        self._genre_to_idx = {g: i for i, g in enumerate(self._all_genres)}
        genre_onehot = self._movies["genres"].str.get_dummies(sep="|")[self._all_genres]
        self._movie_features = dict(zip(
            self._movies["movieId"].astype(int),
            torch.from_numpy(genre_onehot.values).float(),
        ))
        self._users = pd.read_csv(
            data_dir / "users.dat",
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zipCode"],
        )
        self._index = self._build_index()

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
            if len(movie_ids) < self._min_history_size + 1:
                continue

            for i in range(self._min_history_size, len(movie_ids)):
                history = movie_ids[:i][-self._max_history_size :]
                target = movie_ids[i]
                index.append(
                    MovieLensItem(
                        user_id=int(user_idx),
                        history=history,
                        target_movie=int(target),
                        target_feedback=int(ratings[i]),
                        timestamp=int(timestamps[i]),
                    )
                )
        return index

    def __getitem__(self, ind: int) -> UserHistoryItem:
        item = self._index[ind]
        history_ids = [e for e in item.history]
        history_features = [self._movie_features[e] for e in item.history]
        return UserHistoryItem(
            user_id=item.user_id,
            history_ids=history_ids,
            history_features=history_features,
            target=item.target_movie,
            target_feedback=item.target_feedback,
            timestamp=item.timestamp,
        )
