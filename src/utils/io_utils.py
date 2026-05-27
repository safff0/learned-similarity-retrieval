import json
from collections import OrderedDict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname: str | Path) -> Any:
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: str | Path) -> None:
    """
    Write the content to the given json file.

    Args:
        content (Any JSON-friendly): content to write.
        fname (str): filename of the json file.
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def unzip_archive(zip_path: str | Path, output_dir: str | Path) -> Path:
    """
    Unzip a .zip archive into output_dir.

    Args:
        zip_path: Path to the .zip file.
        output_dir: Directory where files will be extracted.

    Returns:
        Path to the output directory.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    return output_dir
