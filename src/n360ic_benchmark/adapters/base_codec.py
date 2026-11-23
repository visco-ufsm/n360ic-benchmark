from pathlib import Path
from typing import Protocol


class Codec(Protocol):
    name: str

    def rate_points(self) -> list[str]: ...

    def run(self, image: Path, rate_point: str, workdir: Path): ...
