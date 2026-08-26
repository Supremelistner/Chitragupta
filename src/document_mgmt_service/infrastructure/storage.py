from __future__ import annotations

from pathlib import Path
import shutil

from document_mgmt_service.domain.ports import FileStorage


class LocalFileStorageAdapter(FileStorage):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def ping(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, source: Path, destination_key: str) -> str:
        destination = self._root / destination_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination_key

    def get(self, storage_key: str) -> Path:
        return self._root / storage_key

    def delete(self, storage_key: str) -> None:
        path = self._root / storage_key
        if path.exists():
            path.unlink()

    def exists(self, storage_key: str) -> bool:
        return (self._root / storage_key).exists()

    def close(self) -> None:
        return None
