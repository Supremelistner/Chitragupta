from __future__ import annotations

from pathlib import Path
import shutil

from document_mgmt_service.domain.ports import FileStorage


# The services are launched with CWD = <project>/src (see activate.py), so
# a relative ``root`` like ``./data/files`` would otherwise resolve to
# ``<project>/src/data/files`` instead of ``<project>/data/files``. To keep
# the documented configuration intuitive, we anchor relative roots to the
# project root (the parent of the ``src/`` directory containing this file).
#
# Layout of this file's path:
#   <project>/src/document_mgmt_service/infrastructure/storage.py
# Hence ``parents[3]`` lands us at <project>.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_root(root: Path) -> Path:
    """Resolve a storage root, anchoring relative paths to the project root.

    Absolute paths are returned unchanged. Relative paths are taken to be
    relative to the project root (the directory that contains ``src/``),
    not the process CWD.
    """
    if root.is_absolute():
        return root
    return (_PROJECT_ROOT / root).resolve()


class LocalFileStorageAdapter(FileStorage):
    def __init__(self, root: Path) -> None:
        self._root = _resolve_root(Path(root))
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
