from __future__ import annotations

from pathlib import Path

from document_mgmt_service.domain.ports import OCRService


class NullOCRAdapter(OCRService):
    def ping(self) -> None:
        return None

    def extract_text(self, file_path: Path) -> str:
        return ""

    def close(self) -> None:
        return None
