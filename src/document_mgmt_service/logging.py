from __future__ import annotations

import logging
import sys
import threading
import traceback


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def install_exception_hooks() -> None:
    def handle_exception(exc_type, exc, tb) -> None:
        logging.getLogger("document_mgmt_service").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc, tb),
        )

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("document_mgmt_service").critical(
            "Unhandled thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
