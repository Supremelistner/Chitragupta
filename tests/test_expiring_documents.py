"""Tests for the expiring-documents path (banner + renewal questions).

Covers, without any live stack:
  * access-service filtering/sorting over repository summaries,
  * the tool-registry entry (route + schema for the LLM),
  * the engine's transport-failure collapse to {"results": []}.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.domain.models import (
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    DocumentSummaryRecord,
    SemanticIndexStatus,
)
from orchestrator_service.application.orchestrator import OrchestrationEngine


def _summary(doc_id: str, days_from_now: int | None) -> DocumentSummaryRecord:
    return DocumentSummaryRecord(
        document_id=doc_id,
        latest_version=1,
        processing_status=DocumentProcessingStatus.COMPLETED,
        privacy=DocumentPrivacyClassification.PRIVATE,
        metadata={},
        description=f"{doc_id} doc",
        expiry_date=(datetime.now(timezone.utc) + timedelta(days=days_from_now))
        if days_from_now is not None else None,
    )


def _service(summaries):
    repo = MagicMock()
    repo.list_documents.return_value = summaries
    return DocumentAccessService(
        repository=repo, storage=MagicMock(), search=MagicMock())


class ListExpiringTests(unittest.TestCase):
    def test_filters_and_sorts_most_urgent_first(self):
        svc = _service([
            _summary("far", 200),
            _summary("soon", 10),
            _summary("overdue", -5),
            _summary("nodate", None),
        ])
        out = svc.list_expiring_documents(within_days=60)
        ids = [r["document_id"] for r in out["results"]]
        self.assertEqual(ids, ["overdue", "soon"])
        # NOTE: no exact day counts — timedelta.days floors toward -inf
        # and truncates positives, so boundary math is inherently ±1.
        self.assertLess(out["results"][0]["days_until_expiry"], 0)
        self.assertGreater(out["results"][1]["days_until_expiry"], 0)

    def test_window_is_honored(self):
        svc = _service([_summary("a", 10), _summary("b", 40)])
        self.assertEqual(len(svc.list_expiring_documents(within_days=30)["results"]), 1)
        self.assertEqual(len(svc.list_expiring_documents(within_days=90)["results"]), 2)

    def test_empty_when_nothing_expires(self):
        svc = _service([_summary("a", 400), _summary("b", None)])
        self.assertEqual(svc.list_expiring_documents()["results"], [])

    def test_naive_datetimes_treated_as_utc(self):
        naive = _summary("n", 5)
        object.__setattr__(naive, "expiry_date", naive.expiry_date.replace(tzinfo=None))
        svc = _service([naive])
        self.assertEqual(len(svc.list_expiring_documents()["results"]), 1)


class RegistryTests(unittest.TestCase):
    def test_tool_registered_without_confirmation(self):
        from orchestrator_service.domain.models import ServiceTarget
        from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
        registry = DefaultToolRegistry()
        self.assertEqual(registry.get_service("list_expiring_documents"), ServiceTarget.DOCUMENT)
        names = [s["function"]["name"] for s in registry.get_all_tool_schemas()]
        self.assertIn("list_expiring_documents", names)
        schema = registry.get_tool_schema("list_expiring_documents")
        assert schema is not None
        self.assertIn("within_days", schema["function"]["parameters"]["properties"])


class EngineExpiringTests(unittest.TestCase):
    def test_transport_failure_collapses_to_empty(self):
        from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
        router = MagicMock()
        router.call_tool.side_effect = ConnectionError("down")
        engine = OrchestrationEngine(
            llm=MagicMock(), sessions=MagicMock(), confirmations=MagicMock(),
            tool_registry=DefaultToolRegistry(), service_router=router,
            translation_service=MagicMock(),
        )
        self.assertEqual(engine.list_expiring_documents(), {"results": []})

    def test_bad_window_falls_back_to_default(self):
        from orchestrator_service.domain.models import ServiceTarget
        from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
        router = MagicMock()
        router.call_tool.return_value = {"results": [{"document_id": "d"}]}
        engine = OrchestrationEngine(
            llm=MagicMock(), sessions=MagicMock(), confirmations=MagicMock(),
            tool_registry=DefaultToolRegistry(), service_router=router,
            translation_service=MagicMock(),
        )
        out = engine.list_expiring_documents(within_days="nonsense")  # type: ignore[arg-type]
        self.assertEqual(len(out["results"]), 1)
        # mock_calls entries are (name, positional_args, kwargs).
        posargs = router.call_tool.mock_calls[0][1]
        self.assertEqual(posargs[1], "list_expiring_documents")
        self.assertEqual(posargs[2], {"within_days": 30})


if __name__ == "__main__":
    unittest.main()
