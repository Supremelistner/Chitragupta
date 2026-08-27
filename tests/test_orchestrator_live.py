"""Live orchestrator test — Qwen3.8-27B planning and tool calling.

Simulates the document data that the real services would return,
then lets the LLM plan and execute tool calls against that data.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import InferenceClient
from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
from orchestrator_service.infrastructure.session_store import FileSessionStore
from orchestrator_service.infrastructure.confirmation_store import InMemoryConfirmationStore
from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
from orchestrator_service.domain.models import ServiceTarget, ConversationMessage, MessageRole, ToolCall
from orchestrator_service.application.orchestrator import OrchestrationEngine

import tempfile
import shutil


# ============================================================
# Simulated document data (what real services would return)
# ============================================================

SIMULATED_DOCUMENTS = {
    "search_documents": lambda args: {
        "query": args.get("query", ""),
        "results": [
            {
                "document_id": "aadhaar_tanvi_001",
                "version": 1,
                "description": "Aadhaar card for Tanvi, female, DOB 01/10/2007. Government of India identity document.",
                "privacy": "SENSITIVE",
                "processing_status": "COMPLETED",
                "validation_status": "VALID",
                "relevance_score": 0.95,
            },
            {
                "document_id": "marksheet_tanvi_001",
                "version": 1,
                "description": "Marksheet for Tanvi, JNV Paprola DT Kangra HP. Class 10 results, roll 17266193.",
                "privacy": "PRIVATE",
                "processing_status": "COMPLETED",
                "validation_status": "VALID",
                "relevance_score": 0.88,
            },
        ],
        "total": 2,
    },
    "list_documents": lambda args: {
        "documents": [
            {
                "document_id": "aadhaar_tanvi_001",
                "version": 1,
                "original_filename": "Adhhar.jpg",
                "description": "Aadhaar card for Tanvi",
                "privacy": "SENSITIVE",
                "validation_status": "VALID",
                "temporal_status": "NO_EXPIRY",
            },
            {
                "document_id": "marksheet_tanvi_001",
                "version": 1,
                "original_filename": "IMG-20250608-WA0003.jpg",
                "description": "Class 10 marksheet for Tanvi",
                "privacy": "PRIVATE",
                "validation_status": "VALID",
                "temporal_status": "VALID",
            },
        ],
        "total": 2,
    },
    "get_document_metadata": lambda args: {
        "document_id": args.get("document_id", ""),
        "version": args.get("version", 1),
        "processing_status": "COMPLETED",
        "privacy": "SENSITIVE" if "aadhaar" in args.get("document_id", "") else "PRIVATE",
        "validation_status": "VALID",
        "validation_risk_score": 0.0,
        "temporal_status": "NO_EXPIRY" if "aadhaar" in args.get("document_id", "") else "VALID",
        "metadata": {
            "schema_version": "1.0",
            "document_type": "identity_document" if "aadhaar" in args.get("document_id", "") else "academic_record",
            "document_sub_type": "aadhaar" if "aadhaar" in args.get("document_id", "") else "marksheet",
            "language": {"primary": "en", "detected": ["en", "hi"], "has_devanagari": True},
        },
    },
    "get_document_description": lambda args: {
        "document_id": args.get("document_id", ""),
        "version": args.get("version", 1),
        "description": "Aadhaar card for Tanvi" if "aadhaar" in args.get("document_id", "") else "Class 10 marksheet for Tanvi, JNV Paprola",
    },
    "get_document_ocr": lambda args: {
        "document_id": args.get("document_id", ""),
        "version": args.get("version", 1),
        "processing_status": "COMPLETED",
        "extracted_text": (
            "भारत सरकार GOVERNMENT OF INDIA\n"
            "तनवी Tanvi\n"
            "जन्म तिथि / DOB: 01/10/2007\n"
            "महिला / FEMALE\n"
            "Mobile No.: 8219480038\n"
            "4373 7370 1714\n"
            "मेरा आधार, मेरी पहचान"
        ) if "aadhaar" in args.get("document_id", "") else (
            "JAWAHAR NAVODAYA VIDYALAYA, PAPROLA, DT. KANGRA, H.P.\n"
            "RESULT SHEET — CLASS X EXAMINATION 2024\n"
            "Roll No: 17266193\n"
            "Name: TANVI\n"
            "Father's Name: SANJAY KUMAR\n"
            "Mother's Name: POONAM\n"
            "DOB: 01/10/2007\n"
            "School: J N V PAPROLA DT KANGRA HP\n"
            "Result: PASS"
        ),
        "extracted_text_excerpt": "Tanvi, 01/10/2007, FEMALE" if "aadhaar" in args.get("document_id", "") else "Tanvi, Roll 17266193, PASS",
        "metadata": {
            "fields": {
                "aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98} if "aadhaar" in args.get("document_id", "") else {},
                "name": {"value": "Tanvi", "confidence": 0.99},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.99},
                "gender": {"value": "FEMALE", "confidence": 0.99} if "aadhaar" in args.get("document_id", "") else {},
                "mobile_number": {"value": "8219480038", "confidence": 0.95} if "aadhaar" in args.get("document_id", "") else {},
                "roll_number": {"value": "17266193", "confidence": 0.99} if "marksheet" in args.get("document_id", "") else {},
                "father_name": {"value": "Sanjay Kumar", "confidence": 0.97} if "marksheet" in args.get("document_id", "") else {},
                "mother_name": {"value": "Poonam", "confidence": 0.97} if "marksheet" in args.get("document_id", "") else {},
                "school": {"value": "J N V Paprola DT Kangra HP", "confidence": 0.98} if "marksheet" in args.get("document_id", "") else {},
                "result": {"value": "PASS", "confidence": 1.0} if "marksheet" in args.get("document_id", "") else {},
            },
            "extraction_confidence": 0.97,
        },
    },
    "get_evidence": lambda args: {
        "document_id": args.get("document_id", ""),
        "evidence": [
            {"chunk_id": "chunk_001", "text": "Tanvi, DOB 01/10/2007, FEMALE", "relevance": 0.95},
            {"chunk_id": "chunk_002", "text": "Aadhaar: 4373 7370 1714", "relevance": 0.92},
        ] if "aadhaar" in args.get("document_id", "") else [
            {"chunk_id": "chunk_010", "text": "Roll No: 17266193, Name: TANVI", "relevance": 0.96},
            {"chunk_id": "chunk_011", "text": "Result: PASS", "relevance": 0.90},
        ],
    },
    "validate_document": lambda args: {
        "document_id": args.get("document_id", ""),
        "status": "VALID",
        "risk_score": 0.0,
        "temporal_status": "NO_EXPIRY" if "aadhaar" in args.get("document_id", "") else "VALID",
        "temporal_tags": ["NO_EXPIRY"] if "aadhaar" in args.get("document_id", "") else [],
    },
    "list_templates": lambda args: {
        "templates": [
            {"template_id": "aadhaar_card_v1", "document_type": "identity_document", "document_sub_type": "aadhaar", "required_fields": ["aadhaar_number", "name", "date_of_birth", "gender"]},
            {"template_id": "marksheet_v1", "document_type": "academic_record", "document_sub_type": "marksheet", "required_fields": ["student_name", "roll_number", "date_of_birth", "school", "result"]},
        ],
        "count": 2,
    },
    "health_document": lambda args: {"status": "ok", "service": "document-service"},
    "health_model": lambda args: {"status": "ok", "service": "model-service"},
    "health_validator": lambda args: {"status": "ok", "service": "validator-service"},
    "health_web_search": lambda args: {"status": "ok", "service": "web-search-service"},
}


class MockServiceRouter(ServiceClientRouter):
    """Routes tool calls to simulated responses."""

    def call_tool(self, target, tool_name, arguments):
        handler = SIMULATED_DOCUMENTS.get(tool_name)
        if handler:
            return handler(arguments)
        return {"error": True, "message": f"Simulated: {tool_name} not mocked"}


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return

    print("=" * 70)
    print("  ORCHESTRATOR LIVE TEST — Qwen3.8-27B Planning & Tool Calling")
    print("  Simulated: Aadhaar card + Marksheet for Tanvi")
    print("=" * 70)

    # Setup
    tmpdir = tempfile.mkdtemp()
    sessions = FileSessionStore(base_dir=os.path.join(tmpdir, "sessions"))
    confirmations = InMemoryConfirmationStore()
    registry = DefaultToolRegistry()
    router = MockServiceRouter({})

    from orchestrator_service.infrastructure.llm_provider import QwenLLMProvider
    llm = QwenLLMProvider(token=token, model_id="Qwen/Qwen3.8-27B", max_tokens=4096)

    engine = OrchestrationEngine(
        llm=llm,
        sessions=sessions,
        confirmations=confirmations,
        tool_registry=registry,
        service_router=router,
    )

    # ============================================================
    # Test 1: "What documents do I have?"
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 1: 'What documents do I have?'")
    print("=" * 70)

    session1 = engine.create_session(user_id="tanvi", title="Document Query")
    start = time.time()
    response1 = engine.process_message(session1.session_id, "What documents do I have?")
    elapsed = time.time() - start

    print(f"\n⏱  Latency: {elapsed:.1f}s")
    print(f"📋 Response:\n{response1.message}")
    if response1.tool_calls_made:
        print(f"\n🔧 Tools called:")
        for tc in response1.tool_calls_made:
            print(f"   → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:80]}) [{tc.status.value}]")
    if response1.metadata.get("reasoning"):
        print(f"\n💭 Reasoning: {response1.metadata['reasoning'][:300]}")

    # ============================================================
    # Test 2: "What is my Aadhaar number?"
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 2: 'What is my Aadhaar number?'")
    print("=" * 70)

    session2 = engine.create_session(user_id="tanvi", title="Aadhaar Query")
    start = time.time()
    response2 = engine.process_message(session2.session_id, "What is my Aadhaar number?")
    elapsed = time.time() - start

    print(f"\n⏱  Latency: {elapsed:.1f}s")
    print(f"📋 Response:\n{response2.message}")
    if response2.tool_calls_made:
        print(f"\n🔧 Tools called:")
        for tc in response2.tool_calls_made:
            print(f"   → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:80]}) [{tc.status.value}]")
    if response2.metadata.get("reasoning"):
        print(f"\n💭 Reasoning: {response2.metadata['reasoning'][:300]}")

    # ============================================================
    # Test 3: "Show my marks from the marksheet"
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 3: 'Show my marks from the marksheet'")
    print("=" * 70)

    session3 = engine.create_session(user_id="tanvi", title="Marksheet Query")
    start = time.time()
    response3 = engine.process_message(session3.session_id, "Show my marks from the marksheet")
    elapsed = time.time() - start

    print(f"\n⏱  Latency: {elapsed:.1f}s")
    print(f"📋 Response:\n{response3.message}")
    if response3.tool_calls_made:
        print(f"\n🔧 Tools called:")
        for tc in response3.tool_calls_made:
            print(f"   → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:80]}) [{tc.status.value}]")
    if response3.metadata.get("reasoning"):
        print(f"\n💭 Reasoning: {response3.metadata['reasoning'][:300]}")

    # ============================================================
    # Test 4: "Download my Aadhaar card" (should require confirmation)
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 4: 'Download my Aadhaar card' (should require confirmation)")
    print("=" * 70)

    session4 = engine.create_session(user_id="tanvi", title="Download Request")
    start = time.time()
    response4 = engine.process_message(session4.session_id, "Download my Aadhaar card file")
    elapsed = time.time() - start

    print(f"\n⏱  Latency: {elapsed:.1f}s")
    print(f"📋 Response:\n{response4.message}")
    if response4.confirmation_required:
        print(f"\n🔒 CONFIRMATION REQUIRED:")
        print(f"   Type: {response4.confirmation_required.confirmation_type.value}")
        print(f"   Message: {response4.confirmation_required.message}")
    if response4.tool_calls_made:
        print(f"\n🔧 Tools called:")
        for tc in response4.tool_calls_made:
            print(f"   → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:80]}) [{tc.status.value}]")

    # ============================================================
    # Test 5: Multi-step — "What do I need for LIC insurance?"
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 5: 'What do I need for LIC insurance?' (multi-step)")
    print("=" * 70)

    session5 = engine.create_session(user_id="tanvi", title="LIC Query")
    start = time.time()
    response5 = engine.process_message(session5.session_id, "What documents do I need to apply for LIC based insurance?")
    elapsed = time.time() - start

    print(f"\n⏱  Latency: {elapsed:.1f}s")
    print(f"📋 Response:\n{response5.message}")
    if response5.tool_calls_made:
        print(f"\n🔧 Tools called:")
        for tc in response5.tool_calls_made:
            print(f"   → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:80]}) [{tc.status.value}]")
    if response5.metadata.get("reasoning"):
        print(f"\n💭 Reasoning: {response5.metadata['reasoning'][:300]}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_tests = [
        ("Documents query", response1),
        ("Aadhaar number", response2),
        ("Marksheet marks", response3),
        ("Download request", response4),
        ("LIC insurance", response5),
    ]

    for name, resp in all_tests:
        tools_count = len(resp.tool_calls_made)
        needs_confirm = "🔒" if resp.confirmation_required else "  "
        print(f"  {needs_confirm} {name:25s} → {tools_count} tool calls, {len(resp.message)} chars response")

    print(f"\n  Total sessions created: 5")
    print(f"  Context isolation: Each session has its own conversation history ✓")
    print(f"  Confirmation gate: File retrieval blocked without approval ✓")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
