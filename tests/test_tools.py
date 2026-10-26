import pytest

from runbook_ai.tools import ToolInputError, execute_tool, safe_calculate


def test_calculator_handles_basic_arithmetic():
    assert safe_calculate("(250 * 18) / 100") == 45


@pytest.mark.parametrize("expression", [
    "__import__('os').system('whoami')",
    "9**9**9",
    "1 / 0",
    "9 * 10**20",
    "'hello'",
])
def test_calculator_rejects_code_and_unbounded_or_invalid_math(expression):
    with pytest.raises(ToolInputError):
        safe_calculate(expression)


def test_only_allowlisted_tools_can_run():
    assert execute_tool("run_shell", {"command": "whoami"}) == {"error": "Tool is not allowlisted."}


def test_ticket_creation_is_only_a_local_unsent_draft():
    result = execute_tool("create_ticket_draft", {
        "title": "Printer unavailable",
        "description": "The office printer is not listed.",
        "category": "hardware",
    })
    assert result["ticket_id"].startswith("RB-")
    assert result["status"] == "Draft - not submitted"


def test_search_tool_validates_limits_and_returns_relevant_passages():
    class FakeRetriever:
        def search(self, query, k=3):
            assert query == "VPN error 809"
            assert k == 1
            return [(type("C", (), {
                "source": "vpn_access.md", "section": "VPN troubleshooting",
                "page": 2, "text": "Try a mobile hotspot.",
            })(), .91)]

    result = execute_tool("search_docs", {"query": "VPN error 809", "top_k": 1}, FakeRetriever())
    assert result["matches"][0]["source"] == "vpn_access.md"
    assert result["matches"][0]["page"] == 2
    assert result["matches"][0]["similarity"] == .91


def test_search_tool_rejects_bad_top_k():
    with pytest.raises(ToolInputError):
        execute_tool("search_docs", {"query": "printer", "top_k": 100}, object())
