from runbook_ai.evaluate import _contains_expected, retrieval_hit


def test_retrieval_hit_supports_answerable_and_unanswerable_cases():
    assert retrieval_hit({"vpn_access.md"}, {"vpn_access.md"})
    assert not retrieval_hit({"printer_troubleshooting.md"}, {"vpn_access.md"})
    assert retrieval_hit(set(), set())
    assert not retrieval_hit({"password_reset.md"}, set())


def test_expected_answer_terms_are_case_and_whitespace_insensitive():
    assert _contains_expected("Use AcmeConnect to connect.", ["acmeconnect"])
    assert _contains_expected("Use 12 characters at minimum.", ["12   characters"])
    assert _contains_expected("anything", []) is None
