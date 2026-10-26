from runbook_ai.answer import NO_ANSWER, answer_question, build_messages
from runbook_ai.chunker import Chunk

CHUNK = Chunk("vpn.md", "Connect", "Open AcmeConnect and sign in.")


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=3):
        return self.hits


def test_refuses_when_nothing_relevant():
    calls = []
    retriever = FakeRetriever([(CHUNK, 0.1)])  # score below the minimum
    result = answer_question("anything", retriever, chat_fn=lambda m: calls.append(m))
    assert result.text == NO_ANSWER
    assert result.grounded is False
    assert calls == []  # the model was never called


def test_returns_answer_with_sources():
    retriever = FakeRetriever([(CHUNK, 0.9)])
    result = answer_question("how to connect", retriever, chat_fn=lambda m: "Do X [1]")
    assert result.text == "Do X [1]"
    assert result.grounded is True
    assert result.sources[0].source == "vpn.md"


def test_prompt_contains_numbered_context():
    messages = build_messages("question?", [(CHUNK, 0.9)])
    assert "[1] (vpn.md - Connect)" in messages[0]["content"]
    assert messages[1]["content"] == "question?"


def test_empty_question_is_handled():
    result = answer_question("   ", FakeRetriever([]), chat_fn=lambda m: "x")
    assert result.grounded is False

def test_model_refusal_hides_sources():
    retriever = FakeRetriever([(CHUNK, 0.9)])
    result = answer_question(
        "slack password", retriever,
        chat_fn=lambda m: "I don't know based on the IT guides.",
    )
    assert result.grounded is False
    assert result.sources == []