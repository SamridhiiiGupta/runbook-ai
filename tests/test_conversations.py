from runbook_ai.conversations import ConversationStore


def test_conversation_history_persists_and_clear_resets_title(tmp_path):
    path = tmp_path / "chat.sqlite3"
    store = ConversationStore(path)
    conversation_id = store.create()
    store.add_message(conversation_id, "user", "How do I install the VPN client?")
    store.add_message(conversation_id, "assistant", "Install AcmeConnect [1].", grounded=True)

    reopened = ConversationStore(path)
    assert reopened.list_conversations()[0]["title"] == "How do I install the VPN client?"
    messages = reopened.get_messages(conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["grounded"] is True

    reopened.clear(conversation_id)
    assert reopened.get_messages(conversation_id) == []
    assert reopened.list_conversations()[0]["title"] == "New conversation"


def test_delete_cascades_to_messages(tmp_path):
    store = ConversationStore(tmp_path / "chat.sqlite3")
    conversation_id = store.create()
    store.add_message(conversation_id, "user", "Question")
    store.delete(conversation_id)
    assert store.list_conversations() == []
    assert store.get_messages(conversation_id) == []
