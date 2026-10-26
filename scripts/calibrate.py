from runbook_ai.chunker import load_runbooks
from runbook_ai.search import Retriever

IN_SCOPE = [
    "I forgot my login and cannot sign in",
    "My VPN shows error 809",
    "How do I install the VPN client?",
    "printer says not found",
    "my account got locked",
    "how long must my password be",
    "there is paper stuck in the printer",
    "my print job is stuck",
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I reset my Slack password?",
    "Write me a poem about cats",
    "How do I apply for annual leave?",
    "my laptop screen is flickering",
    "what is the office wifi password",
]

retriever = Retriever(load_runbooks("data/runbooks"))

for label, questions in [("IN SCOPE", IN_SCOPE), ("OUT OF SCOPE", OUT_OF_SCOPE)]:
    print(f"\n{label}")
    for q in questions:
        chunk, score = retriever.search(q, k=1)[0]
        print(f"{score:.2f}  {q}  ->  {chunk.source} | {chunk.section}")