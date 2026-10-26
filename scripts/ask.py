import sys

from runbook_ai.answer import answer_question
from runbook_ai.chunker import load_runbooks
from runbook_ai.search import Retriever

question = " ".join(sys.argv[1:])
retriever = Retriever(load_runbooks("data/runbooks"))
result = answer_question(question, retriever)

print(result.text)
print("\nSources:")
for s in result.sources:
    print(f"- {s.source} | {s.section} ({s.score:.2f})")