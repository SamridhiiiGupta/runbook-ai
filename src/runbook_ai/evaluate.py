"""A small golden-question runner; reports measured local results only."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .agent import stream_agent
from .chunker import load_runbooks
from .search import MIN_RELEVANCE, Retriever


def read_golden(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from exc
        required = {"id", "question", "expected_sources", "expected_tools", "category"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Golden question on line {line_number} is missing {sorted(missing)}.")
        rows.append(row)
    return rows


def retrieval_hit(actual_sources: set[str], expected_sources: set[str]) -> bool:
    if expected_sources:
        return bool(actual_sources & expected_sources)
    return not actual_sources


def _contains_expected(text: str, phrases: list[str]) -> bool | None:
    if not phrases:
        return None
    normalized = " ".join(text.casefold().split())
    return all(" ".join(phrase.casefold().split()) in normalized for phrase in phrases)


def _percent(values):
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return round(100 * sum(bool(value) for value in measured) / len(measured), 1)


def _render_markdown(report: dict) -> str:
    lines = [
        "# Runbook AI evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Mode: `{report['mode']}` · Questions: {report['question_count']}",
        "",
        "Scores are computed from this run only. They are not a guarantee of answer quality.",
        "",
        "| ID | Category | Retrieval hit@5 | Tools exact | Refusal correct | Answer terms | Latency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        flag = lambda value: "—" if value is None else ("yes" if value else "no")
        latency = "—" if row["latency_s"] is None else f"{row['latency_s']:.2f}s"
        lines.append(
            f"| {row['id']} | {row['category']} | {flag(row['retrieval_hit_at_5'])} | "
            f"{flag(row['tools_exact'])} | {flag(row['refusal_correct'])} | "
            f"{flag(row['answer_terms_match'])} | {latency} |"
        )
    summary = report["summary"]
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Retrieval hit@5: {summary['retrieval_hit_at_5']}%",
        f"- Exact tool selection: {summary['tools_exact']}%",
        f"- Refusal accuracy: {summary['refusal_accuracy']}%",
        f"- Expected answer terms: {summary['answer_terms_match']}%",
        f"- Median latency: {summary['median_latency_s']}s",
        "",
        "This lightweight harness checks retrieval, exact tool choice, refusal behavior, "
        "and manually selected answer terms. It does not use an LLM judge and does not "
        "measure semantic correctness or claim general accuracy.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(mode="agent", limit=None, dataset=Path("evals/golden.jsonl"), output_dir=Path("evals/reports")):
    if mode not in {"retrieval", "agent"}:
        raise ValueError("Mode must be 'retrieval' or 'agent'.")
    rows = read_golden(Path(dataset))
    if limit is not None:
        rows = rows[:max(0, limit)]
    if not rows:
        raise ValueError("The golden question set is empty.")

    retriever = Retriever.load("data/index")
    if not retriever.sources():
        retriever.add_chunks(load_runbooks("data/runbooks"))
    results = []
    for row in rows:
        expected_sources = set(row["expected_sources"])
        hits = retriever.search(row["question"], k=5)
        retrieved_sources = {chunk.source for chunk, score in hits if score >= MIN_RELEVANCE}
        result = {
            "id": row["id"],
            "category": row["category"],
            "retrieval_hit_at_5": retrieval_hit(retrieved_sources, expected_sources),
            "retrieved_sources": sorted(retrieved_sources),
            "expected_sources": sorted(expected_sources),
            "tools_exact": None,
            "refusal_correct": None,
            "answer_terms_match": None,
            "latency_s": None,
            "answer": None,
            "tools_used": [],
        }
        if mode == "agent":
            completion = None
            for event in stream_agent(row["question"], retriever):
                if event["type"] == "complete":
                    completion = event["result"]
            if completion is None:
                raise RuntimeError(f"Agent did not return a result for {row['id']}.")
            result.update({
                "tools_exact": completion["tools_used"] == row["expected_tools"],
                "answer_terms_match": _contains_expected(
                    completion["text"], row.get("expected_answer_contains", [])
                ),
                "answer": completion["text"],
                "tools_used": completion["tools_used"],
                "latency_s": completion["latency_s"],
                "cited_sources": sorted({source["source"] for source in completion["sources"]}),
            })
            if row["category"] == "unanswerable":
                refused = completion["kind"] == "no_answer"
                result["refusal_correct"] = refused
        results.append(result)

    latencies = sorted(row["latency_s"] for row in results if row["latency_s"] is not None)
    median_latency = None if not latencies else round(latencies[(len(latencies) - 1) // 2], 2)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "question_count": len(results),
        "summary": {
            "retrieval_hit_at_5": _percent([row["retrieval_hit_at_5"] for row in results]),
            "tools_exact": _percent([row["tools_exact"] for row in results]),
            "refusal_accuracy": _percent([row["refusal_correct"] for row in results]),
            "answer_terms_match": _percent([row["answer_terms_match"] for row in results]),
            "median_latency_s": median_latency,
        },
        "results": results,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "latest.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("retrieval", "agent"), default="agent")
    parser.add_argument("--limit", type=int, help="Run only the first N questions.")
    parser.add_argument("--dataset", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    args = parser.parse_args()
    report = run_evaluation(args.mode, args.limit, args.dataset, args.output_dir)
    print(_render_markdown(report))
    print(f"\nSaved reports to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
