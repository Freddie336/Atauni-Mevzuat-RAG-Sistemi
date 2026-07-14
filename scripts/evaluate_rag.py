"""Ragas ile RAG cevaplarını değerlendirip JSON ve CSV raporu üretir."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import load_jsonl, rows  # noqa: E402


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toplanmış RAG cevaplarını Ragas metrikleriyle değerlendirir."
    )
    parser.add_argument("--input", required=True, help="JSONL değerlendirme veri kümesi")
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
        help="JSON ve CSV raporlarının yazılacağı klasör",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RAGAS_EVALUATOR_MODEL", "llama-3.3-70b-versatile"),
        help="Ragas hakem modeli (varsayılan: RAGAS_EVALUATOR_MODEL veya Groq Llama)",
    )
    return parser.parse_args()


def run(input_path: str, output_dir: str, model_name: str) -> dict[str, float]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY ortam değişkeni tanımlı değil")

    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Değerlendirme bağımlılıkları eksik. "
            "Önce `pip install -r requirements-evaluation.txt` çalıştırın."
        ) from exc

    records = load_jsonl(input_path)
    dataset = Dataset.from_list(rows(records))

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model=model_name, temperature=0, groq_api_key=api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=True,
    )
    frame = result.to_pandas()
    summary = {
        metric: float(frame[metric].mean())
        for metric in METRIC_NAMES
        if metric in frame.columns
    }

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "ragas_results.csv", index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(input_path).resolve()),
        "model": model_name,
        "record_count": len(records),
        "metrics": summary,
    }
    with (report_dir / "ragas_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args.input, args.output_dir, args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    print("Ragas değerlendirmesi tamamlandı:")
    for name, value in summary.items():
        print(f"- {name}: {value:.4f}")
    print(f"Rapor klasörü: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
