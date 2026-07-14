"""Değerlendirme veri kümesi için küçük ve bağımsız doğrulama katmanı."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationRecord:
    """Ragas'ın ihtiyaç duyduğu tek bir soru-cevap kaydı."""

    question: str
    answer: str
    contexts: tuple[str, ...]
    ground_truth: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any], line_number: int) -> "EvaluationRecord":
        required = ("question", "answer", "contexts", "ground_truth")
        missing = [field for field in required if field not in value]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Satır {line_number}: eksik alan(lar): {joined}")

        question = _required_text(value["question"], "question", line_number)
        answer = _required_text(value["answer"], "answer", line_number)
        ground_truth = _required_text(value["ground_truth"], "ground_truth", line_number)

        raw_contexts = value["contexts"]
        if not isinstance(raw_contexts, list) or not raw_contexts:
            raise ValueError(f"Satır {line_number}: contexts boş olmayan bir liste olmalı")

        contexts = tuple(
            _required_text(context, f"contexts[{index}]", line_number)
            for index, context in enumerate(raw_contexts)
        )
        return cls(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    def to_legacy_ragas_row(self) -> dict[str, Any]:
        """Ragas evaluate() uyumluluk API'sinin beklediği sütun adlarını döndürür."""

        row = asdict(self)
        row["contexts"] = list(self.contexts)
        return row


def load_jsonl(path: str | Path) -> list[EvaluationRecord]:
    """UTF-8 JSONL dosyasını okur ve anlaşılır hata mesajlarıyla doğrular."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Değerlendirme dosyası bulunamadı: {dataset_path}")

    records: list[EvaluationRecord] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Satır {line_number}: geçersiz JSON ({exc.msg})"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Satır {line_number}: kayıt bir JSON nesnesi olmalı")
            records.append(EvaluationRecord.from_mapping(value, line_number))

    if not records:
        raise ValueError("Değerlendirme dosyasında kayıt bulunamadı")
    return records


def rows(records: Iterable[EvaluationRecord]) -> list[dict[str, Any]]:
    """Doğrulanmış kayıtları Ragas veri satırlarına dönüştürür."""

    return [record.to_legacy_ragas_row() for record in records]


def _required_text(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Satır {line_number}: {field} boş olmayan bir metin olmalı")
    return value.strip()
