"""Evaluation metrics: BERTScore, ROUGE, TEDS, and accuracy wrappers."""

import numpy as np


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> dict[str, float]:
    """Compute BERTScore precision, recall, and F1.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        lang: Language code for BERTScore.

    Returns:
        Dict with "precision", "recall", "f1" keys (mean values).

    """
    try:
        from bert_score import score
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p, r, f1 = score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": p.mean().item(),
        "recall": r.mean().item(),
        "f1": f1.mean().item(),
    }


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Dict with "rouge1", "rouge2", "rougeL" keys (F1 scores).

    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references, strict=False):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}


def compute_teds(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute Tree Edit Distance Similarity for table structures.

    Args:
        predictions: List of predicted Markdown table strings.
        references: List of reference Markdown table strings.

    Returns:
        Mean TEDS score across all pairs.

    """
    try:
        from apted import APTED, Config
        from lxml import etree
    except ImportError:
        return 0.0

    def table_to_tree(table_str: str) -> etree._Element:
        """Convert a Markdown table string to an lxml Element tree."""
        root = etree.Element("table")
        rows = [
            [cell.strip() for cell in row.split(",")]
            for row in table_str.strip().split(";")
        ]
        for row_cells in rows:
            tr = etree.SubElement(root, "tr")
            for cell in row_cells:
                td = etree.SubElement(tr, "td")
                td.text = cell
        return root

    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        try:
            pred_tree = table_to_tree(pred)
            ref_tree = table_to_tree(ref)
            ted = APTED(pred_tree, ref_tree, Config())
            distance = ted.compute_edit_distance()
            max_size = max(pred_tree.xpath("count(//*)"), ref_tree.xpath("count(//*)"))
            teds = 1.0 - (distance / max_size) if max_size > 0 else 1.0
            scores.append(teds)
        except Exception:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def compute_accuracy(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute exact-match accuracy.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Accuracy as a float between 0 and 1.

    """
    if not predictions:
        return 0.0
    correct = sum(
        1
        for p, r in zip(predictions, references, strict=False)
        if p.strip() == r.strip()
    )
    return correct / len(predictions)


def compute_set_f1(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute set-based Precision, Recall, F1 for unordered list answers.

    Args:
        predictions: List of predicted comma-separated strings.
        references: List of reference comma-separated strings.

    Returns:
        Dict with "precision", "recall", "f1" keys.

    """
    precisions = []
    recalls = []
    f1s = []

    for pred, ref in zip(predictions, references, strict=False):
        pred_set = set(item.strip() for item in pred.split(",") if item.strip())
        ref_set = set(item.strip() for item in ref.split(",") if item.strip())

        if not pred_set and not ref_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        tp = len(pred_set & ref_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(ref_set) if ref_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
