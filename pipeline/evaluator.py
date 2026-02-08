"""Accuracy evaluation for navigation ground truth.

Compares pipeline-generated navigation state labels against manual annotations.
"""
from __future__ import annotations

from collections import defaultdict

from core.types import NavigationTimeline


def _get_state_at_frame(timeline: NavigationTimeline, object_name: str, frame_idx: int) -> str | None:
    """Look up the predicted state of an object at a given frame."""
    object_name_lower = object_name.lower()
    for obj in timeline.objects:
        if object_name_lower in obj.name.lower() or obj.name.lower() in object_name_lower:
            for entry in obj.state_timeline:
                if entry.frame_start <= frame_idx <= entry.frame_end:
                    return entry.state
    return None


def evaluate_nav_accuracy(
    nav_timeline: NavigationTimeline,
    manual_labels: dict[int, dict[str, str]],
) -> dict:
    """Compare pipeline nav state predictions against manual labels.

    Args:
        nav_timeline: Pipeline-generated NavigationTimeline
        manual_labels: {frame_idx: {"object_name": "state"}}
            e.g. {0: {"door": "closed"}, 150: {"door": "open"}}

    Returns:
        Dict with accuracy metrics: total, correct, accuracy, per_state,
        per_object, confusion_matrix, details
    """
    total = 0
    correct = 0
    details: list[dict] = []

    # Per-state accuracy
    state_correct: dict[str, int] = defaultdict(int)
    state_total: dict[str, int] = defaultdict(int)

    # Per-object accuracy
    object_correct: dict[str, int] = defaultdict(int)
    object_total: dict[str, int] = defaultdict(int)

    # Confusion matrix: (true_state, predicted_state) -> count
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for frame_idx, frame_labels in sorted(manual_labels.items()):
        for object_name, true_state in frame_labels.items():
            true_state = true_state.lower().strip()
            predicted_state = _get_state_at_frame(nav_timeline, object_name, frame_idx)

            total += 1
            state_total[true_state] += 1
            object_total[object_name] += 1

            is_correct = predicted_state is not None and predicted_state == true_state

            if is_correct:
                correct += 1
                state_correct[true_state] += 1
                object_correct[object_name] += 1

            pred_str = predicted_state if predicted_state else "not_found"
            confusion[(true_state, pred_str)] += 1

            details.append({
                "frame": frame_idx,
                "object": object_name,
                "true_state": true_state,
                "predicted_state": pred_str,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0

    # Per-state metrics
    per_state = {}
    for state in sorted(state_total.keys()):
        st_total = state_total[state]
        st_correct = state_correct.get(state, 0)
        per_state[state] = {
            "total": st_total,
            "correct": st_correct,
            "accuracy": round(st_correct / st_total, 4) if st_total > 0 else 0.0,
        }

    # Per-object metrics
    per_object = {}
    for obj in sorted(object_total.keys()):
        ob_total = object_total[obj]
        ob_correct = object_correct.get(obj, 0)
        per_object[obj] = {
            "total": ob_total,
            "correct": ob_correct,
            "accuracy": round(ob_correct / ob_total, 4) if ob_total > 0 else 0.0,
        }

    # Format confusion matrix
    confusion_matrix = {
        f"{true}->{pred}": count
        for (true, pred), count in sorted(confusion.items())
    }

    return {
        "total_labels": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_state": per_state,
        "per_object": per_object,
        "confusion_matrix": confusion_matrix,
        "details": details,
    }


def parse_manual_labels(labels_json: dict | list) -> dict[int, dict[str, str]]:
    """Parse manual labels from various input formats.

    Accepts either:
    - dict: {frame_idx: {"object_name": "state"}}
    - list: [{"frame": int, "object": str, "state": str}]

    Returns normalized dict format.
    """
    if isinstance(labels_json, list):
        result: dict[int, dict[str, str]] = defaultdict(dict)
        for entry in labels_json:
            frame = int(entry.get("frame", 0))
            obj = str(entry.get("object", "unknown"))
            state = str(entry.get("state", "unknown"))
            result[frame][obj] = state
        return dict(result)
    elif isinstance(labels_json, dict):
        # Already in the right format, but ensure int keys
        return {int(k): v for k, v in labels_json.items()}
    return {}
