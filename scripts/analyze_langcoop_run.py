#!/usr/bin/env python3
"""Analyze a copied LangCoop run directory and summarize what happened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VEHICLE_TOKENS = ("vehicle", "car", "truck", "bus", "van", "motorcycle")
WALKER_TOKENS = ("pedestrian", "walker", "person", "people")
SIDE_TOKENS = ("left", "right", "center")


def _contains_any(text: str, tokens) -> bool:
    text = (text or "").lower()
    return any(token in text for token in tokens)


def compute_ground_truth_perception(records):
    if not records:
        return {}

    gt_records = [record for record in records if isinstance(record.get("gt_dynamic_perception"), dict)]
    if not gt_records:
        return {}

    any_gt_frames = 0
    vehicle_gt_frames = 0
    walker_gt_frames = 0
    any_dynamic_mentions = 0
    vehicle_mentions = 0
    walker_mentions = 0
    no_gt_frames = 0
    false_positive_frames = 0
    side_frames = 0
    side_hits = 0

    for record in gt_records:
        gt = record.get("gt_dynamic_perception", {}) or {}
        visible_vehicle_count = int(gt.get("visible_vehicle_count", 0) or 0)
        visible_walker_count = int(gt.get("visible_walker_count", 0) or 0)
        visible_actor_count = int(gt.get("visible_actor_count", 0) or 0)
        dominant_side = str(gt.get("dominant_side", "none"))

        model_text = " ".join(
            str(record.get(key, "") or "")
            for key in ("planner_objects_full", "planner_scene_full", "planner_objects_excerpt", "planner_scene_excerpt")
        ).lower()

        model_mentions_vehicle = _contains_any(model_text, VEHICLE_TOKENS)
        model_mentions_walker = _contains_any(model_text, WALKER_TOKENS)
        model_mentions_dynamic = model_mentions_vehicle or model_mentions_walker

        if visible_actor_count > 0:
            any_gt_frames += 1
            any_dynamic_mentions += int(model_mentions_dynamic)
        else:
            no_gt_frames += 1
            false_positive_frames += int(model_mentions_dynamic)

        if visible_vehicle_count > 0:
            vehicle_gt_frames += 1
            vehicle_mentions += int(model_mentions_vehicle)

        if visible_walker_count > 0:
            walker_gt_frames += 1
            walker_mentions += int(model_mentions_walker)

        if visible_actor_count > 0 and dominant_side in SIDE_TOKENS:
            side_frames += 1
            side_hits += int(dominant_side in model_text)

    metrics = {
        "gt_sampled_frames": len(gt_records),
        "any_dynamic_recall": any_dynamic_mentions / max(any_gt_frames, 1),
        "vehicle_recall": vehicle_mentions / max(vehicle_gt_frames, 1),
        "walker_recall": walker_mentions / max(walker_gt_frames, 1),
        "side_awareness_rate": side_hits / max(side_frames, 1),
        "dynamic_false_positive_rate": false_positive_frames / max(no_gt_frames, 1),
        "frames_with_any_gt_actor": any_gt_frames,
        "frames_with_gt_vehicle": vehicle_gt_frames,
        "frames_with_gt_walker": walker_gt_frames,
        "frames_without_gt_actor": no_gt_frames,
    }

    weighted_terms = []
    weighted_terms.append((0.40, metrics["any_dynamic_recall"]))
    weighted_terms.append((0.25, metrics["vehicle_recall"]))
    if walker_gt_frames > 0:
        weighted_terms.append((0.15, metrics["walker_recall"]))
    weighted_terms.append((0.10, metrics["side_awareness_rate"]))
    weighted_terms.append((0.10, 1.0 - metrics["dynamic_false_positive_rate"]))

    total_weight = sum(weight for weight, _ in weighted_terms) or 1.0
    gt_score = sum(weight * value for weight, value in weighted_terms) / total_weight
    metrics["ground_truth_perception_score"] = gt_score * 100.0
    return metrics


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def summarize_trace(records):
    if not records:
        return {
            "num_records": 0,
            "final": {},
        }

    final = records[-1]
    first_collision = next((record for record in records if int(record.get("collisions", 0)) > 0), None)
    speeds = [float(record.get("speed_mps", 0.0)) for record in records]
    steers = [float(record.get("steer", 0.0)) for record in records]
    target_speeds = sorted({round(float(record.get("target_speed_mps", 0.0)), 3) for record in records})
    hazard_records = [record for record in records if bool(record.get("static_hazard_detected", False))]
    proximity_records = [record for record in records if bool(record.get("hazard_proximity_detected", False))]
    guard_records = [record for record in records if bool(record.get("static_guard_applied", False))]
    wall_records = [record for record in records if bool(record.get("wall_barrier_mentioned", False))]
    low_steer_hazard_records = [
        record for record in hazard_records if abs(float(record.get("steer", 0.0))) < 0.01
    ]
    hazard_action_gap_records = [
        record
        for record in records
        if bool(record.get("static_hazard_detected", False))
        and bool(record.get("hazard_proximity_detected", False))
        and not bool(record.get("static_guard_applied", False))
        and abs(float(record.get("steer", 0.0))) < 0.01
    ]
    hazard_side_counts = {}
    for record in records:
        side = str(record.get("hazard_side", "none"))
        hazard_side_counts[side] = hazard_side_counts.get(side, 0) + 1

    sampled = max(len(records), 1)
    perception_proxy = {
        "hazard_detection_rate": len(hazard_records) / sampled,
        "hazard_proximity_rate": len(proximity_records) / sampled,
        "wall_barrier_mention_rate": len(wall_records) / sampled,
        "guard_application_rate": len(guard_records) / sampled,
        "low_steer_when_hazard_rate": (len(low_steer_hazard_records) / max(len(hazard_records), 1)),
        "hazard_action_gap_rate": (len(hazard_action_gap_records) / max(len(hazard_records), 1)),
        "hazard_side_counts": hazard_side_counts,
    }

    perception_score_proxy = (
        0.35 * perception_proxy["hazard_detection_rate"]
        + 0.25 * perception_proxy["hazard_proximity_rate"]
        + 0.20 * perception_proxy["wall_barrier_mention_rate"]
        + 0.20 * (1.0 - perception_proxy["hazard_action_gap_rate"])
    ) * 100.0

    return {
        "num_records": len(records),
        "final": final,
        "first_collision_step": first_collision.get("step") if first_collision else None,
        "first_collision_speed_mps": first_collision.get("speed_mps") if first_collision else None,
        "max_collisions": max(int(record.get("collisions", 0)) for record in records),
        "max_rc": max(float(record.get("rc", 0.0)) for record in records),
        "mean_speed_mps": sum(speeds) / max(len(speeds), 1),
        "min_speed_mps": min(speeds),
        "max_speed_mps": max(speeds),
        "min_steer": min(steers),
        "max_steer": max(steers),
        "target_speed_values": target_speeds,
        "perception_proxy": perception_proxy,
        "perception_score_proxy": perception_score_proxy,
        "ground_truth_perception": compute_ground_truth_perception(records),
    }


def classify_run(summary):
    final = summary.get("final", {})
    final_rc = float(final.get("rc", 0.0))
    final_ds = float(final.get("ds", 0.0))
    final_collisions = int(final.get("collisions", 0))
    mean_speed = float(summary.get("mean_speed_mps", 0.0))
    max_abs_steer = max(abs(float(summary.get("min_steer", 0.0))), abs(float(summary.get("max_steer", 0.0))))
    perception_proxy = summary.get("perception_proxy", {})
    hazard_action_gap_rate = float(perception_proxy.get("hazard_action_gap_rate", 0.0))
    guard_application_rate = float(perception_proxy.get("guard_application_rate", 0.0))
    gt_perception = summary.get("ground_truth_perception", {})
    gt_score = float(gt_perception.get("ground_truth_perception_score", 0.0))

    notes = []
    if final_rc < 5.0 and final_collisions >= 3:
        notes.append("stuck_with_repeated_collisions")
    if mean_speed < 0.5 and final_rc < 10.0:
        notes.append("near_stationary")
    if max_abs_steer < 0.05 and final_rc < 10.0:
        notes.append("very_low_steering_authority")
    if final_ds <= 1.0 and final_collisions > 0:
        notes.append("driving_score_collapsed_after_collisions")
    if hazard_action_gap_rate > 0.6:
        notes.append("hazard_seen_but_not_acted_on")
    if guard_application_rate == 0.0 and float(perception_proxy.get("hazard_detection_rate", 0.0)) > 0.5:
        notes.append("safety_guard_never_triggered")
    if gt_perception and gt_score < 45.0:
        notes.append("low_ground_truth_perception")
    if not notes:
        notes.append("no_obvious_failure_signature")
    return notes


def analyze_run(run_dir: Path):
    report = {
        "run_dir": str(run_dir),
        "metrics_json": None,
        "agents": {},
    }

    metrics_path = run_dir / "metrics.json"
    metrics = load_json(metrics_path)
    if metrics is not None:
        report["metrics_json"] = metrics

    live_trace_root = run_dir / "live_traces"
    if live_trace_root.exists():
        for scenario_dir in sorted(path for path in live_trace_root.iterdir() if path.is_dir()):
            for trace_path in sorted(scenario_dir.glob("*_trace.jsonl")):
                route_id = trace_path.stem.replace("_trace", "")
                records = load_jsonl(trace_path)
                summary = summarize_trace(records)
                summary["classification"] = classify_run(summary)
                report["agents"][route_id] = summary

    image_root = run_dir / "images"
    if image_root.exists():
        for scenario_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            for summary_path in sorted(scenario_dir.glob("agent_*/hazard_debug_summary.json")):
                agent_key = f"{scenario_dir.name}_{summary_path.parent.name}"
                summary_data = load_json(summary_path)
                if summary_data is not None:
                    report["agents"].setdefault(agent_key, {})
                    report["agents"][agent_key]["hazard_debug_summary"] = summary_data

    return report


def render_text_report(report):
    lines = []
    lines.append(f"Run: {report['run_dir']}")
    lines.append("")

    if not report["agents"]:
        lines.append("No agent traces found.")
        return "\n".join(lines)

    for agent_id, summary in sorted(report["agents"].items()):
        final = summary.get("final", {})
        lines.append(f"[{agent_id}]")
        lines.append(
            "  Final: "
            f"RC={float(final.get('rc', 0.0)):.2f}% "
            f"DS={float(final.get('ds', 0.0)):.2f} "
            f"Collisions={int(final.get('collisions', 0))} "
            f"Violations={int(final.get('violations', 0))}"
        )
        if "mean_speed_mps" in summary:
            lines.append(
                "  Motion: "
                f"mean_speed={float(summary['mean_speed_mps']):.3f} "
                f"max_speed={float(summary['max_speed_mps']):.3f} "
                f"steer_range=[{float(summary['min_steer']):+.4f}, {float(summary['max_steer']):+.4f}]"
            )
            lines.append(
                "  Events: "
                f"first_collision_step={summary.get('first_collision_step')} "
                f"target_speeds={summary.get('target_speed_values', [])}"
            )
            lines.append(
                "  Classification: "
                + ", ".join(summary.get("classification", []))
            )
        if "perception_score_proxy" in summary:
            proxy = summary.get("perception_proxy", {})
            lines.append(
                "  Perception Proxy: "
                f"score={float(summary.get('perception_score_proxy', 0.0)):.1f}/100 "
                f"hazard_rate={float(proxy.get('hazard_detection_rate', 0.0)):.2f} "
                f"proximity_rate={float(proxy.get('hazard_proximity_rate', 0.0)):.2f} "
                f"action_gap={float(proxy.get('hazard_action_gap_rate', 0.0)):.2f}"
            )
        gt = summary.get("ground_truth_perception", {})
        if gt:
            lines.append(
                "  GT Perception: "
                f"score={float(gt.get('ground_truth_perception_score', 0.0)):.1f}/100 "
                f"any_recall={float(gt.get('any_dynamic_recall', 0.0)):.2f} "
                f"vehicle_recall={float(gt.get('vehicle_recall', 0.0)):.2f} "
                f"walker_recall={float(gt.get('walker_recall', 0.0)):.2f} "
                f"side_rate={float(gt.get('side_awareness_rate', 0.0)):.2f} "
                f"fp_rate={float(gt.get('dynamic_false_positive_rate', 0.0)):.2f}"
            )
        hazard_summary = summary.get("hazard_debug_summary")
        if hazard_summary:
            lines.append(
                "  Hazard Debug: "
                f"static_rate={hazard_summary.get('hazard_recall_indicators', {}).get('static_hazard_detection_rate')} "
                f"guard_rate={hazard_summary.get('hazard_recall_indicators', {}).get('guard_application_rate')}"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a copied LangCoop run directory")
    parser.add_argument("--run-dir", required=True, help="Local run directory to analyze")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    parser.add_argument("--output-text", default=None, help="Optional output text report path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    report = analyze_run(run_dir)
    text_report = render_text_report(report)
    print(text_report)

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))

    if args.output_text:
        output_text = Path(args.output_text).expanduser().resolve()
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(text_report)


if __name__ == "__main__":
    main()
