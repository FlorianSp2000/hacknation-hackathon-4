from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

import models  # triggers registration
from core.types import DetectionResult, FrameResult, NavigationTimeline
from core.registry import create, get_class, list_models
from core.video import export_annotated_video, get_video_info
from models.vlm.lfm25 import GENERIC_JSON_SCHEMA_PROMPT, NAV_STATE_PROMPT
from pipeline.evaluator import evaluate_nav_accuracy, parse_manual_labels
from pipeline.exporter import export_ground_truth, export_ground_truth_json
from pipeline.interactions import InteractionConfig, InteractionEngine
from pipeline.nav_state import build_nav_timeline
from pipeline.runner import Pipeline
from pipeline.visualizer import (
    draw_all,
    draw_detections,
    draw_hands_and_interactions,
    draw_segmentation,
    draw_temporal_changes,
    draw_tracking,
)


st.set_page_config(layout="wide", page_title="World2Data - Live")
st.title("Live Perception")


def _init_state() -> None:
    defaults = {
        "live_tracker": None,
        "live_segmenter": None,
        "live_detector": None,
        "live_vlm": None,
        "live_trk_name": "bytetrack",
        "live_seg_name": "rf-detr-seg",
        "live_det_name": "yolov8",
        "live_vlm_name": "lfm2.5-vl-mlx",
        "live_loaded": False,
        "live_auto_loaded": False,
        "live_interaction_engine": None,
        "live_recording": False,
        "live_recording_path": None,
        "live_recorded_video": None,
        "live_results": [],
        "live_nav_timeline": None,
        "live_video_info": None,
        "live_exported_video": None,
        "live_prev_playing": False,
        "live_last_analyzed_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _pick_model_size(category: str, name: str, label: str):
    cls = get_class(category, name)
    kwargs = {}
    if hasattr(cls, "VALID_SIZES"):
        sizes = list(cls.VALID_SIZES)
        kwargs["model_size"] = st.selectbox(label, sizes, index=0)
    return kwargs


def _safe_unload(obj):
    if obj is not None:
        obj.unload()


def _load_selected_models(
    *,
    enable_tracker: bool,
    enable_seg: bool,
    enable_det: bool,
    enable_vlm: bool,
    trk_choice: str,
    seg_choice: str,
    det_choice: str,
    vlm_choice: str,
    trk_kwargs: dict,
    seg_kwargs: dict,
    det_kwargs: dict,
) -> None:
    _safe_unload(st.session_state["live_tracker"])
    _safe_unload(st.session_state["live_segmenter"])
    _safe_unload(st.session_state["live_detector"])
    _safe_unload(st.session_state["live_vlm"])

    st.session_state["live_tracker"] = None
    st.session_state["live_segmenter"] = None
    st.session_state["live_detector"] = None
    st.session_state["live_vlm"] = None

    if enable_tracker:
        trk = create("tracker", trk_choice, **trk_kwargs)
        trk.load()
        st.session_state["live_tracker"] = trk
        st.session_state["live_trk_name"] = trk_choice

    if enable_seg:
        seg = create("segmenter", seg_choice, **seg_kwargs)
        seg.load()
        st.session_state["live_segmenter"] = seg
        st.session_state["live_seg_name"] = seg_choice

    if enable_det:
        det = create("detector", det_choice, **det_kwargs)
        det.load()
        st.session_state["live_detector"] = det
        st.session_state["live_det_name"] = det_choice

    if enable_vlm:
        vlm = create("vlm", vlm_choice)
        vlm.load()
        st.session_state["live_vlm"] = vlm
        st.session_state["live_vlm_name"] = vlm_choice

    st.session_state["live_loaded"] = any(
        m is not None
        for m in (
            st.session_state["live_tracker"],
            st.session_state["live_segmenter"],
            st.session_state["live_detector"],
            st.session_state["live_vlm"],
        )
    )


def _resize_max_side(frame, max_side: int):
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame
    scale = float(max_side) / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _overlay_vlm_summary(frame, parsed: dict | None, raw_text: str, latency_s: float | None):
    out = frame
    h, w = out.shape[:2]
    panel_w = min(520, int(w * 0.45))
    x0 = w - panel_w - 10
    y0 = 10
    y1 = min(h - 10, 300)
    cv2.rectangle(out, (x0, y0), (w - 10, y1), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (w - 10, y1), (80, 80, 80), 1)

    lines = []
    header = "LFM state/action"
    if latency_s is not None:
        header += f" ({latency_s * 1000:.0f} ms)"
    lines.append(header)

    if parsed and isinstance(parsed, dict):
        objects = parsed.get("objects", [])
        if isinstance(objects, list):
            for obj in objects[:6]:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "object"))
                state = str(obj.get("state", "unknown"))
                actions = obj.get("robot_actions", [])
                if isinstance(actions, list):
                    act = ", ".join(str(a) for a in actions[:3])
                else:
                    act = str(actions)
                lines.append(f"- {name}: {state}")
                if act:
                    lines.append(f"  actions: {act}")
    elif raw_text:
        lines.append(raw_text[:180])

    y = y0 + 22
    for line in lines[:12]:
        cv2.putText(out, line[:70], (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        y += 18

    return out


def _build_live_prompt(classes_seen: list[str]) -> str:
    classes = ", ".join(sorted(set(classes_seen))[:20]) if classes_seen else "unknown"
    return (
        "Output ONLY valid JSON with this schema: "
        '{"objects":[{"name":"str","state":"str","robot_actions":["str"]}]}. '
        "Focus on door state (open/closed/ajar/locked/unknown) if a door exists. "
        "For each visible object, provide concise robot actions (e.g. pickup, place, open, close, push, pull, sit, switch_on, switch_off). "
        f"Detected classes hint: {classes}."
    )


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = None
        self.segmenter = None
        self.detector = None
        self.vlm = None
        self.interaction_engine = None
        self.seg_prompt = "person"
        self.vlm_interval = 12
        self.inference_max_side = 640
        self._frame_count = 0
        self._last_vlm_raw = ""
        self._last_vlm_parsed = None
        self._last_vlm_latency_s = None

        self._record_lock = threading.Lock()
        self._recording = False
        self._record_path: str | None = None
        self._record_writer: cv2.VideoWriter | None = None
        self._record_fps = 20.0

    @property
    def recording(self) -> bool:
        with self._record_lock:
            return self._recording

    def start_recording(self, path: str, fps: float = 20.0) -> None:
        with self._record_lock:
            self._recording = True
            self._record_path = path
            self._record_fps = fps
            if self._record_writer is not None:
                self._record_writer.release()
                self._record_writer = None

    def stop_recording(self) -> str | None:
        with self._record_lock:
            self._recording = False
            if self._record_writer is not None:
                self._record_writer.release()
                self._record_writer = None
            return self._record_path

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        with self._record_lock:
            if self._recording:
                if self._record_writer is None and self._record_path:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._record_writer = cv2.VideoWriter(self._record_path, fourcc, self._record_fps, (w, h))
                if self._record_writer is not None:
                    self._record_writer.write(img)

        run_img = _resize_max_side(img, int(self.inference_max_side))
        classes_seen = []
        interaction_detection = None

        if self.tracker is not None:
            trk = self.tracker.update(run_img)
            if trk.boxes:
                classes_seen = [b.class_name for b in trk.boxes]
                interaction_detection = DetectionResult(boxes=list(trk.boxes), frame_idx=self._frame_count)
                img = draw_tracking(img, trk)
        elif self.detector is not None:
            det = self.detector.predict(run_img)
            if det.boxes:
                classes_seen = [b.class_name for b in det.boxes]
                interaction_detection = det
                img = draw_detections(img, det)

        if self.segmenter is not None:
            seg = self.segmenter.predict(run_img, text_prompt=self.seg_prompt)
            if seg.masks:
                img = draw_segmentation(img, seg)

        if self.vlm is not None and (self._frame_count % max(1, int(self.vlm_interval)) == 0):
            prompt = _build_live_prompt(classes_seen)
            t0 = time.perf_counter()
            vlm_result = self.vlm.predict(run_img, prompt)
            self._last_vlm_latency_s = time.perf_counter() - t0
            self._last_vlm_raw = vlm_result.raw_text
            self._last_vlm_parsed = vlm_result.parsed

        if self.vlm is not None:
            img = _overlay_vlm_summary(img, self._last_vlm_parsed, self._last_vlm_raw, self._last_vlm_latency_s)

        if self.interaction_engine is not None and self.interaction_engine.available:
            hand_poses, interactions = self.interaction_engine.process(run_img, interaction_detection)
            if hand_poses or interactions:
                temp_result = FrameResult(frame_idx=self._frame_count, frame=img, hand_poses=hand_poses, interactions=interactions)
                img = draw_hands_and_interactions(img, temp_result)

        rec_text = "REC" if self.recording else "LIVE"
        rec_color = (0, 0, 255) if self.recording else (0, 255, 0)
        cv2.putText(img, rec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rec_color, 2)
        cv2.putText(img, f"Frame #{self._frame_count}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _render_results(results: list[FrameResult], info: dict, nav_timeline: NavigationTimeline | None, key_prefix: str) -> None:
    timing_keys = sorted({k for r in results for k in r.timings.keys()})
    if timing_keys:
        st.caption("Inference timing (seconds)")
        timing_rows = []
        for key in timing_keys:
            vals = [r.timings[key] for r in results if key in r.timings]
            if vals:
                timing_rows.append(
                    {
                        "stage": key,
                        "mean_s": float(np.mean(vals)),
                        "p95_s": float(np.percentile(vals, 95)),
                        "max_s": float(np.max(vals)),
                    }
                )
        if timing_rows:
            timing_rows = sorted(timing_rows, key=lambda x: x["mean_s"], reverse=True)
            st.dataframe(timing_rows, use_container_width=True, hide_index=True)

    frame_idx = st.slider(
        "Frame",
        0,
        len(results) - 1,
        0,
        format=f"Frame %d / {len(results) - 1}",
        key=f"{key_prefix}_frame_slider",
    )
    r = results[frame_idx]
    hand_poses = getattr(r, "hand_poses", [])
    interactions = getattr(r, "interactions", [])

    tab_names = ["Navigation GT", "Original", "Combined"]
    if any(res.tracking for res in results):
        tab_names.append("Tracking")
    tab_names.append("Detections")
    tab_names.append("Segmentation")
    tab_names.append("Interactions")
    tab_names.append("VLM Output")
    if any(res.temporal_changes for res in results):
        tab_names.append("Temporal Changes")
    tab_names.extend(["Ground Truth JSON", "Accuracy"])

    tabs = st.tabs(tab_names)
    tab_map = {name: tab for name, tab in zip(tab_names, tabs)}

    with tab_map["Navigation GT"]:
        if nav_timeline and nav_timeline.objects:
            st.markdown("### Navigation State Timeline")
            mcols = st.columns(3)
            mcols[0].metric("Nav Objects Tracked", len(nav_timeline.objects))
            mcols[1].metric("State Transitions", nav_timeline.total_transitions)
            obj_types = set(o.object_type for o in nav_timeline.objects)
            mcols[2].metric("Object Types", ", ".join(sorted(obj_types)) if obj_types else "none")

            st.markdown("#### Per-Object State Timeline")
            state_colors = {
                "open": "#2ecc71",
                "closed": "#e74c3c",
                "ajar": "#f39c12",
                "blocked": "#9b59b6",
                "clear": "#3498db",
                "unknown": "#95a5a6",
            }
            total_frames = max(results[-1].frame_idx, 1)

            for obj in nav_timeline.objects:
                st.markdown(f"**{obj.name}** (Track #{obj.track_id}, type: {obj.object_type})")
                bar_html = '<div style="display:flex;height:28px;border-radius:4px;overflow:hidden;margin-bottom:8px;border:1px solid #555;">'
                for entry in obj.state_timeline:
                    width_pct = max((entry.frame_end - entry.frame_start) / total_frames * 100, 1.0)
                    color = state_colors.get(entry.state, "#95a5a6")
                    bar_html += (
                        f'<div style="width:{width_pct:.1f}%;background:{color};display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;font-weight:bold;min-width:20px;" '
                        f'title="frames {entry.frame_start}-{entry.frame_end}: {entry.state}">{entry.state}</div>'
                    )
                bar_html += "</div>"
                st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("#### State Transition Events")
            all_transitions = []
            for obj in nav_timeline.objects:
                for frame, from_s, to_s in obj.transitions:
                    all_transitions.append(
                        {
                            "Frame": frame,
                            "Time (s)": f"{frame / nav_timeline.video_fps:.2f}" if nav_timeline.video_fps > 0 else "-",
                            "Object": obj.name,
                            "Type": obj.object_type,
                            "Track ID": obj.track_id,
                            "From": from_s,
                            "To": to_s,
                        }
                    )
            if all_transitions:
                all_transitions.sort(key=lambda x: x["Frame"])
                st.dataframe(all_transitions, use_container_width=True)
            else:
                st.info("No state transitions detected.")
        else:
            st.info("No navigation objects detected.")

    with tab_map["Original"]:
        st.image(cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB), caption=f"Frame {r.frame_idx}", use_container_width=True)

    with tab_map["Combined"]:
        vis = draw_all(r.frame, r)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Combined - Frame {r.frame_idx}", use_container_width=True)

    if "Tracking" in tab_map:
        with tab_map["Tracking"]:
            if r.tracking:
                vis = draw_tracking(r.frame, r.tracking)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.tracking.boxes)} tracked objects", use_container_width=True)
            else:
                st.info("No tracking results for this frame.")

    with tab_map["Detections"]:
        if r.detection:
            vis = draw_detections(r.frame, r.detection)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.detection.boxes)} detections", use_container_width=True)
        else:
            st.info("No detection results for this frame.")

    with tab_map["Segmentation"]:
        if r.segmentation:
            vis = draw_segmentation(r.frame, r.segmentation)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.segmentation.masks)} masks", use_container_width=True)
        else:
            st.info("No segmentation results for this frame.")

    if "Interactions" in tab_map:
        with tab_map["Interactions"]:
            fps = float(info.get("fps", 0.0) or 0.0)
            any_hands = any(getattr(res, "hand_poses", []) for res in results)
            any_interactions = any(getattr(res, "interactions", []) for res in results)
            hand_frames = [res for res in results if getattr(res, "hand_poses", [])]
            interaction_frames = [res for res in results if getattr(res, "interactions", [])]
            st.caption(
                f"Frames with hands: {len(hand_frames)} / {len(results)} | "
                f"Frames with interactions: {len(interaction_frames)} / {len(results)}"
            )
            if hand_frames or interaction_frames:
                frame_rows = []
                for res in results:
                    res_hands = getattr(res, "hand_poses", [])
                    res_inter = getattr(res, "interactions", [])
                    if not res_hands and not res_inter:
                        continue
                    targets = sorted({f"{it.target_class}[{it.target_index}]" for it in res_inter})
                    frame_rows.append(
                        {
                            "frame_idx": res.frame_idx,
                            "time_s": round((res.frame_idx / fps), 3) if fps > 0 else None,
                            "hands": len(res_hands),
                            "interactions": len(res_inter),
                            "targets": ", ".join(targets),
                        }
                    )
                st.write("Frames with hands/interactions:")
                st.dataframe(frame_rows, use_container_width=True, hide_index=True)

            if hand_poses:
                st.dataframe(
                    [
                        {
                            "hand_id": h.hand_id,
                            "handedness": h.handedness,
                            "score": f"{h.score:.3f}",
                            "bbox": f"{h.bbox}",
                        }
                        for h in hand_poses
                    ]
                )
            else:
                st.info("No hands detected for this frame.")

            if interactions:
                st.dataframe(
                    [
                        {
                            "relation": inter.relation,
                            "hand": inter.hand_id,
                            "target": f"{inter.target_class}[{inter.target_index}]",
                            "contact_score": f"{inter.contact_score:.3f}",
                        }
                        for inter in interactions
                    ]
                )
            else:
                st.info("No interactions for this frame.")
            if not any_hands and not any_interactions:
                st.warning(
                    "No hand/object interaction data was produced for this run. "
                    "Enable interactions and verify MediaPipe is available."
                )

    with tab_map["VLM Output"]:
        if r.vlm:
            st.code(r.vlm.raw_text, language="text")
            if r.vlm.parsed:
                st.json(r.vlm.parsed)
        else:
            st.info("No VLM output for this frame.")
            vlm_frames = [
                res for res in results if res.vlm and (res.vlm.raw_text or res.vlm.parsed)
            ]
            if not vlm_frames:
                st.warning("No VLM output exists in this analyzed recording.")
            else:
                prev_candidates = [res for res in vlm_frames if res.frame_idx < r.frame_idx]
                next_candidates = [res for res in vlm_frames if res.frame_idx > r.frame_idx]
                prev_res = prev_candidates[-1] if prev_candidates else None
                next_res = next_candidates[0] if next_candidates else None
                if prev_res is not None:
                    with st.expander(f"Nearest previous VLM output (frame {prev_res.frame_idx})", expanded=True):
                        if prev_res.vlm is not None:
                            st.code(prev_res.vlm.raw_text, language="text")
                            if prev_res.vlm.parsed:
                                st.json(prev_res.vlm.parsed)
                if next_res is not None:
                    with st.expander(f"Nearest next VLM output (frame {next_res.frame_idx})", expanded=False):
                        if next_res.vlm is not None:
                            st.code(next_res.vlm.raw_text, language="text")
                            if next_res.vlm.parsed:
                                st.json(next_res.vlm.parsed)

    if "Temporal Changes" in tab_map:
        with tab_map["Temporal Changes"]:
            if r.temporal_changes:
                tc = r.temporal_changes
                st.markdown(f"**Comparing frames {tc.frame_idx_before} -> {tc.frame_idx_after}**")
                if tc.state_changes:
                    st.dataframe(
                        [
                            {
                                "Object": sc.object_name,
                                "Before": sc.before_state,
                                "After": sc.after_state,
                                "Confidence": sc.confidence,
                            }
                            for sc in tc.state_changes
                        ]
                    )
                vis = draw_temporal_changes(r.frame, tc)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Temporal Changes Overlay", use_container_width=True)
            else:
                st.info("No temporal changes for this frame.")

    with tab_map["Ground Truth JSON"]:
        st.markdown("**Full Video Summary:**")
        gt = export_ground_truth(results, info, nav_timeline=nav_timeline)
        st.json(gt)
        gt_json = export_ground_truth_json(results, info, nav_timeline=nav_timeline)
        st.download_button(
            "Download Ground Truth JSON",
            gt_json,
            file_name="ground_truth.json",
            mime="application/json",
            type="primary",
            key=f"{key_prefix}_download_gt",
        )

    with tab_map["Accuracy"]:
        st.markdown("### Accuracy Measurement")
        if not nav_timeline or not nav_timeline.objects:
            st.warning("Run with Tracker + VLM first to generate predictions.")
        else:
            uploaded_labels = st.file_uploader(
                "Upload labels JSON",
                type=["json"],
                help='Format: [{"frame": 0, "object": "door", "state": "closed"}, ...]',
                key=f"{key_prefix}_labels_upload",
            )

            with st.expander("Or enter labels manually"):
                num_labels = st.number_input("Number of labels", 1, 50, 5, key=f"{key_prefix}_num_labels")
                manual_entries = []
                for i in range(int(num_labels)):
                    lcols = st.columns(3)
                    frame_num = lcols[0].number_input("Frame", 0, 100000, 0, key=f"{key_prefix}_lbl_frame_{i}")
                    obj_name = lcols[1].text_input("Object", value="door", key=f"{key_prefix}_lbl_obj_{i}")
                    obj_state = lcols[2].selectbox(
                        "State",
                        ["closed", "open", "ajar", "blocked", "clear", "unknown"],
                        key=f"{key_prefix}_lbl_state_{i}",
                    )
                    manual_entries.append({"frame": frame_num, "object": obj_name, "state": obj_state})

            eval_btn = st.button("Evaluate Accuracy", type="primary", key=f"{key_prefix}_eval_btn")
            if eval_btn:
                if uploaded_labels:
                    try:
                        labels_data = json.load(uploaded_labels)
                        manual_labels = parse_manual_labels(labels_data)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON file.")
                        manual_labels = {}
                else:
                    manual_labels = parse_manual_labels(manual_entries)

                if not manual_labels:
                    st.error("No labels provided.")
                else:
                    metrics = evaluate_nav_accuracy(nav_timeline, manual_labels)
                    rcols = st.columns(3)
                    rcols[0].metric("Overall Accuracy", f"{metrics['accuracy']:.1%}")
                    rcols[1].metric("Correct", f"{metrics['correct']} / {metrics['total_labels']}")
                    rcols[2].metric("Labels Evaluated", metrics["total_labels"])
                    if metrics["details"]:
                        st.dataframe(metrics["details"], use_container_width=True)

    st.divider()
    export_btn = st.button("Export Annotated Video", key=f"{key_prefix}_export_btn")
    if export_btn:
        with st.spinner("Rendering video..."):
            out_path = Path(tempfile.mktemp(suffix=".mp4"))
            export_annotated_video(results, out_path, info["fps"], draw_all)
            st.session_state["live_exported_video"] = out_path.read_bytes()
            out_path.unlink(missing_ok=True)

    if st.session_state.get("live_exported_video") is not None:
        st.video(st.session_state["live_exported_video"], format="video/mp4")
        st.download_button(
            "Download MP4",
            st.session_state["live_exported_video"],
            file_name="annotated_output.mp4",
            mime="video/mp4",
            key=f"{key_prefix}_download_annotated",
        )


def _analyze_recording(
    record_path: str,
    *,
    use_interactions: bool,
    interaction_max_hands: int,
    interaction_contact_thresh: float,
    interaction_min_detect_conf: float,
    interaction_min_track_conf: float,
    frame_skip: int,
    vlm_skip: int,
    max_frames: int,
    sequential_offload: bool,
    enable_temporal_diff: bool,
    inference_max_side: int,
    vlm_prompt: str,
    enable_vlm: bool,
    seg_prompt: str,
    enable_seg: bool,
) -> None:
    path = Path(record_path)
    if not path.exists() or path.stat().st_size <= 0:
        st.warning("Selected recording is missing or empty.")
        return

    st.session_state["live_recording_path"] = str(path)
    st.session_state["live_recorded_video"] = path.read_bytes()
    st.info(f"Saved recording to: `{path.resolve()}`")

    with st.status("Post-processing recorded video...", expanded=True) as status:
        status.write("Reading recorded video metadata...")
        info = get_video_info(str(path))
        status.write(
            f"Loaded video: {info.get('frame_count', 0)} frames @ {info.get('fps', 0):.2f} FPS"
        )
        interaction_engine = None
        if use_interactions:
            status.write("Initializing MediaPipe interactions...")
            interaction_engine = InteractionEngine(
                InteractionConfig(
                    enabled=True,
                    max_hands=interaction_max_hands,
                    contact_threshold=interaction_contact_thresh,
                    min_detect_conf=interaction_min_detect_conf,
                    min_track_conf=interaction_min_track_conf,
                )
            )
            if not interaction_engine.available:
                detail = f" Details: {interaction_engine.error}" if interaction_engine.error else ""
                st.warning("MediaPipe interactions requested but unavailable." + detail)
                interaction_engine = None

        status.write("Building analysis pipeline...")
        pipeline = Pipeline(
            detector=st.session_state["live_detector"],
            segmenter=st.session_state["live_segmenter"],
            vlm=st.session_state["live_vlm"],
            tracker=st.session_state["live_tracker"],
            vlm_prompt=(vlm_prompt if enable_vlm else ""),
            seg_prompt=(seg_prompt if enable_seg else ""),
            frame_skip=frame_skip,
            vlm_skip=vlm_skip,
            sequential_offload=sequential_offload,
            interaction_engine=interaction_engine,
            enable_temporal_diff=enable_temporal_diff,
            inference_max_side=(None if inference_max_side == 0 else int(inference_max_side)),
        )

        results: list[FrameResult] = []
        total = min(info["frame_count"], int(max_frames))
        status.write("Running inference...")
        progress = st.progress(0, text="Processing recorded frames...")
        denom = max(1, (total // max(1, frame_skip)) + 1)

        for result in pipeline.run(str(path), max_frames=int(max_frames)):
            results.append(result)
            pct = len(results) / denom
            progress.progress(min(pct, 1.0), text=f"Frame {result.frame_idx}/{total}")

        progress.progress(1.0, text="Done")
        st.session_state["live_results"] = results
        st.session_state["live_video_info"] = info
        st.session_state["live_nav_timeline"] = build_nav_timeline(results, fps=info["fps"]) if results else None
        st.session_state["live_last_analyzed_path"] = str(path)
        status.write(f"Post-processing complete. Generated {len(results)} analyzed frames.")
        status.update(label="Post-processing complete", state="complete")

    st.success("Analysis complete.")


_init_state()

with st.sidebar:
    st.header("Live Config")

    st.subheader("Tracking")
    enable_tracker = st.checkbox("Enable Tracker", value=True)
    tracker_names = list_models("tracker")
    trk_default = tracker_names.index(st.session_state["live_trk_name"]) if st.session_state["live_trk_name"] in tracker_names else 0
    trk_choice = st.selectbox("Tracker", tracker_names, index=trk_default, disabled=not enable_tracker)
    trk_kwargs = _pick_model_size("tracker", trk_choice, "Tracker Size") if enable_tracker else {}

    st.divider()
    enable_seg = st.checkbox("Enable Segmentation", value=True)
    segmenter_names = list_models("segmenter")
    seg_default = segmenter_names.index(st.session_state["live_seg_name"]) if st.session_state["live_seg_name"] in segmenter_names else 0
    seg_choice = st.selectbox("Segmenter", segmenter_names, index=seg_default, disabled=not enable_seg)
    seg_kwargs = _pick_model_size("segmenter", seg_choice, "Segmenter Size") if enable_seg else {}
    seg_prompt = st.text_input("Segmentation Prompt", value="person, door", disabled=not enable_seg)

    st.divider()
    enable_det = st.checkbox("Enable Detection", value=True)
    detector_names = list_models("detector")
    det_default = detector_names.index(st.session_state["live_det_name"]) if st.session_state["live_det_name"] in detector_names else 0
    det_choice = st.selectbox("Detector", detector_names, index=det_default, disabled=not enable_det)
    det_kwargs = _pick_model_size("detector", det_choice, "Detector Size") if enable_det else {}

    st.divider()
    enable_vlm = st.checkbox("Enable Liquid LFM", value=True)
    vlm_names = list_models("vlm")
    preferred = "lfm2.5-vl-mlx" if "lfm2.5-vl-mlx" in vlm_names else vlm_names[0]
    vlm_default = vlm_names.index(st.session_state["live_vlm_name"]) if st.session_state["live_vlm_name"] in vlm_names else vlm_names.index(preferred)
    vlm_choice = st.selectbox("VLM", vlm_names, index=vlm_default, disabled=not enable_vlm)
    prompt_mode = st.selectbox("Prompt Mode", ["Navigation (focused)", "Generic (broad)"], index=0, disabled=not enable_vlm)
    vlm_prompt = NAV_STATE_PROMPT if prompt_mode == "Navigation (focused)" else GENERIC_JSON_SCHEMA_PROMPT
    vlm_interval = st.slider("Run VLM every N frames", min_value=1, max_value=60, value=12, disabled=not enable_vlm)

    st.divider()
    inference_max_side = st.select_slider(
        "Inference Max Side (px)",
        options=[0, 320, 480, 640, 720, 960, 1280],
        value=960,
        help="0 disables resizing.",
    )

    st.subheader("Post-Stop Analysis")
    frame_skip = st.slider("Process every N frames", 1, 30, 1)
    vlm_skip = st.slider("VLM every N processed frames", 1, 50, 5)
    max_frames = st.number_input("Max frames to process", 10, 100000, 10000)
    sequential_offload = st.checkbox("Sequential Offload (low VRAM)", value=False)
    enable_temporal_diff = st.checkbox("Enable Temporal Diff", value=True)
    st.subheader("Interactions")
    use_interactions = st.checkbox("Enable hand-object interactions (MediaPipe)", value=True)
    interaction_max_hands = st.slider("Max hands", 1, 4, 2, disabled=not use_interactions)
    interaction_contact_thresh = st.slider(
        "Contact threshold",
        min_value=0.05,
        max_value=0.9,
        value=0.1,
        step=0.05,
        disabled=not use_interactions,
    )
    interaction_min_detect_conf = st.slider(
        "Hand detect confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        disabled=not use_interactions,
    )
    interaction_min_track_conf = st.slider(
        "Hand track confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        disabled=not use_interactions,
    )

    load_btn = st.button("Load Selected", type="primary", use_container_width=True)
    unload_btn = st.button("Unload All", use_container_width=True)

    if not st.session_state["live_loaded"] and not st.session_state["live_auto_loaded"]:
        with st.spinner("Auto-loading defaults..."):
            _load_selected_models(
                enable_tracker=enable_tracker,
                enable_seg=enable_seg,
                enable_det=enable_det,
                enable_vlm=enable_vlm,
                trk_choice=trk_choice,
                seg_choice=seg_choice,
                det_choice=det_choice,
                vlm_choice=vlm_choice,
                trk_kwargs=trk_kwargs,
                seg_kwargs=seg_kwargs,
                det_kwargs=det_kwargs,
            )
        st.session_state["live_auto_loaded"] = True
        st.success("Default models auto-loaded.")

    if load_btn:
        with st.spinner("Loading selected models..."):
            _load_selected_models(
                enable_tracker=enable_tracker,
                enable_seg=enable_seg,
                enable_det=enable_det,
                enable_vlm=enable_vlm,
                trk_choice=trk_choice,
                seg_choice=seg_choice,
                det_choice=det_choice,
                vlm_choice=vlm_choice,
                trk_kwargs=trk_kwargs,
                seg_kwargs=seg_kwargs,
                det_kwargs=det_kwargs,
            )
        st.success("Models loaded.")

    if unload_btn:
        _safe_unload(st.session_state["live_tracker"])
        _safe_unload(st.session_state["live_segmenter"])
        _safe_unload(st.session_state["live_detector"])
        _safe_unload(st.session_state["live_vlm"])
        st.session_state["live_tracker"] = None
        st.session_state["live_segmenter"] = None
        st.session_state["live_detector"] = None
        st.session_state["live_vlm"] = None
        st.session_state["live_loaded"] = False
        st.info("All models unloaded.")

    loaded = []
    if st.session_state["live_tracker"] is not None:
        loaded.append(f"trk:{st.session_state['live_trk_name']}")
    if st.session_state["live_segmenter"] is not None:
        loaded.append(f"seg:{st.session_state['live_seg_name']}")
    if st.session_state["live_detector"] is not None:
        loaded.append(f"det:{st.session_state['live_det_name']}")
    if st.session_state["live_vlm"] is not None:
        loaded.append(f"vlm:{st.session_state['live_vlm_name']}")
    st.caption("Loaded: " + (", ".join(loaded) if loaded else "none"))

    if use_interactions:
        if (
            st.session_state["live_interaction_engine"] is None
            or not st.session_state["live_interaction_engine"].config.enabled
            or st.session_state["live_interaction_engine"].config.max_hands != interaction_max_hands
            or abs(st.session_state["live_interaction_engine"].config.contact_threshold - interaction_contact_thresh) > 1e-9
            or abs(st.session_state["live_interaction_engine"].config.min_detect_conf - interaction_min_detect_conf) > 1e-9
            or abs(st.session_state["live_interaction_engine"].config.min_track_conf - interaction_min_track_conf) > 1e-9
        ):
            old_engine = st.session_state["live_interaction_engine"]
            if old_engine is not None:
                old_engine.close()
            st.session_state["live_interaction_engine"] = InteractionEngine(
                InteractionConfig(
                    enabled=True,
                    max_hands=interaction_max_hands,
                    contact_threshold=interaction_contact_thresh,
                    min_detect_conf=interaction_min_detect_conf,
                    min_track_conf=interaction_min_track_conf,
                )
            )
        live_ie = st.session_state["live_interaction_engine"]
        if live_ie is not None and not live_ie.available:
            detail = f" Details: {live_ie.error}" if live_ie.error else ""
            st.warning("MediaPipe interactions requested but unavailable." + detail)
    else:
        old_engine = st.session_state["live_interaction_engine"]
        if old_engine is not None:
            old_engine.close()
        st.session_state["live_interaction_engine"] = None


if not st.session_state["live_loaded"]:
    st.warning("Load at least one model from the sidebar before starting the stream.")
else:
    ctx = webrtc_streamer(
        key="live-perception",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Recover from stale state when user navigates away and returns.
    if not ctx.video_processor:
        if st.session_state.get("live_prev_playing", False):
            st.session_state["live_prev_playing"] = False
        if st.session_state.get("live_recording", False):
            st.session_state["live_recording"] = False
            st.info("Recovered live session state after page navigation. Start the stream again.")

    if ctx.video_processor:
        ctx.video_processor.tracker = st.session_state["live_tracker"]
        ctx.video_processor.segmenter = st.session_state["live_segmenter"]
        ctx.video_processor.detector = st.session_state["live_detector"]
        ctx.video_processor.vlm = st.session_state["live_vlm"]
        ctx.video_processor.interaction_engine = st.session_state["live_interaction_engine"]
        ctx.video_processor.seg_prompt = seg_prompt
        ctx.video_processor.vlm_interval = vlm_interval
        ctx.video_processor.inference_max_side = inference_max_side

        is_playing = bool(getattr(getattr(ctx, "state", None), "playing", False))

        c1, c2, c3 = st.columns(3)
        c1.write("Recording auto-starts when live stream is active.")
        c2.write("Use WebRTC START/STOP above.")
        st.caption("Press WebRTC `STOP` to stop stream, then automatic post-processing runs on the saved clip.")

        if is_playing and not ctx.video_processor.recording and not st.session_state["live_recording"]:
            captures_dir = Path("outputs/live_captures")
            captures_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            record_path = str(captures_dir / f"live_capture_{ts}.mp4")
            ctx.video_processor.start_recording(record_path, fps=20.0)
            st.session_state["live_recording"] = True
            st.session_state["live_recording_path"] = record_path
            st.info(f"Recording started automatically: `{Path(record_path).resolve()}`")

        just_stopped = st.session_state.get("live_prev_playing", False) and not is_playing
        st.session_state["live_prev_playing"] = is_playing

        if just_stopped and st.session_state["live_recording"]:
            record_path = ctx.video_processor.stop_recording()
            st.session_state["live_recording"] = False
            st.session_state["live_recording_path"] = record_path
            if (
                record_path
                and Path(record_path).exists()
                and Path(record_path).stat().st_size > 0
                and st.session_state.get("live_last_analyzed_path") != record_path
            ):
                _analyze_recording(
                    record_path,
                    use_interactions=use_interactions,
                    interaction_max_hands=interaction_max_hands,
                    interaction_contact_thresh=interaction_contact_thresh,
                    interaction_min_detect_conf=interaction_min_detect_conf,
                    interaction_min_track_conf=interaction_min_track_conf,
                    frame_skip=frame_skip,
                    vlm_skip=vlm_skip,
                    max_frames=int(max_frames),
                    sequential_offload=sequential_offload,
                    enable_temporal_diff=enable_temporal_diff,
                    inference_max_side=inference_max_side,
                    vlm_prompt=vlm_prompt,
                    enable_vlm=enable_vlm,
                    seg_prompt=seg_prompt,
                    enable_seg=enable_seg,
                )
            else:
                st.warning("No recorded video found. Start recording first, then stop.")

        c3.write("Recording: **ON**" if ctx.video_processor.recording else "Recording: **OFF**")


if st.session_state.get("live_recorded_video"):
    st.subheader("Recorded Video")
    if st.session_state.get("live_recording_path"):
        st.caption(f"Saved path: `{Path(st.session_state['live_recording_path']).resolve()}`")
    st.video(st.session_state["live_recorded_video"], format="video/mp4")
    st.download_button(
        "Download Recorded MP4",
        st.session_state["live_recorded_video"],
        file_name="live_capture.mp4",
        mime="video/mp4",
        key="download_recorded_live",
    )

captures_dir = Path("outputs/live_captures")
history_files = sorted(captures_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if captures_dir.exists() else []
if history_files:
    st.subheader("Recording History")
    selected_name = st.selectbox(
        "Open previous recording",
        options=[p.name for p in history_files],
        key="live_history_select",
    )
    selected_path = captures_dir / selected_name
    if selected_path.exists():
        selected_bytes = selected_path.read_bytes()
        st.caption(f"History path: `{selected_path.resolve()}`")
        st.video(selected_bytes, format="video/mp4")
        st.download_button(
            "Download Selected Recording",
            selected_bytes,
            file_name=selected_path.name,
            mime="video/mp4",
            key="download_selected_live_history",
        )
        if st.button("Analyze Selected Recording", key="analyze_selected_live_history", type="primary"):
            _analyze_recording(
                str(selected_path),
                use_interactions=use_interactions,
                interaction_max_hands=interaction_max_hands,
                interaction_contact_thresh=interaction_contact_thresh,
                interaction_min_detect_conf=interaction_min_detect_conf,
                interaction_min_track_conf=interaction_min_track_conf,
                frame_skip=frame_skip,
                vlm_skip=vlm_skip,
                max_frames=int(max_frames),
                sequential_offload=sequential_offload,
                enable_temporal_diff=enable_temporal_diff,
                inference_max_side=inference_max_side,
                vlm_prompt=vlm_prompt,
                enable_vlm=enable_vlm,
                seg_prompt=seg_prompt,
                enable_seg=enable_seg,
            )

live_results = st.session_state.get("live_results", [])
if live_results:
    st.divider()
    st.subheader("Post-Stop Analysis")
    _render_results(
        live_results,
        st.session_state["live_video_info"],
        st.session_state.get("live_nav_timeline"),
        key_prefix="live",
    )
