from __future__ import annotations

import tempfile
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import models  # triggers registration
from core.registry import create, list_models
from core.video import get_video_info, export_annotated_video
from core.types import FrameResult
from models.vlm.lfm25 import DEFAULT_JSON_SCHEMA_PROMPT
from pipeline.runner import Pipeline
from pipeline.visualizer import draw_detections, draw_segmentation, draw_vlm_text, draw_all

st.set_page_config(layout="wide", page_title="World2Data — Multi-Model Video Inference")

# --- Session state defaults ---
for key, default in [
    ("results", []),
    ("models_loaded", False),
    ("detector_instance", None),
    ("segmenter_instance", None),
    ("vlm_instance", None),
    ("exported_video", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Display size CSS ---
fullscreen = st.checkbox("Fullscreen images", value=False)
if not fullscreen:
    st.markdown(
        """<style>
        img, video { max-height: 500px !important; object-fit: contain !important; }
        </style>""",
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Model Configuration")

    # --- Detector ---
    st.subheader("Object Detection")
    use_detector = st.checkbox("Enable Detector", value=True)
    if use_detector:
        det_names = list_models("detector")
        det_choice = st.selectbox("Detector Model", det_names, index=0)
        det_size = st.selectbox("Model Size", ["n", "s", "m", "l", "x"], index=0)

    # --- Segmenter ---
    st.subheader("Segmentation")
    use_segmenter = st.checkbox("Enable Segmenter", value=False)
    if use_segmenter:
        seg_names = list_models("segmenter")
        seg_choice = st.selectbox("Segmenter Model", seg_names, index=0)
        seg_prompt = st.text_input("Segmentation Text Prompt", value="object",
                                   help="Open-vocabulary: what to segment (e.g. 'person', 'door', 'obstacle')")

    # --- VLM ---
    st.subheader("Vision-Language Model")
    use_vlm = st.checkbox("Enable VLM", value=False)
    if use_vlm:
        vlm_names = list_models("vlm")
        vlm_choice = st.selectbox("VLM Model", vlm_names, index=0)
        vlm_prompt = st.text_area("VLM Prompt", value=DEFAULT_JSON_SCHEMA_PROMPT, height=150)

    # --- Pipeline settings ---
    st.subheader("Pipeline Settings")
    frame_skip = st.slider("Process every N frames", 1, 30, 1)
    vlm_skip = st.slider("VLM: run every N processed frames", 1, 50, 5,
                          help="VLM is slow. This skips frames independently of the main frame skip.")
    max_frames = st.number_input("Max frames to process", 10, 100000, 100)
    sequential_offload = st.checkbox("Sequential Offload (low VRAM)",
                                     value=False,
                                     help="Load/unload each model per-frame. Slower but uses less VRAM.")

    st.divider()

    # --- Load / Unload ---
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("Load Models", type="primary", use_container_width=True)
    with col2:
        unload_btn = st.button("Unload All", use_container_width=True)

    if load_btn:
        with st.spinner("Loading models..."):
            # Detector
            if use_detector:
                det = create("detector", det_choice, model_size=det_size)
                if not sequential_offload:
                    det.load()
                st.session_state["detector_instance"] = det
            else:
                st.session_state["detector_instance"] = None

            # Segmenter
            if use_segmenter:
                seg = create("segmenter", seg_choice)
                if not sequential_offload:
                    seg.load()
                st.session_state["segmenter_instance"] = seg
            else:
                st.session_state["segmenter_instance"] = None

            # VLM
            if use_vlm:
                vlm = create("vlm", vlm_choice)
                if not sequential_offload:
                    vlm.load()
                st.session_state["vlm_instance"] = vlm
            else:
                st.session_state["vlm_instance"] = None

            st.session_state["models_loaded"] = True
        st.success("Models loaded!")

    if unload_btn:
        for key in ("detector_instance", "segmenter_instance", "vlm_instance"):
            inst = st.session_state.get(key)
            if inst is not None:
                inst.unload()
            st.session_state[key] = None
        st.session_state["models_loaded"] = False
        st.session_state["results"] = []
        st.success("All models unloaded.")

    # Status
    st.divider()
    st.caption("Loaded models:")
    for label, key in [("Detector", "detector_instance"), ("Segmenter", "segmenter_instance"), ("VLM", "vlm_instance")]:
        inst = st.session_state.get(key)
        st.write(f"- {label}: {'✓ ' + type(inst).__name__ if inst else '—'}")


# ============================================================
# MAIN AREA
# ============================================================
st.title("World2Data — Multi-Model Video Inference")

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush()
    video_path = tmp.name

    # Show video info
    info = get_video_info(video_path)
    cols = st.columns(4)
    cols[0].metric("FPS", f"{info['fps']:.1f}")
    cols[1].metric("Frames", info["frame_count"])
    cols[2].metric("Width", info["width"])
    cols[3].metric("Height", info["height"])

    # Run inference
    run_btn = st.button("Run Inference", type="primary",
                        disabled=not st.session_state["models_loaded"])

    if run_btn:
        pipeline = Pipeline(
            detector=st.session_state["detector_instance"],
            segmenter=st.session_state["segmenter_instance"],
            vlm=st.session_state["vlm_instance"],
            vlm_prompt=vlm_prompt if use_vlm else "",
            seg_prompt=seg_prompt if use_segmenter else "",
            frame_skip=frame_skip,
            vlm_skip=vlm_skip,
            sequential_offload=sequential_offload,
        )

        results: list[FrameResult] = []
        total = min(info["frame_count"], max_frames)
        progress = st.progress(0, text="Processing frames...")
        frame_display = st.empty()

        for result in pipeline.run(video_path, max_frames=max_frames):
            results.append(result)
            pct = len(results) / (total // frame_skip + 1)
            progress.progress(min(pct, 1.0), text=f"Frame {result.frame_idx}/{total}")

            # Live preview every 5 processed frames
            if len(results) % 5 == 1:
                preview = draw_all(result.frame, result)
                frame_display.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {result.frame_idx}", use_container_width=True)

        progress.progress(1.0, text="Done!")
        st.session_state["results"] = results
        st.success(f"Processed {len(results)} frames.")

    # --- Results viewer ---
    results = st.session_state["results"]
    if results:
        st.divider()
        st.subheader("Results Viewer")

        frame_idx = st.slider("Frame", 0, len(results) - 1, 0,
                               format=f"Frame %d / {len(results) - 1}")
        r = results[frame_idx]

        tab_orig, tab_det, tab_seg, tab_combined, tab_vlm = st.tabs(
            ["Original", "Detections", "Segmentation", "Combined", "VLM Output"]
        )

        with tab_orig:
            st.image(cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB),
                     caption=f"Frame {r.frame_idx}", use_container_width=True)

        with tab_det:
            if r.detection:
                vis = draw_detections(r.frame, r.detection)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{len(r.detection.boxes)} detections", use_container_width=True)
                # Detection table
                if r.detection.boxes:
                    st.dataframe([
                        {"class": b.class_name, "confidence": f"{b.confidence:.3f}",
                         "bbox": f"({int(b.x1)},{int(b.y1)},{int(b.x2)},{int(b.y2)})"}
                        for b in r.detection.boxes
                    ])
            else:
                st.info("No detection results for this frame.")

        with tab_seg:
            if r.segmentation:
                vis = draw_segmentation(r.frame, r.segmentation)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{len(r.segmentation.masks)} masks", use_container_width=True)
            else:
                st.info("No segmentation results for this frame.")

        with tab_combined:
            vis = draw_all(r.frame, r)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption=f"Combined — Frame {r.frame_idx}", use_container_width=True)

        with tab_vlm:
            if r.vlm:
                st.text(f"Raw output (frame {r.frame_idx}):")
                st.code(r.vlm.raw_text, language="text")
                if r.vlm.parsed:
                    st.json(r.vlm.parsed)
                else:
                    st.warning("JSON parsing failed. Raw text shown above.")
            else:
                st.info("No VLM output for this frame (VLM skip or not enabled).")

        # --- Export ---
        st.divider()
        export_col1, export_col2 = st.columns([1, 3])
        with export_col1:
            export_btn = st.button("Export Annotated Video")

        if export_btn:
            with st.spinner("Rendering video..."):
                out_path = Path(tempfile.mktemp(suffix=".mp4"))
                export_annotated_video(results, out_path, info["fps"] / frame_skip, draw_all)
                video_bytes = out_path.read_bytes()
                st.session_state["exported_video"] = video_bytes
                out_path.unlink(missing_ok=True)

        # Show video player + download if export exists
        if st.session_state["exported_video"] is not None:
            st.video(st.session_state["exported_video"], format="video/mp4")
            st.download_button(
                "Download MP4",
                st.session_state["exported_video"],
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )
