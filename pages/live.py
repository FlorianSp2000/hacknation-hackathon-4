from __future__ import annotations

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import models  # triggers registration
from core.registry import create
from core.types import SegmentationResult
from pipeline.visualizer import draw_segmentation


st.set_page_config(layout="wide", page_title="World2Data — Live SAM3")
st.title("Live SAM3 Video Segmentation")

# --- Session state ---
if "live_segmenter" not in st.session_state:
    st.session_state["live_segmenter"] = None
if "live_loaded" not in st.session_state:
    st.session_state["live_loaded"] = False

# --- Sidebar ---
with st.sidebar:
    st.header("SAM3 Live Config")
    seg_prompt = st.text_input("Text Prompt", value="person",
                               help="What to segment (e.g. 'person', 'door', 'chair')")

    load_btn = st.button("Load SAM3", type="primary", use_container_width=True)
    unload_btn = st.button("Unload SAM3", use_container_width=True)

    if load_btn:
        with st.spinner("Loading SAM3 VideoModel..."):
            seg = create("segmenter", "sam3")
            seg.load()
            st.session_state["live_segmenter"] = seg
            st.session_state["live_loaded"] = True
        st.success("SAM3 loaded!")

    if unload_btn:
        seg = st.session_state["live_segmenter"]
        if seg is not None:
            seg.unload()
        st.session_state["live_segmenter"] = None
        st.session_state["live_loaded"] = False
        st.info("SAM3 unloaded.")

    st.divider()
    status = "Loaded" if st.session_state["live_loaded"] else "Not loaded"
    st.caption(f"SAM3 status: {status}")


class SAM3VideoProcessor(VideoProcessorBase):
    """WebRTC video processor that runs SAM3 streaming segmentation."""

    def __init__(self):
        self.segmenter = None
        self.prompt = "person"
        self._frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.segmenter is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        result: SegmentationResult = self.segmenter.predict(img, text_prompt=self.prompt)
        self._frame_count += 1

        if result.masks:
            img = draw_segmentation(img, result)

        cv2.putText(img, f"Frame #{self._frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if not st.session_state["live_loaded"]:
    st.warning("Load SAM3 from the sidebar before starting the stream.")
else:
    st.info(f"Segmenting: **{seg_prompt}** — SAM3 builds temporal context across frames.")

    ctx = webrtc_streamer(
        key="sam3-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SAM3VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Pass model + prompt to processor thread
    if ctx.video_processor:
        ctx.video_processor.segmenter = st.session_state["live_segmenter"]
        ctx.video_processor.prompt = seg_prompt
