"""Webapp Streamlit para reconhecimento de gestos via webcam do navegador."""

import sys
import time
import threading
from pathlib import Path

# Adiciona a pasta `core/` ao path para imports dos módulos internos
_CORE_DIR = str(Path(__file__).resolve().parent.parent / "core")
sys.path.insert(0, _CORE_DIR)

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from model_loader import (
    check_model_files,
    load_custom_model,
    create_gesture_recognizer,
)
from process_frame_image import process_single_image


class GestureVideoProcessor(VideoProcessorBase):
    """Processa cada frame de vídeo usando o pipeline de reconhecimento de gestos."""

    def __init__(self):
        if not check_model_files():
            raise FileNotFoundError(
                "Arquivos de modelo não encontrados (.task, .joblib)"
            )

        self.clf, self.label_encoder = load_custom_model()
        options = create_gesture_recognizer()
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.recognizer = GestureRecognizer.create_from_options(options)

        # Timestamp crescente para modo VIDEO
        self._start_ns = time.time_ns()

        # Compartilha detecções entre a thread do recv e o loop do Streamlit
        self.result_lock = threading.Lock()
        self.last_detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Callback chamado a cada frame da webcam."""
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Timestamp crescente em ms (exigido pelo modo VIDEO)
        timestamp_ms = (time.time_ns() - self._start_ns) // 1_000_000

        try:
            annotated, detections = process_single_image(
                img, self.recognizer, self.clf, self.label_encoder, timestamp_ms
            )
            with self.result_lock:
                self.last_detections = detections
        except Exception as e:
            print(f"[recv] Erro ao processar frame: {e}")
            annotated = img

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")



# ─── Layout Streamlit ─────────────────────────────────────────
st.set_page_config(page_title="Gesture Recognition", page_icon="🤚", layout="wide")

# ─── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(145deg, #0a0a0f 0%, #111128 40%, #0d1b2a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* ── Main title ── */
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 30%, #6d28d9 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.6rem !important;
        letter-spacing: -0.03em;
        text-align: center;
        padding-top: 1.5rem;
        animation: fadeInDown 0.8s ease-out;
    }

    /* ── Subtitle / markdown text ── */
    .stMarkdown p {
        color: #a5b4cb;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        text-align: center;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* ── Glassmorphism card around video ── */
    .stElementContainer:has(iframe),
    .stElementContainer:has(video),
    [data-testid="stWebRtcStreamer"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(167, 139, 250, 0.15);
        border-radius: 20px;
        padding: 1rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow:
            0 8px 32px rgba(124, 58, 237, 0.08),
            0 0 60px rgba(124, 58, 237, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.6s ease-out 0.2s both;
        max-width: 900px;
        width: 100%;
        margin: 0 auto;
    }

    /* ── Video element itself ── */
    video {
        border-radius: 14px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        width: 100% !important;
        height: auto !important;
    }

    /* ── START/STOP button ── */
    .stButton > button,
    button[kind="primary"],
    [data-testid="stWebRtcStreamer"] button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3) !important;
        cursor: pointer !important;
    }

    .stButton > button:hover,
    [data-testid="stWebRtcStreamer"] button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%) !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active,
    [data-testid="stWebRtcStreamer"] button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
    }

    /* ── Divider ── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.2), transparent);
        margin: 2rem 0;
    }

    /* ── Caption / footer text ── */
    .stCaption, small, .stMarkdown small {
        color: #5a6478 !important;
        font-family: 'Inter', sans-serif;
        text-align: center;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-live {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    .badge-model {
        background: rgba(124, 58, 237, 0.12);
        color: #a78bfa;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }

    /* ── Info cards ── */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.2rem 0;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(167, 139, 250, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .info-card:hover {
        border-color: rgba(167, 139, 250, 0.2);
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }

    .info-card .icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }

    .info-card .label {
        color: #5a6478;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    .info-card .value {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* ── Animations ── */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.5; }
    }

    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────
st.markdown("# 🤚 Reconhecimento de Gestos")

st.markdown("""
<p style="text-align:center; margin-top:-0.5rem; margin-bottom:1.5rem;">
    Reconhecimento de gestos em tempo real com
    <span class="badge badge-model">MediaPipe</span> +
    <span class="badge badge-model">Custom ML</span>
</p>
""", unsafe_allow_html=True)

# ─── Info cards ───────────────────────────────────────────────
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="icon">🎯</div>
        <div class="label">Modelo</div>
        <div class="value">Custom Sklearn</div>
    </div>
    <div class="info-card">
        <div class="icon">✋</div>
        <div class="label">Mãos</div>
        <div class="value">Até 2 mãos</div>
    </div>
    <div class="info-card">
        <div class="icon">⚡</div>
        <div class="label">Modo</div>
        <div class="value">Tempo Real</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WebRTC streamer ─────────────────────────────────────────
ctx = webrtc_streamer(
    key="gesture-recognition",
    video_processor_factory=GestureVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ─── Painel de Detecção em Tempo Real ─────────────────────────
st.markdown("---")

if ctx.video_processor:
    # Placeholder para atualizar sem recarregar a página toda
    detection_placeholder = st.empty()

    while ctx.video_processor:
        with ctx.video_processor.result_lock:
            detections = list(ctx.video_processor.last_detections)

        if detections:
            cards_html = ""
            for d in detections:
                num_hands = d.get("num_hands", 1)
                hand_emoji = "🤝" if num_hands >= 2 else "🤚"
                prob_pct = int(d["probability"] * 100)
                bar_color = "#4ade80" if prob_pct >= 80 else "#facc15" if prob_pct >= 50 else "#f87171"
                cards_html += f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(167,139,250,0.15);
                    border-radius: 16px;
                    padding: 1.2rem 1.5rem;
                    margin-bottom: 0.75rem;
                    backdrop-filter: blur(12px);
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <span style="font-size:1.05rem; font-weight:700; color:#e2e8f0;">
                            {hand_emoji} {d.get("hand_labels", "—")}
                        </span>
                        <span style="
                            background: rgba(124,58,237,0.15);
                            color: #a78bfa;
                            border: 1px solid rgba(124,58,237,0.25);
                            border-radius: 50px;
                            padding: 0.2rem 0.8rem;
                            font-size: 0.75rem;
                            font-weight: 600;
                            letter-spacing: 0.05em;
                        ">{prob_pct}% confidence</span>
                    </div>
                    <div style="font-size:1.8rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:0.6rem;">
                        {d["gesture_name"]}
                    </div>
                    <div style="background: rgba(255,255,255,0.05); border-radius:50px; height:6px; overflow:hidden;">
                        <div style="
                            width:{prob_pct}%;
                            height:100%;
                            background: linear-gradient(90deg, {bar_color}, {bar_color}cc);
                            border-radius:50px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                </div>
                """
            detection_placeholder.markdown(
                f'<div style="font-family:Inter,sans-serif;">{cards_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            detection_placeholder.markdown("""
            <div style="
                text-align:center;
                padding: 1.5rem;
                background: rgba(255,255,255,0.02);
                border: 1px dashed rgba(167,139,250,0.12);
                border-radius: 16px;
                color: #5a6478;
                font-family: Inter, sans-serif;
                font-size: 0.9rem;
            ">
                👋 Nenhuma mão detectada — mostre a mão para a câmera
            </div>
            """, unsafe_allow_html=True)

        import time
        time.sleep(0.1)  # atualiza ~10x por segundo

# ─── Footer ──────────────────────────────────────────────────
st.markdown("""
<p style="text-align:center; font-size:0.82rem; color:#5a6478; margin-top:1rem;">
    <span class="pulse-dot"></span>
    Pressione <strong style="color:#a78bfa;">START</strong> para iniciar a câmera
    · Mostre gestos com as mãos
</p>
""", unsafe_allow_html=True)

