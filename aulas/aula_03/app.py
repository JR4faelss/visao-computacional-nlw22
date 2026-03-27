"""FastHTML backend mínimo para reconhecimento de gestos."""

import sys
import time
import json
import base64
import traceback
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# Core modules
_CORE_DIR = str(Path(__file__).resolve().parent / "core")
sys.path.insert(0, _CORE_DIR)

from model_loader import check_model_files, load_custom_model, create_gesture_recognizer_image
from process_frame_image import process_single_image

from fasthtml.common import *
from starlette.websockets import WebSocket
from starlette.routing import WebSocketRoute
from starlette.staticfiles import StaticFiles

# ── Caminhos ──
_BASE_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _BASE_DIR / "assets"

# ── Carrega modelos ──
assert check_model_files(), "Arquivos de modelo não encontrados."
clf, label_encoder = load_custom_model()
options = create_gesture_recognizer_image()
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# ── WebSocket Nativo do Starlette ──
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> WebSocket Conectado!", flush=True)
    
    while True:
        try:
            msg = await websocket.receive_text()
            
            # Decode JPEG
            img_bytes = base64.b64decode(msg)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "Frame vazio."}))
                continue

            frame = cv2.flip(frame, 1)

            # Processa o frame
            annotated, detections = process_single_image(frame, recognizer, clf, label_encoder)

            # Transforma em base64 novamente
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode("ascii")

            await websocket.send_text(json.dumps({"frame": b64, "detections": detections}))

        except Exception as e:
            e_str = str(e)
            if any(x in e_str.lower() for x in ["disconnect", "close", "1000", "1001"]):
                print(">>> WebSocket: cliente desconectou.", flush=True)
                break
            print(f">>> Erro WS: {e_str}", flush=True)
            try:
                await websocket.send_text(json.dumps({"error": e_str}))
            except:
                break

# ── Iniciando FastHTML com Rota WS nativa ──
app, rt = fast_app(routes=[WebSocketRoute("/ws", endpoint=websocket_endpoint)])

# ── Servir arquivos estáticos da pasta assets ──
app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

@rt("/")
def get():
    return Html(
        Head(
            Title("Gesture Recognition"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Link(rel="stylesheet", href="/assets/style.css"),
        ),
        Body(
            H1("🤚 Reconhecimento de Gestos"),
            Div("Iniciando...", id="status"),
            Canvas(id="canvas", width="640", height="480"),
            Div("---", id="gesture_text"),
            Script(src="/assets/main.js"),
        )
    )

serve(port=5001)

