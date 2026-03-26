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

@rt("/")
def get():
    return Html(
        Head(
            Title("Gesture Recognition"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Style("""
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    background: #0a0a0f;
                    color: #e2e8f0;
                    font-family: system-ui, sans-serif;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 2rem 1rem;
                    gap: 1rem;
                }
                h1 { color: #a78bfa; font-size: 1.6rem; letter-spacing: -0.02em; }
                #status { color: #5a6478; font-size: 0.8rem; }
                #canvas { border-radius: 12px; display: block; border: 1px solid rgba(124,58,237,0.3); }
                #gesture_text {
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #fff;
                    padding: 0.5rem 1.5rem;
                    background: rgba(124,58,237,0.12);
                    border: 1px solid rgba(124,58,237,0.25);
                    border-radius: 10px;
                    min-width: 220px;
                    text-align: center;
                }
            """)
        ),
        Body(
            H1("🤚 Reconhecimento de Gestos"),
            Div("Iniciando...", id="status"),
            Canvas(id="canvas", width="640", height="480"),
            Div("---", id="gesture_text"),
            Script("""
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                const statusEl = document.getElementById('status');
                const gestureEl = document.getElementById('gesture_text');
                
                // Câmera escondida para captura
                const video = document.createElement('video');
                video.setAttribute('playsinline', '');
                video.setAttribute('autoplay', '');
                video.style.opacity = '0';
                video.style.position = 'absolute';
                document.body.appendChild(video);

                const off = document.createElement('canvas');
                off.width = 640; off.height = 480;
                const offCtx = off.getContext('2d');

                let ws = null;
                let busy = false;
                let videoReady = false;

                video.addEventListener('playing', () => {
                    videoReady = true;
                    if(ws && ws.readyState === 1) sendFrame();
                });

                async function start() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({video: {width:640, height:480}});
                        video.srcObject = stream;
                        await video.play();
                    } catch(err) {
                        statusEl.textContent = 'Erro câmera: ' + err.message;
                        return;
                    }

                    ws = new WebSocket(`ws://${location.host}/ws`);
                    ws.onopen = () => { 
                        statusEl.textContent = 'Conectado! Streaming...'; 
                        if(videoReady) sendFrame(); 
                    };
                    ws.onclose = () => { statusEl.textContent = 'Desconectado.'; };
                    
                    ws.onmessage = (e) => {
                        const data = JSON.parse(e.data);
                        if (data.error) {
                            statusEl.textContent = 'Erro Servidor: ' + data.error;
                            busy = false;
                            return;
                        }
                        
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0);
                            busy = false;
                            requestAnimationFrame(sendFrame);
                        };
                        img.src = 'data:image/jpeg;base64,' + data.frame;

                        if (data.detections && data.detections.length > 0) {
                            gestureEl.textContent = data.detections[0].gesture_name + ' (' + Math.round(data.detections[0].probability*100) + '%)';
                        } else {
                            gestureEl.textContent = '---';
                        }
                    };
                }

                function sendFrame() {
                    if(!ws || ws.readyState !== 1 || busy || !videoReady) return;
                    busy = true;
                    offCtx.drawImage(video, 0, 0, 640, 480);
                    ws.send(off.toDataURL('image/jpeg', 0.6).split(',')[1]);
                }

                start();
            """)
        )
    )

serve(port=5001)
