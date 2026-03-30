"""Handler do WebSocket — decodifica frames, processa e devolve resultado."""

import json
import base64
import time

import cv2
import numpy as np
import asyncio
from starlette.websockets import WebSocket

from process_frame_image import process_single_image
from gesture_images import get_gesture_image


async def websocket_endpoint(websocket: WebSocket):
    """Recebe frames JPEG em base64, processa com MediaPipe + modelo customizado
    e devolve o frame anotado + detecções via WebSocket."""
    await websocket.accept()
    print(">>> WebSocket Conectado!", flush=True)

    # Os modelos são injetados via app.state
    recognizer = websocket.app.state.recognizer
    clf = websocket.app.state.clf
    label_encoder = websocket.app.state.label_encoder

    # Variáveis para cálculo de FPS e Tracking de Vídeo
    prev_time = time.time()
    start_time = time.time()
    last_timestamp_ms = -1

    while True:
        try:
            msg = await websocket.receive_text()

            # Tenta decodificar como JSON com configurações
            try:
                data = json.loads(msg)
                img_b64 = data.get("image", "")
                show_landmarks = data.get("show_landmarks", True)
                quality_val = data.get("quality", 70)
            except json.JSONDecodeError:
                img_b64 = msg
                show_landmarks = True
                quality_val = 70

            # Decode JPEG
            img_bytes = base64.b64decode(img_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "Frame vazio."}))
                continue

            # Calcula tempo decorrido no modo contínuo garantindo valor estritamente crescente
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            # Processa o frame e converte via pool de threads para evitar bloqueios de latência do WebSocket (Outro Boost Massivo)
            def offload_cpu_works():
                nonlocal frame
                frame = cv2.flip(frame, 1)
                ann, det = process_single_image(
                    frame, recognizer, clf, label_encoder, timestamp_ms=timestamp_ms, show_landmarks=show_landmarks
                )
                _, bfr = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, int(quality_val)])
                return ann, det, bfr
            
            annotated, detections, buf = await asyncio.to_thread(offload_cpu_works)

            # Transforma em base64 e envia text text
            b64 = base64.b64encode(buf).decode("ascii")

            # Determina a imagem do gesto apenas se a precisão for maior que 40% (0.40)
            image_name = None
            if detections:
                if detections[0]["probability"] > 0.40:
                    image_name = get_gesture_image(detections[0]["gesture_name"])
                else:
                    detections[0]["gesture_name"] = "Desconhecido"

            # Cálculo do FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
            prev_time = curr_time

            await websocket.send_text(
                json.dumps({"frame": b64, "detections": detections, "image": image_name, "fps": round(fps, 1)})
            )

        except Exception as e:
            e_str = str(e)
            if any(
                x in e_str.lower()
                for x in ["disconnect", "close", "1000", "1001"]
            ):
                print(">>> WebSocket: cliente desconectou.", flush=True)
                break
            print(f">>> Erro WS: {e_str}", flush=True)
            try:
                await websocket.send_text(json.dumps({"error": e_str}))
            except Exception:
                break
