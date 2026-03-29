"""Handler do WebSocket — decodifica frames, processa e devolve resultado."""

import json
import base64

import cv2
import numpy as np
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
            annotated, detections = process_single_image(
                frame, recognizer, clf, label_encoder
            )

            # Transforma em base64 novamente
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode("ascii")

            # Determina a imagem do gesto (se houver)
            image_name = None
            if detections:
                image_name = get_gesture_image(detections[0]["gesture_name"])

            await websocket.send_text(
                json.dumps({"frame": b64, "detections": detections, "image": image_name})
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
