"""Processamento de frame em modo VIDEO para uso com streamlit-webrtc."""

import sys
from pathlib import Path

# Garante que a pasta `core/` esteja no path para imports locais
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple

from landmark_utils import extract_features_two_hands


def process_single_image(
    frame: np.ndarray,
    recognizer,
    clf,
    label_encoder,
    timestamp_ms: int = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Processa um frame BGR e retorna o frame anotado + lista de detecções.

    Se timestamp_ms for passado, usa recognize_for_video (modo VIDEO).
    Caso contrário, usa recognize (modo IMAGE).
    Extrai 129 features (2 mãos) para o modelo customizado.

    Args:
        frame: Imagem BGR (np.ndarray).
        recognizer: Instância do GestureRecognizer do MediaPipe.
        clf: Classificador treinado (sklearn).
        label_encoder: LabelEncoder treinado.
        timestamp_ms: Timestamp crescente em milissegundos (opcional).

    Returns:
        Tuple:
            - np.ndarray: Frame BGR com landmarks desenhados e texto de predição.
            - List[Dict]: Lista de detecções com 'gesture_name', 'probability', 'num_hands'.
    """
    annotated = frame.copy()
    detections: List[Dict] = []

    # Converte para RGB e cria a imagem do MediaPipe
    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    if timestamp_ms is not None:
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)
    else:
        result = recognizer.recognize(mp_image)

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    mp_hands = mp.tasks.vision.HandLandmarksConnections

    if result.hand_landmarks:
        # Desenha landmarks de TODAS as mãos
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Extrai features (129 valores: num_hands + hand1 + hand2)
        features = extract_features_two_hands(result)

        if features is not None:
            prediction_idx = clf.predict(features)[0]
            prediction_prob = float(np.max(clf.predict_proba(features)))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            num_hands = len(result.hand_landmarks)
            # Inverte os labels para exibição: o cv2.flip espelha o frame,
            # então MediaPipe reporta "Left" para a mão direita real do usuário.
            # Os VALORES numéricos nas features não mudam (compatível com treino).
            _swap = {"Left": "Right", "Right": "Left"}
            hand_labels = [
                _swap.get(result.handedness[i][0].category_name, "?")
                for i in range(num_hands)
            ]
            hands_str = " + ".join(hand_labels)

            # Desenha texto no frame
            color = (0, 255, 0)
            display_text = f"{gesture_name} ({prediction_prob:.0%})"
            cv2.putText(
                annotated, display_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
            )
            cv2.putText(
                annotated, f"Maos: {hands_str}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )

            detections.append({
                "gesture_name": gesture_name,
                "probability": prediction_prob,
                "num_hands": num_hands,
                "hand_labels": hands_str,
            })

    return annotated, detections
