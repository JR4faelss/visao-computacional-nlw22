"""Função principal de processamento: recebe uma imagem e retorna a imagem anotada."""

import sys
from pathlib import Path

# Garante que a pasta `core/` esteja no path para imports locais
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import extract_features


def process_frame(
    frame: np.ndarray,
    recognizer,
    clf,
    label_encoder,
    timestamp_ms: int,
) -> np.ndarray:
    """Processa um frame BGR e retorna o frame com anotações de gestos.

    Args:
        frame: Imagem BGR (np.ndarray) já espelhada (flip).
        recognizer: Instância do GestureRecognizer do MediaPipe.
        clf: Classificador treinado (sklearn).
        label_encoder: LabelEncoder treinado.
        timestamp_ms: Timestamp em milissegundos para o modo VIDEO.

    Returns:
        np.ndarray: Frame BGR com landmarks desenhados e texto de predição.
    """
    annotated = frame.copy()

    # Converte para RGB e cria a imagem do MediaPipe
    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Extrai landmarks
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    mp_hands = mp.tasks.vision.HandLandmarksConnections

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Desenha landmarks e conexões
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extrai features e faz predição
            hand_label = result.handedness[i][0].category_name
            features = extract_features(hand_landmarks, hand_label)

            prediction_idx = clf.predict(features)[0]
            prediction_prob = np.max(clf.predict_proba(features))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            # Desenha texto com resultado
            color = (0, 255, 0)
            display_text = (
                f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
            )
            cv2.putText(
                annotated,
                display_text,
                (20, 50 + (i * 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    return annotated