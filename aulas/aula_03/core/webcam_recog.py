"""Loop principal da webcam para reconhecimento de gestos customizado."""

import sys
from pathlib import Path

# Garante que a pasta `core/` esteja no path para imports locais
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mediapipe as mp

from model_loader import check_model_files, load_custom_model, create_gesture_recognizer
from process_frame import process_frame


def main():
    if not check_model_files():
        print("Erro: Um ou mais arquivos de modelo não foram encontrados (.task, .joblib).")
        return

    # Carrega modelos
    print("--- Carregando modelos customizados ---")
    clf, label_encoder = load_custom_model()
    options = create_gesture_recognizer()

    # Abre webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    GestureRecognizer = mp.tasks.vision.GestureRecognizer

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Processa o frame (imagem → imagem)
            annotated = process_frame(frame, recognizer, clf, label_encoder, timestamp_ms)

            cv2.imshow("Custom Gesture Recognition", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
