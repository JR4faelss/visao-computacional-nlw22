"""Inicialização centralizada de modelos (MediaPipe + modelo customizado)."""

import mediapipe as mp

from model_loader import check_model_files, load_custom_model, create_gesture_recognizer_image


def init_all():
    """Verifica os arquivos, carrega o classificador e cria o recognizer.

    Returns:
        tuple: (recognizer, clf, label_encoder)

    Raises:
        AssertionError: Se algum arquivo de modelo estiver faltando.
    """
    assert check_model_files(), "Arquivos de modelo não encontrados."

    clf, label_encoder = load_custom_model()
    options = create_gesture_recognizer_image()
    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    return recognizer, clf, label_encoder
