"""Módulo responsável por carregar os modelos de reconhecimento de gestos."""

from pathlib import Path
import joblib
import mediapipe as mp

# Pasta de modelos: ../modelos/ relativo a este arquivo
_MODELS_DIR = Path(__file__).resolve().parent.parent / "modelos"

# Caminhos absolutos para os modelos
MP_MODEL_PATH = str(_MODELS_DIR / "gesture_recognizer.task")
CUSTOM_MODEL_PATH = str(_MODELS_DIR / "gesture_model.joblib")
ENCODER_PATH = str(_MODELS_DIR / "label_encoder.joblib")

def _load_mp_model_bytes(mp_path: str = MP_MODEL_PATH) -> bytes:
    """Lê o arquivo .task como bytes — contorna o bug do MediaPipe com
    caminhos que contêm espaços ou caracteres especiais (ex: 'ã')."""
    with open(mp_path, "rb") as f:
        return f.read()


def check_model_files(
    mp_path: str = MP_MODEL_PATH,
    custom_path: str = CUSTOM_MODEL_PATH,
    encoder_path: str = ENCODER_PATH,
) -> bool:
    """Verifica se todos os arquivos de modelo existem."""
    return all(Path(p).exists() for p in [mp_path, custom_path, encoder_path])


def load_custom_model(
    custom_path: str = CUSTOM_MODEL_PATH,
    encoder_path: str = ENCODER_PATH,
):
    """Carrega o classificador e o label encoder treinados."""
    clf = joblib.load(custom_path)
    label_encoder = joblib.load(encoder_path)
    return clf, label_encoder


def create_gesture_recognizer(mp_path: str = MP_MODEL_PATH):
    """Cria e retorna as opções do GestureRecognizer do MediaPipe (modo VIDEO)."""
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=_load_mp_model_bytes(mp_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return options


def create_gesture_recognizer_image(mp_path: str = MP_MODEL_PATH):
    """Cria e retorna as opções do GestureRecognizer do MediaPipe (modo IMAGE).

    Modo IMAGE é adequado para processar frames independentes (sem timestamp).
    Usado pelo webapp Streamlit com streamlit-webrtc.
    """
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=_load_mp_model_bytes(mp_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return options

