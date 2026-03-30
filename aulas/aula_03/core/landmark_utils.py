"""Utilitários para extração de features a partir de landmarks do MediaPipe."""

import warnings
import numpy as np

# Suprime o UserWarning do sklearn sobre feature names (modelo treinado com
# DataFrame, mas passamos numpy array — funciona normalmente, é só um aviso)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


def _extract_hand_features(hand_landmarks, handedness_label: str) -> list:
    """Extrai features de UMA mão: [handedness, x0, y0, z0, ..., x20, y20, z20] = 64 valores."""
    handedness_val = 0 if handedness_label == "Left" else 1
    features = [handedness_val]
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features  # 1 + 63 = 64 valores


def _empty_hand() -> list:
    """Retorna features zeradas para quando não há segunda mão."""
    return [0] + [0.0] * 63  # 64 zeros


def extract_features_two_hands(result) -> np.ndarray:
    """Extrai o vetor de 129 features do resultado do MediaPipe.

    Formato: [num_hands, hand1_handedness, hand1_x0..z20, hand2_handedness, hand2_x0..z20]
    Se apenas 1 mão for detectada, a segunda é preenchida com zeros.

    Args:
        result: Resultado do GestureRecognizer do MediaPipe.

    Returns:
        np.ndarray de shape (1, 129) pronto para clf.predict().
    """
    if not result.hand_landmarks:
        return None

    num_hands = len(result.hand_landmarks)

    # Mão 1 — usa a ordem do MediaPipe (idêntico ao treinamento)
    hand1_name = result.handedness[0][0].category_name
    hand1_features = _extract_hand_features(result.hand_landmarks[0], hand1_name)

    # Mão 2 (ou zeros se não existir)
    if num_hands >= 2:
        hand2_name = result.handedness[1][0].category_name
        hand2_features = _extract_hand_features(result.hand_landmarks[1], hand2_name)
    else:
        hand2_features = _empty_hand()

    row = [num_hands] + hand1_features + hand2_features  # 1 + 64 + 64 = 129
    return np.array(row).reshape(1, -1)
