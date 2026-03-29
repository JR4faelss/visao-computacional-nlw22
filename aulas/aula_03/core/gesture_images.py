"""Mapeamento de nomes de gestos para imagens em assets/imagens/."""

# gesture_name (do label_encoder) → nome do arquivo na pasta assets/imagens/
GESTURE_IMAGE_MAP = {
    "ABSOLUTE CINEMA": "absolute-cinema.webp",
    "fuck you":        "fuck_you.gif",
    "hangloose":       "hangloose.webp",
    "hare":            "hare.webp",
    "heart":           "heart.jpg",
    "horse":           "horse.webp",
    "legal":           "legal.gif",
    "paia":            "paia.jpg",
    "palma aberta":    "palma aberta.jpg",
    "paz":             "paz.jpg",
    "punho fechado":   "punho-fechado.jpg",
    "rock":            "rock.webp",
    "tiger":           "tiger.webp",
}


def get_gesture_image(gesture_name: str) -> str | None:
    """Retorna o nome de arquivo da imagem para o gesto, ou None se não houver."""
    return GESTURE_IMAGE_MAP.get(gesture_name)
