"""FastHTML backend mínimo para reconhecimento de gestos."""

import sys
from pathlib import Path

# Adiciona a pasta core ao path
_CORE_DIR = str(Path(__file__).resolve().parent / "core")
sys.path.insert(0, _CORE_DIR)

from init_models import init_all
from websocket_handler import websocket_endpoint

from fasthtml.common import *
from starlette.routing import WebSocketRoute

# ── Caminhos ──
_BASE_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _BASE_DIR / "assets"

# ── Carrega modelos ──
recognizer, clf, label_encoder = init_all()

# ── App FastHTML com rota WebSocket e arquivos estáticos ──
app, rt = fast_app(
    routes=[WebSocketRoute("/ws", endpoint=websocket_endpoint)],
    static_path=str(_ASSETS_DIR),
)

# Injeta modelos no app.state para o handler acessar
app.state.recognizer = recognizer
app.state.clf = clf
app.state.label_encoder = label_encoder


@rt("/")
def get():
    return Html(
        Head(
            Title("Gesture Recognition"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Link(rel="stylesheet", href="/style.css"),
        ),
        Body(
            H1("🤚 Reconhecimento de Gestos"),
            Div("Iniciando...", id="status"),
            Canvas(id="canvas", width="640", height="480"),
            Div("---", id="gesture_text"),
            Img(id="gesture_img", style="display:none; max-width:300px; border-radius:12px; margin-top:0.5rem;"),
            Script(src="/main.js"),
        ),
    )


serve(port=5001)