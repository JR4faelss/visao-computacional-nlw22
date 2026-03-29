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
            Link(rel="stylesheet", href="/style.css?v=2"),
        ),
        Body(
            H1("🤚 Reconhecimento de Gestos"),
            Div("Iniciando...", id="status"),
            Div(
                Canvas(id="canvas", width="640", height="480"),
                Div(
                    Div(
                        Label("Qualidade (Compressão):", **{"for": "rng_quality"}, style="display:block; margin-bottom:0.2rem;"),
                        Div(
                            Input(type="range", id="rng_quality", min="10", max="100", value="70", step="1", style="width: 150px; cursor: pointer;"),
                            Span("70%", id="val_quality", style="margin-left: 0.5rem; color: #d4af37; font-weight: bold;"),
                            style="display:flex; align-items:center; margin-bottom:1rem;"
                        ),
                        Div(
                            Input(type="checkbox", id="chk_landmarks", checked=True, style="margin-right: 0.5rem; cursor: pointer;"),
                            Label("Mostrar Landmarks", **{"for": "chk_landmarks"}, style="cursor: pointer;"),
                            style="display:flex; align-items:center; margin-bottom:0.7rem;"
                        ),
                        Div(
                            Span("Desempenho:", style="margin-right: 0.5rem;"),
                            Span("0", id="val_fps", style="color: #d4af37; font-weight: 800; font-family: monospace; font-size: 1.1rem;"),
                            Span(" FPS", style="color: #5a6478; font-size: 0.8rem; margin-left: 0.2rem;"),
                            style="display:flex; align-items:flex-end;"
                        ),
                        id="controls", style="color: #e2e8f0; font-size: 0.95rem; width: 100%; border-bottom: 1px solid rgba(212, 175, 55, 0.3); padding-bottom: 1rem; margin-bottom: 1rem; position: relative; z-index: 100;"
                    ),
                    Div("---", id="gesture_text", style="position: relative; z-index: 10;"),
                    Img(id="gesture_img", style="display:none; max-width:300px; border-radius:12px; margin-top:0.5rem;"),
                    id="info_panel"
                ),
                id="main_container"
            ),
            Script(src="/main.js?v=4"),
        ),
    )


serve(port=5001)