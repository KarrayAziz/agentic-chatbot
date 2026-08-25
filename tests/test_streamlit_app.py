"""Small Streamlit smoke test that requires no live backend or paid API."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_shows_concise_backend_unavailable_state(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_API_URL", "http://127.0.0.1:9")
    app_path = (
        Path(__file__).parents[1]
        / "src"
        / "agentic_chatbot"
        / "streamlit_app.py"
    )

    app = AppTest.from_file(app_path, default_timeout=10).run()

    assert not app.exception
    assert "Cannot reach the FastAPI backend" in app.error[0].value
    assert "uv run uvicorn" in app.info[0].value
