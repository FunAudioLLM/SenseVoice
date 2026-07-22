from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pull_request_template_keeps_sensevoice_validation_visible():
    template = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
    assert template.exists()

    text = template.read_text()
    required = [
        "Summary",
        "User impact",
        "Model, API, and runtime impact",
        "Validation",
        "Screenshots, logs, or transcripts",
    ]
    for marker in required:
        assert marker in text
