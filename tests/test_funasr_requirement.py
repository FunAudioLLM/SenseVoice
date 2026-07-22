from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_funasr_minimum_version_matches_current_examples():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "funasr>=1.3.23" in requirements
    assert "funasr>=1.1.3" not in requirements


def test_readmes_explain_upgrade_for_existing_installs():
    for readme in ["README.md", "README_zh.md", "README_ja.md"]:
        text = (ROOT / readme).read_text()
        assert 'pip install -U "funasr>=1.3.23"' in text
