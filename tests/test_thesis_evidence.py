from pathlib import Path

import pytest

from unified_multi_agent_coordination.thesis_evidence import render, write_or_check


def test_generated_thesis_evidence_is_deterministic_and_current(tmp_path):
    manifest = Path("evidence-manifest.json")
    expected = render(manifest, Path("."))
    output = tmp_path / "macros.tex"
    write_or_check(manifest, Path("."), output, check=False)
    assert output.read_text() == expected
    write_or_check(manifest, Path("."), output, check=True)

    output.write_text("stale")
    with pytest.raises(RuntimeError, match="differ"):
        write_or_check(manifest, Path("."), output, check=True)
