"""O4↔O2 constructive ambiguity-pair gate (spec §8.1 P4) — paper-artifact protection.

The matching script (kino_vla.eval.ambiguity) must show the compliance/tether pair is
proprioceptively indistinguishable (matched tangential-resistance, base-height, and slip
traces) while visually well-separated. Per the kino-fail-operators skill this script is a
paper artifact: this test pins its verdict against the configs tolerances so the matching
can never silently regress.
"""

from __future__ import annotations

from kino_vla.eval.ambiguity import build_matched_pair, match_ambiguity_pair
from kino_vla.utils.config import load_config

TOL = load_config("operators/apply_tolerances.yaml").ambiguity_o4_o2


def test_matched_pair_proprioception_indistinguishable():
    o2, o4 = build_matched_pair()
    result = match_ambiguity_pair(o2, o4)
    assert result.resistance_curve_max_abs_diff_n <= float(TOL.resistance_curve_max_abs_diff_n)
    assert result.base_height_max_abs_diff_m <= float(TOL.base_height_max_abs_diff_m)
    assert result.slip_max_abs_diff <= float(TOL.slip_max_abs_diff)


def test_matched_pair_visually_separable():
    o2, o4 = build_matched_pair()
    result = match_ambiguity_pair(o2, o4)
    # Vision MUST be able to tell mud from adhesive even though proprioception cannot.
    assert result.appearance_separation >= float(TOL.min_appearance_separation)


def test_appearance_classes_differ():
    o2, o4 = build_matched_pair()
    assert o2.scene_region().appearance_class != o4.scene_region().appearance_class


def test_match_script_runs_and_passes():
    import subprocess
    import sys

    from kino_vla.utils.config import REPO_ROOT

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ambiguity_match.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: ambiguity pair matched" in proc.stdout
