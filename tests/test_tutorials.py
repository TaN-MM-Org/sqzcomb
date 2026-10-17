"""0.16: the tutorials in examples/ run and print what they document.

Each tutorial is run as a script; its output must equal the output
quoted in its own docstring (which the README repeats), and its key
numbers are checked against the true values the synthetic data were
made from, within the error bars the tutorial prints.
"""
import os
import re
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "examples")


def _run(name):
    path = os.path.join(EXAMPLES, name)
    if not os.path.exists(path):          # e.g. an sdist without examples
        pytest.skip(f"{name} not present")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, env=env, check=True, timeout=120)
    with open(path) as fh:
        src = fh.read()
    doc = src[src.index("Output (sqzcomb"):src.index('"""', 10)]
    quoted = [ln[4:] for ln in doc.splitlines()[2:] if ln.strip()]
    assert out.stdout.splitlines() == quoted
    return out.stdout


def _pm(text, label):
    """(value, error) printed as '<label> ... = v +/- s'."""
    m = re.search(re.escape(label) + r"[^=]*=\s*([-+0-9.]+) \+/- ([0-9.]+)",
                  text)
    assert m, label
    return float(m.group(1)), float(m.group(2))


def test_ring_calibration_tutorial():
    out = _run("tutorial_ring_calibration.py")
    k, sk = _pm(out, "linewidth kappa")
    e, se = _pm(out, "escape efficiency")
    assert abs(k - 100.0) < 3 * sk          # MHz, true value 100
    assert abs(e - 0.8) < 3 * se            # true value 0.8
    assert "mu = 0.4481 +/- 0.0010" in out
    assert "-5.10 dB at mu = 0.8325" in out


def test_squeezing_measurement_tutorial():
    out = _run("tutorial_squeezing_measurement.py")
    for label, true in (("fit: mu", 0.6), ("eta", 0.75), ("kappa", 20.0)):
        v, s = _pm(out, label)
        assert abs(v - true) < 3 * s, (label, v, s)
    assert "chi^2 / dof = 57.6 / 57" in out
    assert "pair at 1 MHz: eta = 0.750 +/- 0.006" in out
