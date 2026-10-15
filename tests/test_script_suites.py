"""Run the four script-style checks under pytest.

test_numerics.py, test_m5_traces.py, test_gated_cells.py and test_gated_numerics.py are scripts
(``python -m tests.<name>``) with no test functions of their own. Each test below runs one of
them as ``__main__`` and fails if the script reports a failure, so that

    python -m pytest tests -q --import-mode=importlib

covers all seven files.
"""
import contextlib
import io
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(name):
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(os.path.join(HERE, name + ".py"), run_name="__main__")
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    out = buf.getvalue()
    print(out)
    return code, out


def test_numerics_script():
    # exact RTRL vs BPTT, SK-RTRL(r=n) vs exact, Lemma-1 certificate checks
    code, out = _run("test_numerics")
    assert code == 0 and "ALL PASS" in out


def test_m5_traces_script():
    # LRU/RTU eligibility traces vs autograd, T-maze environment
    code, out = _run("test_m5_traces")
    assert code == 0 and "ALL OK" in out


def test_gated_cells_script():
    # GRU/LSTM Jacobian pieces against autograd
    code, out = _run("test_gated_cells")
    assert code == 0 and "PASS" in out


def test_gated_numerics_script():
    # estimators on gated cells: exact vs BPTT, SK-RTRL(r=n) vs exact, certificate
    code, out = _run("test_gated_numerics")
    assert code == 0
