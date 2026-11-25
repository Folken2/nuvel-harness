"""The examples' own tests, each in a fresh process: an example is a product, and names its
own wire namespace, where these tests have set a neutral one."""

import pathlib
import subprocess
import sys

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def test_the_assistant_example_runs_end_to_end():
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(EXAMPLES / "test_assistant.py")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=EXAMPLES,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
