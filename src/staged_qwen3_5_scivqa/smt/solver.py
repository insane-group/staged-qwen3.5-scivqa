"""SMT-LIB solver integration via cvc5 subprocess."""

import os
import subprocess  # nosec B404
import tempfile

from staged_qwen3_5_scivqa.config import CVC5_PATH


def validate_smt(code: str) -> tuple[bool, str]:
    """Validate SMT-LIB code by executing cvc5.

    Args:
        code: The SMT-LIB code string to validate.

    Returns:
        Tuple of (is_satisfiable, output_message).

    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as tf:
        tf.write(code)
        temp_path = tf.name

    try:
        result = subprocess.run(  # nosec B603
            [
                str(CVC5_PATH),
                "--lang",
                "smt2",
                "--produce-models",
                "--incremental",
                temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        lines = [line for line in stdout.split("\n") if line.strip()]
        status = lines[0].lower() if lines else ""

        if stderr or "error" in stdout.lower():
            return False, stderr if stderr else stdout
        if status == "sat":
            return True, stdout
        elif status == "unsat":
            return False, stdout
        elif status == "unknown":
            return False, stdout
        else:
            return False, f"Unexpected Solver Output: {stdout}"
    except subprocess.TimeoutExpired:
        return (
            False,
            "Timeout: The solver took too long (potential infinite search space).",
        )
    except Exception as e:
        return False, f"Internal Execution Error: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
