"""Generate HDECAY input files (hdecay.in)."""

from .config import SM_PARAMS, PARAM_ORDER, INT_PARAMS


def format_value(name: str, value) -> str:
    """Format a single parameter line for hdecay.in.

    HDECAY FORMAT 100 (floats): FORMAT(10X,G30.20)
    HDECAY FORMAT 101 (ints):   FORMAT(10X,I30)
    """
    if name in INT_PARAMS:
        return f"{'':10s}{int(value):30d}"
    else:
        return f"{'':10s}{float(value):30.20E}"


def generate_input(overrides: dict | None = None) -> str:
    """Generate a complete hdecay.in file content.

    Args:
        overrides: dict of parameter name -> value to override SM defaults.

    Returns:
        String content of hdecay.in file.
    """
    params = dict(SM_PARAMS)
    if overrides:
        params.update(overrides)

    lines = []
    for name in PARAM_ORDER:
        lines.append(format_value(name, params[name]))
    return "\n".join(lines) + "\n"
