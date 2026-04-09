"""Parse HDECAY output files (br.sm1, br.sm2)."""

import os

from .config import BR_SM1_CHANNELS, BR_SM2_CHANNELS


def _parse_br_file(filepath: str, channels: list[str]) -> dict[str, float]:
    """Parse an HDECAY branching ratio file.

    The format is: header lines, then data rows with
    MHSM followed by BR values in scientific notation.
    """
    results = {}
    with open(filepath) as f:
        lines = f.readlines()

    # Find the data line (skip headers — lines starting with text or underscores)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try to parse as numbers
        try:
            values = line.split()
            float(values[0])  # first column is MHSM
        except (ValueError, IndexError):
            continue

        # This is a data line
        values = [float(v) for v in values]
        # values[0] is MHSM, values[1:] are BRs (and possibly WIDTH)
        for i, ch in enumerate(channels):
            if i + 1 < len(values):
                results[ch] = values[i + 1]
        break  # we only have one mass point

    return results


def parse_output(work_dir: str) -> dict[str, float]:
    """Parse both br.sm1 and br.sm2 from a work directory.

    Returns:
        Dict mapping channel name -> branching ratio.
        Also includes 'width' key for the total width.
    """
    branching_ratios = {}

    sm1_path = os.path.join(work_dir, "br.sm1")
    if os.path.exists(sm1_path):
        branching_ratios.update(_parse_br_file(sm1_path, BR_SM1_CHANNELS))

    sm2_path = os.path.join(work_dir, "br.sm2")
    if os.path.exists(sm2_path):
        sm2_channels = BR_SM2_CHANNELS + ["width"]
        branching_ratios.update(_parse_br_file(sm2_path, sm2_channels))

    return branching_ratios
