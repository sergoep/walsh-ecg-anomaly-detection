"""Optional PTB-XL loading hook.

This file is intentionally a hook, not the default demo. Full PTB-XL experiments
require downloading/streaming records, patient-wise splitting, and careful clinical
label handling. The journal submission should cite only archived runs that exactly
match the reported tables.
"""

from __future__ import annotations


def read_ptbxl_record_example(record_name="records100/00000/00001_lr", pn_dir="ptb-xl/1.0.3"):
    """Read one PTB-XL low-resolution record through WFDB.

    Requires: pip install wfdb
    """

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install WFDB first: pip install wfdb") from exc

    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    return record.p_signal, record
