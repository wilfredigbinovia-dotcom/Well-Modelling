"""Project files: the whole model (inputs, tables, PRP text, accepted matches) as one JSON document.

JSON rather than pickle on purpose - opening a project file never executes code. DataFrames keep their column
dtypes and index; numpy values become plain numbers; NaN is written as JSON null inside tables and as the
non-standard literal NaN for loose numbers (Python's json reads it back).
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FORMAT = "fluxwell-project"
LEGACY_FORMATS = ("wellmodel-project",)   # projects saved before the rename to FluxWell
VERSION = 1


def _enc(o):
    if isinstance(o, pd.DataFrame):
        return {"__df__": json.loads(o.to_json(orient="split", date_format="iso", double_precision=15)),
                "dtypes": {str(c): str(t) for c, t in o.dtypes.items()}, "index_name": o.index.name}
    if isinstance(o, dict):
        return {str(k): _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_enc(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    raise TypeError(f"cannot store a {type(o).__name__} in a project file")


def _dec(o):
    if isinstance(o, dict):
        if "__df__" in o:
            d = o["__df__"]
            df = pd.DataFrame(d["data"], columns=d["columns"], index=d["index"])
            for c, t in o.get("dtypes", {}).items():
                if c in df:
                    try:
                        df[c] = df[c].astype(object) if t == "object" else df[c].astype(t)
                    except (TypeError, ValueError):
                        pass
                    if t == "object":                       # keep empty text cells as None, not NaN
                        df[c] = df[c].where(df[c].notna(), None)
            df.index.name = o.get("index_name")
            return df
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(v) for v in o]
    return o


def dumps(state: dict, name: str = "") -> str:
    """Serialise a project state dict (sections: inputs, widgets, tables, extras)."""
    doc = {"format": FORMAT, "version": VERSION, "name": name,
           "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"), **_enc(state)}
    return json.dumps(doc, indent=1)


def loads(text: str | bytes) -> dict:
    """Parse a project file. Raises ValueError for anything that is not a fluxwell project."""
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"not a project file ({e})") from None
    if not isinstance(doc, dict) or doc.get("format") not in (FORMAT, *LEGACY_FORMATS):
        raise ValueError("not a fluxwell project file")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"project file version {doc.get('version')} is newer than this tool ({VERSION})")
    out = _dec({k: v for k, v in doc.items() if k not in ("format", "version")})
    for k in ("inputs", "widgets", "tables", "extras"):
        out.setdefault(k, {})
    return out
