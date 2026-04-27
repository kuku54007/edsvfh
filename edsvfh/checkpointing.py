from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=str(path.parent), encoding='utf-8') as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        temp_name = tmp.name
    Path(temp_name).replace(path)


def atomic_write_pickle(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=str(path.parent)) as tmp:
        pickle.dump(payload, tmp)
        tmp.flush()
        temp_name = tmp.name
    Path(temp_name).replace(path)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_pickle(path: str | Path) -> Any:
    with Path(path).open('rb') as f:
        return pickle.load(f)
