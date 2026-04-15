from __future__ import annotations

from pathlib import Path


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value == "[]":
        return []
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _load_yaml_fallback(path: Path) -> dict:
    data: dict = {}
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Invalid YAML list item in {path}: {raw_line}")
            data[current_list_key].append(_parse_scalar(stripped[2:]))
            continue
        current_list_key = None
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML mapping line in {path}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_scalar(value)
    return data


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        return _load_yaml_fallback(path)
