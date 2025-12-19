import os
import time
import socket
import signal
import threading
import json
import platform
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import requests

try:
    import psutil
except ImportError:
    psutil = None

# =========================
#   CONFIG
# =========================

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080").rstrip("/")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
AGENT_TYPE = os.getenv("AGENT_TYPE", "lite")
AGENT_VERSION = os.getenv("AGENT_VERSION", "lite")
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC", "5"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "3"))

# "Lite" guardrails
BUSY_CPU_THRESHOLD = float(os.getenv("BUSY_CPU_THRESHOLD", "75"))  # percent
DISABLE_ON_BATTERY = os.getenv("DISABLE_ON_BATTERY", "0").strip() in ("1", "true", "True", "yes", "YES")

# Tokenize/chunk config
CHUNK_BYTES = int(os.getenv("CHUNK_BYTES", "2048"))

# Cache-ish size classes (you mentioned L2/L3 thinking)
L2_TARGET_BYTES = int(os.getenv("L2_TARGET_BYTES", str(256 * 1024)))     # 256KB
L3_TARGET_BYTES = int(os.getenv("L3_TARGET_BYTES", str(8 * 1024 * 1024))) # 8MB

SAFE_MAX_BYTES = int(os.getenv("SAFE_MAX_BYTES", str(2 * 1024 * 1024)))  # 2MB soft
HARD_MAX_BYTES = int(os.getenv("HARD_MAX_BYTES", str(8 * 1024 * 1024)))  # 8MB hard (route to heavy)

AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

# Worker profile (controller can use this for scheduling/routing)
WORKER_PROFILE: Dict[str, Any] = {
    "profile": "lite",
    "limits": {
        "max_payload_bytes": SAFE_MAX_BYTES,
        "hard_max_payload_bytes": HARD_MAX_BYTES,
        "l2_target_bytes": L2_TARGET_BYTES,
        "l3_target_bytes": L3_TARGET_BYTES,
    },
}

CONFIG: Dict[str, Any] = {
    "agent_type": AGENT_TYPE,
    "version": AGENT_VERSION,
}

CAPABILITIES: Dict[str, Any] = {
    "cpu": True,
    "gpu": False,
    "disk_io": False,
    "net_io": True,  # controller comms
}

BASE_LABELS: Dict[str, Any] = {
    "agent_type": CONFIG["agent_type"],
    "version": CONFIG["version"],
    "worker_profile": WORKER_PROFILE,
}

if AGENT_LABELS_RAW.strip():
    for item in AGENT_LABELS_RAW.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            BASE_LABELS[k.strip()] = v.strip()
        else:
            BASE_LABELS[item.strip()] = True

# =========================
#   OPS IMPLEMENTATIONS
# =========================

def op_echo(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Required for speed tests / sanity checks."""
    return {"ok": True, "echo": payload}

def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bread-and-butter tokenize: simple whitespace tokenize + chunking.
    Cache-friendly. No external deps.
    """
    text = str(payload.get("text", ""))
    tokens = text.split()

    encoded = text.encode("utf-8", errors="ignore")
    chunks: List[str] = []
    for i in range(0, len(encoded), CHUNK_BYTES):
        chunk_bytes = encoded[i : i + CHUNK_BYTES]
        chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))

    return {
        "ok": True,
        "tokens": tokens,
        "length": len(tokens),
        "chunks": chunks,
        "chunk_bytes": CHUNK_BYTES,
        "total_bytes": len(encoded),
        "num_chunks": len(chunks),
    }

_POSITIVE_WORDS = {"good", "great", "excellent", "awesome", "love", "like"}
_NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "dislike"}

def op_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lite classifier: deterministic keyword heuristic (no ML deps).
    """
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"ok": True, "label": "NEUTRAL", "score": 0.0}

    tokens = [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]
    pos_hits = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_hits = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

    if pos_hits > neg_hits:
        label = "POSITIVE"
    elif neg_hits > pos_hits:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    score = float(pos_hits - neg_hits)
    return {"ok": True, "label": label, "score": score, "pos_hits": pos_hits, "neg_hits": neg_hits}

def op_map_route(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route suggestion op: keep lite work inside size limits.
    """
    limits = WORKER_PROFILE.get("limits", {})
    max_bytes = int(limits.get("max_payload_bytes", SAFE_MAX_BYTES))

    text = str(payload.get("text", "") or "")
    approx_bytes = int(payload.get("payload_bytes") or len(text.encode("utf-8")))

    if approx_bytes <= L2_TARGET_BYTES:
        size_class = "l2"
    elif approx_bytes <= L3_TARGET_BYTES:
        size_class = "l3"
    else:
        size_class = "ram"

    if approx_bytes > HARD_MAX_BYTES:
        return {"ok": True, "route": "heavy", "reason": "exceeds_hard_limit"}
    elif approx_bytes > max_bytes and size_class == "ram":
        return {"ok": True, "route": "heavy", "reason": "exceeds_soft_cpu_limit"}

    return {"ok": True, "route": f"local_{size_class}", "reason": "within_cpu_limits"}

def op_fibonacci(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lite-safe fibonacci: iterative, bounded.
    payload: { "n": int }
    """
    try:
        n = int(payload.get("n", 0))
    except Exception:
        return {"ok": False, "error": "invalid_n"}

    # Hard cap to keep this lite-safe
    if n < 0:
        return {"ok": False, "error": "n_must_be_nonnegative"}
    if n > 200000:
        return {"ok": False, "error": "n_too_large_for_lite"}

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return {"ok": True, "n": n, "value": a}

def op_prime_factor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lite-safe prime factorization: trial division with caps.
    payload: { "n": int }
    """
    try:
        n = int(payload.get("n", 0))
    except Exception:
        return {"ok": False, "error": "invalid_n"}

    if n <= 1:
        return {"ok": True, "n": n, "factors": []}

    # Cap to avoid pathological runtimes on lite endpoints
    if n > 10**12:
        return {"ok": False, "error": "n_too_large_for_lite"}

    factors: List[int] = []
    x = n

    while x % 2 == 0:
        factors.append(2)
        x //= 2

    f = 3
    # simple guardrail on iterations
    steps = 0
    max_steps = 5_000_000

    while f * f <= x:
        while x % f == 0:
            factors.append(f)
            x //= f
        f += 2
        steps += 1
        if steps > max_steps:
            return {"ok": False, "error": "factorization_step_limit"}

    if x > 1:
        factors.append(x)

    return {"ok": True, "n": n, "factors": factors}

def op_csv_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lite CSV sharding: operates on CSV text (no disk).
    payload:
      - csv_text: str   (preferred)
      - text: str       (fallback)
      - shard_index: int (0-based)
      - shard_count: int
      - has_header: bool (default True)
    """
    csv_text = payload.get("csv_text")
    if csv_text is None:
        csv_text = payload.get("text", "")
    csv_text = str(csv_text)

    # Hard cap input size for lite
    b = len(csv_text.encode("utf-8", errors="ignore"))
    if b > SAFE_MAX_BYTES:
        return {"ok": False, "error": "csv_text_too_large_for_lite", "bytes": b, "max_bytes": SAFE_MAX_BYTES}

    try:
        shard_index = int(payload.get("shard_index", 0))
        shard_count = int(payload.get("shard_count", 1))
    except Exception:
        return {"ok": False, "error": "invalid_shard_params"}

    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        return {"ok": False, "error": "shard_index_out_of_range"}

    has_header = bool(payload.get("has_header", True))

    lines = [ln for ln in csv_text.splitlines() if ln.strip() != ""]
    if not lines:
        return {"ok": True, "rows": 0, "shard_rows": 0, "shard_index": shard_index, "shard_count": shard_count, "csv_text": ""}

    header = lines[0] if has_header else None
    data = lines[1:] if has_header else lines

    total = len(data)
    start = (total * shard_index) // shard_count
    end = (total * (shard_index + 1)) // shard_count

    shard_lines = data[start:end]
    out_lines: List[str] = []
    if header is not None:
        out_lines.append(header)
    out_lines.extend(shard_lines)

    out = "\n".join(out_lines)

    # Cap output too
    out_b = len(out.encode("utf-8", errors="ignore"))
    if out_b > SAFE_MAX_BYTES:
        return {"ok": False, "error": "csv_shard_output_too_large_for_lite", "bytes": out_b, "max_bytes": SAFE_MAX_BYTES}

    return {
        "ok": True,
        "rows": total,
        "shard_rows": len(shard_lines),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "start": start,
        "end": end,
        "csv_text": out,
    }

OPS = {
    "echo": op_echo,
    "map_tokenize": op_map_tokenize,
    "map_classify": op_map_classify,
    "map_route": op_map_route,
    "csv_shard": op_csv_shard,
    "fibonacci": op_fibonacci,
    "prime_factor": op_prime_factor,
}

# =========================
#   METRICS & GUARD
# =========================

_metrics_lock = threading.Lock()
_tasks_completed = 0
_tasks_failed = 0
_task_durations: List[float] = []

def _record_task_result(duration_ms: float, ok: bool):
    global _tasks_completed, _tasks_failed
    with _metrics_lock:
        if ok:
            _tasks_completed += 1
        else:
            _tasks_failed += 1
        _task_durations.append(duration_ms)
        if len(_task_durations) > 100:
            _task_durations.pop(0)

def _collect_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if psutil:
        metrics["cpu_util"] = psutil.cpu_percent(interval=0.0) / 100.0
        try:
            metrics["ram_mb"] = int(psutil.virtual_memory().used / 1048576)
        except Exception:
            pass
    with _metrics_lock:
        metrics["tasks_completed"] = _tasks_completed
        metrics["tasks_failed"] = _tasks_failed
    return metrics

def system_allows_work() -> bool:
    """Returns False if the machine is busy or on battery (if enabled)."""
    if not psutil:
        return True

    if psutil.cpu_percent(interval=0.1) >= BUSY_CPU_THRESHOLD:
        return False

    if DISABLE_ON_BATTERY:
        try:
            batt = psutil.sensors_battery()
            if batt and not batt.power_plugged:
                return False
        except Exception:
            pass

    return True

# =========================
#   NETWORKING
# =========================

def _post_json(path: str, payload: Dict[str, Any]):
    try:
        r = requests.post(f"{CONTROLLER_URL}{path}", json=payload, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json() if r.content else None
    except Exception:
        return None

def _get_json(path: str, params: Dict[str, Any]):
    try:
        r = requests.get(f"{CONTROLLER_URL}{path}", params=params, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json() if r.content else None
    except Exception:
        return None

def register_agent():
    print(f"[*] Registering {AGENT_NAME}...")
    _post_json("/agents/register", {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics(),
    })

# =========================
#   WORK LOOP
# =========================

_running = True

def worker_loop():
    global _running
    while _running:
        if not system_allows_work():
            time.sleep(0.5)
            continue

        task = _get_json("/task", {"agent": AGENT_NAME})
        if not task:
            time.sleep(0.05)
            continue

        try:
            op = task.get("op")
            payload = task.get("payload", {}) or {}
            if op not in OPS:
                _post_json("/result", {
                    "id": task.get("id"),
                    "agent": AGENT_NAME,
                    "ok": False,
                    "result": None,
                    "error": f"unknown_op:{op}",
                    "duration_ms": 0.0,
                })
                continue

            start = time.time()
            res = OPS[op](payload)
            duration = (time.time() - start) * 1000.0
            ok = bool(res.get("ok", True))

            _record_task_result(duration, ok)
            _post_json("/result", {
                "id": task.get("id"),
                "agent": AGENT_NAME,
                "ok": ok,
                "result": res if ok else None,
                "error": res.get("error") if not ok else None,
                "duration_ms": duration,
            })
        except Exception as e:
            _record_task_result(0.0, False)
            _post_json("/result", {
                "id": task.get("id"),
                "agent": AGENT_NAME,
                "ok": False,
                "result": None,
                "error": f"exception:{type(e).__name__}:{e}",
                "duration_ms": 0.0,
            })

# =========================
#   MAIN
# =========================

def _stop(*_args, **_kwargs):
    global _running
    print("[agent] stop signal received, shutting down...")
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

def main():
    register_agent()

    def _hb_loop():
        while _running:
            _post_json("/agents/heartbeat", {
                "agent": AGENT_NAME,
                "labels": BASE_LABELS,
                "capabilities": CAPABILITIES,
                "worker_profile": WORKER_PROFILE,
                "metrics": _collect_metrics(),
            })
            time.sleep(HEARTBEAT_SEC)

    threading.Thread(target=_hb_loop, daemon=True).start()
    worker_loop()

if __name__ == "__main__":
    main()
