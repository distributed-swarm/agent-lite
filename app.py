import os
import time
import socket
import signal
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from ops import csv_shard

import requests

try:
    import psutil
except ImportError:
    psutil = None

PROGRAM_DATA_DIR = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "AgentLite"
PROGRAM_DATA_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG_FILE = PROGRAM_DATA_DIR / "agent.log"

_log_lock = threading.Lock()

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        AGENT_LOG_FILE.open("a", encoding="utf-8").write(f"{ts} [agent-lite] {msg}\n")

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080").rstrip("/")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())

BUSY_CPU_THRESHOLD = float(os.getenv("BUSY_CPU_THRESHOLD", "75"))
DISABLE_ON_BATTERY = os.getenv("DISABLE_ON_BATTERY", "0").strip().lower() in ("1", "true", "yes")

HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC", "5"))
POLL_IDLE_SEC = float(os.getenv("POLL_IDLE_SEC", "0.05"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "3"))

CHUNK_BYTES = int(os.getenv("CHUNK_BYTES", "2048"))
L2_TARGET_BYTES = int(os.getenv("L2_TARGET_BYTES", str(256 * 1024)))
L3_TARGET_BYTES = int(os.getenv("L3_TARGET_BYTES", str(8 * 1024 * 1024)))
SAFE_MAX_BYTES = int(os.getenv("SAFE_MAX_BYTES", str(2 * 1024 * 1024)))
HARD_MAX_BYTES = int(os.getenv("HARD_MAX_BYTES", str(8 * 1024 * 1024)))

WORKER_PROFILE: Dict[str, Any] = {
    "profile": "lite",
    "limits": {
        "max_payload_bytes": SAFE_MAX_BYTES,
        "hard_max_payload_bytes": HARD_MAX_BYTES,
        "l2_target_bytes": L2_TARGET_BYTES,
        "l3_target_bytes": L3_TARGET_BYTES,
    },
}

CAPABILITIES: Dict[str, Any] = {"cpu": True, "gpu": False}

LABELS: Dict[str, Any] = {
    "agent_type": "lite",
    "version": "lite",
    "worker_profile": WORKER_PROFILE,
}

def op_echo(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "echo": payload}

def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    tokens = text.split()
    encoded = text.encode("utf-8", errors="ignore")
    chunks: List[str] = []
    for i in range(0, len(encoded), CHUNK_BYTES):
        chunks.append(encoded[i:i+CHUNK_BYTES].decode("utf-8", errors="ignore"))
    return {"ok": True, "tokens": tokens, "length": len(tokens), "chunks": chunks}

def op_fibonacci(payload: Dict[str, Any]) -> Dict[str, Any]:
    n = int(payload.get("n", 0))
    if n < 0 or n > 200000:
        return {"ok": False, "error": "n_out_of_range_for_lite"}
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return {"ok": True, "n": n, "value": a}

def op_prime_factor(payload: Dict[str, Any]) -> Dict[str, Any]:
    n = int(payload.get("n", 0))
    if n <= 1:
        return {"ok": True, "n": n, "factors": []}
    if n > 10**12:
        return {"ok": False, "error": "n_too_large_for_lite"}

    factors: List[int] = []
    x = n
    while x % 2 == 0:
        factors.append(2)
        x //= 2
    f = 3
    while f * f <= x:
        while x % f == 0:
            factors.append(f)
            x //= f
        f += 2
    if x > 1:
        factors.append(x)
    return {"ok": True, "n": n, "factors": factors}

OPS = {
    "echo": op_echo,
    "map_tokenize": op_map_tokenize,
    "fibonacci": op_fibonacci,
    "prime_factor": op_prime_factor,
    "read_csv_shard": csv_shard.op_read_csv_shard
}

_session = requests.Session()

_metrics_lock = threading.Lock()
_tasks_completed = 0
_tasks_failed = 0

def _metrics() -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    if psutil:
        try:
            m["cpu_util"] = psutil.cpu_percent(interval=0.0) / 100.0
        except Exception:
            pass
    with _metrics_lock:
        m["tasks_completed"] = _tasks_completed
        m["tasks_failed"] = _tasks_failed
    return m

def _inc(ok: bool) -> None:
    global _tasks_completed, _tasks_failed
    with _metrics_lock:
        if ok:
            _tasks_completed += 1
        else:
            _tasks_failed += 1

def system_allows_work() -> bool:
    if not psutil:
        return True
    try:
        if psutil.cpu_percent(interval=0.1) >= BUSY_CPU_THRESHOLD:
            return False
    except Exception:
        pass
    if DISABLE_ON_BATTERY:
        try:
            batt = psutil.sensors_battery()
            if batt and not batt.power_plugged:
                return False
        except Exception:
            pass
    return True

def _req(method: str, path: str, *, json_payload=None, params=None) -> Optional[Dict[str, Any]]:
    try:
        r = _session.request(method, f"{CONTROLLER_URL}{path}", json=json_payload, params=params, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json() if r.content else None
    except Exception as e:
        log(f"http_fail {method} {path} {type(e).__name__}:{e}")
        return None

def register_agent() -> None:
    payload = {
        "agent": AGENT_NAME,
        "labels": LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
        "metrics": _metrics(),
    }
    out = _req("POST", "/api/agents/register", json_payload=payload)
    if out is None:
        log("register_failed /api/agents/register")
    else:
        log("registered /api/agents/register")

def heartbeat_once() -> None:
    payload = {
        "agent": AGENT_NAME,
        "labels": LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
        "metrics": _metrics(),
    }
    out = _req("POST", "/api/agents/heartbeat", json_payload=payload)
    if out is None:
        log("heartbeat_failed /api/agents/heartbeat")

def get_task() -> Optional[Dict[str, Any]]:
    return _req("GET", "/api/task", params={"agent": AGENT_NAME})

def post_result(task_id: Any, ok: bool, result: Optional[Dict[str, Any]], error: Optional[str], duration_ms: float) -> None:
    payload = {
        "id": task_id,
        "agent": AGENT_NAME,
        "ok": ok,
        "result": result if ok else None,
        "error": error if not ok else None,
        "duration_ms": duration_ms,
    }
    out = _req("POST", "/api/result", json_payload=payload)
    if out is None:
        log(f"result_post_failed id={task_id}")

_running = True

def _stop(*_a, **_k):
    global _running
    _running = False
    log("stop_signal")

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

def worker_loop() -> None:
    global _running
    while _running:
        if not system_allows_work():
            time.sleep(0.5)
            continue

        task = get_task()
        if not task:
            time.sleep(POLL_IDLE_SEC)
            continue

        task_id = task.get("id")
        op = task.get("op")
        payload = task.get("payload", {}) or {}

        if op not in OPS:
            post_result(task_id, False, None, f"unknown_op:{op}", 0.0)
            _inc(False)
            continue

        try:
            start = time.time()
            res = OPS[op](payload)
            duration_ms = (time.time() - start) * 1000.0
            ok = bool(res.get("ok", True))

            if ok:
                post_result(task_id, True, res, None, duration_ms)
            else:
                post_result(task_id, False, None, str(res.get("error", "op_failed")), duration_ms)

            _inc(ok)
        except Exception as e:
            post_result(task_id, False, None, f"exception:{type(e).__name__}:{e}", 0.0)
            _inc(False)

def main() -> None:
    log(f"starting name={AGENT_NAME} controller={CONTROLLER_URL}")
    register_agent()

    def hb_loop():
        while _running:
            heartbeat_once()
            time.sleep(HEARTBEAT_SEC)

    threading.Thread(target=hb_loop, daemon=True).start()
    worker_loop()
    log("exiting")

if __name__ == "__main__":
    main()
