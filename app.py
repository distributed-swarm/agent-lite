#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import socket
import signal
import traceback
from typing import Any, Dict, Optional, List, Tuple

import requests

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://10.11.12.54:8080").rstrip("/")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())

HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "5"))
IDLE_SLEEP_SEC = float(os.getenv("IDLE_SLEEP_SEC", "0.10"))
ERROR_BACKOFF_SEC = float(os.getenv("ERROR_BACKOFF_SEC", "1.0"))

# Lite should usually lease 1 at a time
MAX_TASKS = int(os.getenv("MAX_TASKS", "1"))
LEASE_TIMEOUT_MS = int(os.getenv("LEASE_TIMEOUT_MS", "2000"))

# Lite safety knobs (carried over conceptually from your legacy agent-lite)
BUSY_CPU_THRESHOLD = float(os.getenv("BUSY_CPU_THRESHOLD", "75"))  # percent
DISABLE_ON_BATTERY = os.getenv("DISABLE_ON_BATTERY", "0").strip().lower() in ("1", "true", "yes")

# payload sizing (kept for honesty / labeling)
CHUNK_BYTES = int(os.getenv("CHUNK_BYTES", "2048"))
L2_TARGET_BYTES = int(os.getenv("L2_TARGET_BYTES", str(256 * 1024)))
L3_TARGET_BYTES = int(os.getenv("L3_TARGET_BYTES", str(8 * 1024 * 1024)))
SAFE_MAX_BYTES = int(os.getenv("SAFE_MAX_BYTES", str(2 * 1024 * 1024)))
HARD_MAX_BYTES = int(os.getenv("HARD_MAX_BYTES", str(8 * 1024 * 1024)))

# Comma-separated ops list
TASKS_RAW = os.getenv("TASKS", "echo,map_tokenize,fibonacci,prime_factor,read_csv_shard")

# Optional labels: "k=v,k2=v2"
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True


# ---------------- helpers ----------------

def _parse_labels(raw: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[part] = True
    return out


def _capabilities_list() -> List[str]:
    ops = [x.strip() for x in (TASKS_RAW or "").split(",") if x.strip()]
    seen = set()
    out: List[str] = []
    for o in ops:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def _collect_metrics() -> Dict[str, Any]:
    if psutil is None:
        return {}
    m: Dict[str, Any] = {}
    try:
        m["cpu_util"] = float(psutil.cpu_percent(interval=0.0)) / 100.0
    except Exception:
        pass
    try:
        m["ram_mb"] = float(psutil.virtual_memory().used) / (1024 * 1024)
    except Exception:
        pass
    return m


def system_allows_work() -> bool:
    if psutil is None:
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


# ---------------- worker profile ----------------

WORKER_PROFILE: Dict[str, Any] = {
    "tier": "lite",
    "limits": {
        "max_payload_bytes": SAFE_MAX_BYTES,
        "hard_max_payload_bytes": HARD_MAX_BYTES,
        "l2_target_bytes": L2_TARGET_BYTES,
        "l3_target_bytes": L3_TARGET_BYTES,
    },
    "cpu": {"total_cores": 4, "reserved_cores": 3, "usable_cores": 1, "min_cpu_workers": 1, "max_cpu_workers": 1},
    "gpu": {"gpu_present": False, "gpu_count": 0, "vram_gb": None, "devices": [], "max_gpu_workers": 0},
    "workers": {"max_total_workers": 1, "current_workers": 0},
}


# ---------------- ops ----------------

def op_echo(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "echo": payload}


def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    tokens = text.split()
    encoded = text.encode("utf-8", errors="ignore")
    chunks: List[str] = []
    for i in range(0, len(encoded), CHUNK_BYTES):
        chunks.append(encoded[i:i + CHUNK_BYTES].decode("utf-8", errors="ignore"))
    return {"ok": True, "tokens": tokens, "length": len(tokens), "chunks": chunks}


def op_fibonacci(payload: Dict[str, Any]) -> Dict[str, Any]:
    n = int(payload.get("n", 0))
    if n < 0 or n > 200_000:
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


# Optional: reuse your existing csv_shard op if the module exists
try:
    from ops import csv_shard  # type: ignore

    def op_read_csv_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
        return csv_shard.op_read_csv_shard(payload)

except Exception:
    def op_read_csv_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": False, "error": "read_csv_shard_unavailable_on_lite"}


OPS: Dict[str, Any] = {
    "echo": op_echo,
    "map_tokenize": op_map_tokenize,
    "fibonacci": op_fibonacci,
    "prime_factor": op_prime_factor,
    "read_csv_shard": op_read_csv_shard,
}


# ---------------- v1 client ----------------

_session = requests.Session()


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        r = _session.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
    except Exception as e:
        return 0, {"error": str(e), "url": url}

    if r.status_code == 204:
        return 204, None

    try:
        body = r.json()
    except Exception:
        body = r.text

    return r.status_code, body


def _lease_once(caps: List[str], labels: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    payload: Dict[str, Any] = {
        "agent": AGENT_NAME,
        "capabilities": {"ops": caps},
        "max_tasks": MAX_TASKS,
        "timeout_ms": LEASE_TIMEOUT_MS,
        "labels": labels,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics(),
    }
    code, body = _post_json("/v1/leases", payload)
    if code == 204:
        return None
    if code == 0:
        raise RuntimeError(f"lease failed: {body}")
    if code >= 400:
        raise RuntimeError(f"lease HTTP {code}: {body}")

    if not isinstance(body, dict):
        raise RuntimeError(f"lease body not dict: {body!r}")

    lease_id = body.get("lease_id")
    tasks = body.get("tasks")

    if not isinstance(lease_id, str) or not lease_id:
        raise RuntimeError(f"lease missing lease_id: {body!r}")
    if not isinstance(tasks, list) or not tasks:
        return None

    task = tasks[0]
    if not isinstance(task, dict):
        raise RuntimeError(f"task not dict: {task!r}")

    return lease_id, task


def _post_result(lease_id: str, job_id: str, status: str, result: Any = None, error: Any = None) -> None:
    payload: Dict[str, Any] = {
        "lease_id": lease_id,
        "job_id": job_id,
        "status": status,  # "succeeded" | "failed"
        "result": result,
        "error": error,
    }
    code, body = _post_json("/v1/results", payload)
    if code == 0:
        raise RuntimeError(f"result failed: {body}")
    if code >= 400:
        raise RuntimeError(f"result HTTP {code}: {body}")


def _extract_task(task: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    job_id = task.get("id") or task.get("job_id")
    op = task.get("op")
    payload = task.get("payload") or {}
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"task missing job id: {task!r}")
    if not isinstance(op, str) or not op:
        raise RuntimeError(f"task missing op: {task!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"task payload not dict: {task!r}")
    return job_id, op, payload


# ---------------- runtime ----------------

def _shutdown(signum: int, _frame: Any) -> None:
    global _running
    _running = False
    print(f"[agent-lite-v1] shutdown signal {signum}", flush=True)


def main() -> int:
    global _running
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    caps = _capabilities_list()
    labels = _parse_labels(AGENT_LABELS_RAW)
    labels.setdefault("agent_type", "lite")
    labels.setdefault("version", "v1")

    if not caps:
        print("[agent-lite-v1] no TASKS configured; exiting", flush=True)
        return 2

    print(f"[agent-lite-v1] starting name={AGENT_NAME} controller={CONTROLLER_URL} ops={caps}", flush=True)

    while _running:
        if not system_allows_work():
            time.sleep(0.5)
            continue

        try:
            leased = _lease_once(caps, labels)
        except Exception as e:
            print(f"[agent-lite-v1] lease error: {e}", flush=True)
            time.sleep(ERROR_BACKOFF_SEC)
            continue

        if not leased:
            time.sleep(IDLE_SLEEP_SEC)
            continue

        lease_id, task = leased

        try:
            job_id, op, payload = _extract_task(task)
        except Exception as e:
            print(f"[agent-lite-v1] bad task: {e}", flush=True)
            continue

        start = time.time()
        ok = True
        out: Any = None
        err: Any = None

        try:
            fn = OPS.get(op)
            if fn is None:
                raise RuntimeError(f"unknown_op:{op}")
            # Lite rule: inline execution, keep it predictable
            out = fn(payload)
            if isinstance(out, dict):
                ok = bool(out.get("ok", True))
        except Exception as e:
            ok = False
            err = {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc(limit=10)}

        duration_ms = (time.time() - start) * 1000.0

        try:
            _post_result(
                lease_id,
                job_id,
                "succeeded" if ok else "failed",
                result=(out if ok else None),
                error=(None if ok else {"duration_ms": duration_ms, **(err or {})}),
            )
        except Exception as e:
            print(f"[agent-lite-v1] post result error: {e}", flush=True)

    print("[agent-lite-v1] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
