import os
import time
import socket
import signal
import threading
import json
from typing import Optional, List, Dict, Any

import requests

try:
    import psutil
except ImportError:
    psutil = None

from agent_config import generate_agent_config

# =========================
#   CONFIG & CONSTANTS
# =========================

# Pull base config from agent_config.py
CONFIG = generate_agent_config() or {}

AGENT_ID = CONFIG.get("agent_id") or f"agent-cpu-{socket.gethostname()}"
AGENT_TYPE = CONFIG.get("agent_type", "desktop-lite")
AGENT_VERSION = CONFIG.get("version", "1.0.0")

# Agent name can still be overridden by env
AGENT_NAME = os.getenv("AGENT_NAME", AGENT_ID)
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True

# Harvesting defaults from config, then env overrides
_harvest_cfg = CONFIG.get("harvesting_config", {}) or {}
BUSY_CPU_THRESHOLD = float(
    os.getenv(
        "CPU_BUSY_CPU_THRESHOLD",
        str(_harvest_cfg.get("cpu_threshold_percent", 30.0)),
    )
)
CPU_CHECK_INTERVAL = float(
    os.getenv(
        "CPU_CPU_CHECK_INTERVAL",
        str(_harvest_cfg.get("check_interval_seconds", 2.0)),
    )
)
DISABLE_ON_BATTERY = os.getenv(
    "CPU_DISABLE_ON_BATTERY",
    "1" if _harvest_cfg.get("respect_battery", True) else "0",
) not in ("0", "false", "False")

# Worker profile from config
_worker_profile_cfg = CONFIG.get("worker_profile", {}) or {}
_limits_cfg = (_worker_profile_cfg.get("limits") or {}).copy()

# Soft, config-based safe max (config wins, then env)
SAFE_MAX_BYTES_CONFIG = int(_limits_cfg.get("max_payload_bytes", 64 * 1024))
SAFE_MAX_BYTES = int(
    os.getenv("CPU_SAFE_MAX_PAYLOAD_BYTES", str(SAFE_MAX_BYTES_CONFIG))
)

# Hard max for this CPU agent (up to 120 MB by default)
HARD_MAX_BYTES = int(
    os.getenv("CPU_HARD_MAX_PAYLOAD_BYTES", str(120 * 1024 * 1024))
)

MAX_TOKENS_CONFIG = int(_limits_cfg.get("max_tokens", 2048))
MAX_TOKENS = int(os.getenv("CPU_MAX_TOKENS", str(MAX_TOKENS_CONFIG)))

# L2 / L3 cache-ish targets (these are heuristics, not exact cache sizes)
L2_TARGET_BYTES = int(os.getenv("CPU_L2_TARGET_BYTES", str(256 * 1024)))      # 256 KB
L3_TARGET_BYTES = int(os.getenv("CPU_L3_TARGET_BYTES", str(8 * 1024 * 1024))) # 8 MB

# Chunk size for tokenize op (default 1 KB)
CHUNK_BYTES = int(os.getenv("CPU_TOKEN_CHUNK_BYTES", "1024"))

# Enrich worker profile with enforced limits & normalized tier
WORKER_PROFILE: Dict[str, Any] = {
    **_worker_profile_cfg,
    "tier": _worker_profile_cfg.get("tier", "cpu"),
}

updated_limits = dict(_limits_cfg)
updated_limits["max_payload_bytes"] = SAFE_MAX_BYTES
updated_limits["max_tokens"] = MAX_TOKENS
WORKER_PROFILE["limits"] = updated_limits

# Capabilities from config, but we make sure ops list is present
_capabilities_cfg = CONFIG.get("capabilities", {}) or {}
_cap_ops = _capabilities_cfg.get("ops") or ["echo", "map_tokenize", "map_classify", "map_route"]

CAPABILITIES: Dict[str, Any] = {
    "ops": _cap_ops,
    "features": _capabilities_cfg.get("features", []),
    "agent_type": AGENT_TYPE,
    "version": AGENT_VERSION,
}

# Base labels sent to controller
BASE_LABELS: Dict[str, Any] = {
    "agent_type": AGENT_TYPE,
    "agent_version": AGENT_VERSION,
    "worker_profile": WORKER_PROFILE,
}

if AGENT_LABELS_RAW.strip():
    for item in AGENT_LABELS_RAW.split(","):
        if not item.strip():
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            BASE_LABELS[k.strip()] = v.strip()
        else:
            BASE_LABELS[item.strip()] = True

# =========================
#   OPS IMPLEMENTATIONS
# =========================

def op_echo(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple echo op, mainly for testing and sanity checks.
    """
    return {
        "ok": True,
        "echo": payload,
    }


def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple tokenization + ~1 KB chunking helper.

    Payload:
      { "text": "some text" }

    Result:
      {
        "ok": true,
        "tokens": [...],
        "length": int,
        "chunks": [...],         # list of ~CHUNK_BYTES slices
        "chunk_bytes": int,
        "total_bytes": int,
        "num_chunks": int,
      }
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


_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "awesome",
    "love",
    "like",
    "amazing",
    "fantastic",
    "happy",
    "pleased",
    "cool",
}

_NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "dislike",
    "horrible",
    "sad",
    "angry",
    "upset",
    "annoying",
    "disappointed",
}


def op_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight rules-based sentiment classifier for CPU agent.

    Payload:
      { "text": "some text" }

    Result:
      { "ok": true, "label": "POSITIVE"/"NEGATIVE"/"NEUTRAL", "score": float }
    """
    text = str(payload.get("text", "")).strip()
    if not text:
        return {
            "ok": True,
            "label": "NEUTRAL",
            "score": 0.0,
            "detail": "empty text",
        }

    tokens = [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]

    pos_hits = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_hits = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

    if pos_hits == 0 and neg_hits == 0:
        label = "NEUTRAL"
        score = 0.0
    elif pos_hits > neg_hits:
        label = "POSITIVE"
        score = float(pos_hits) / float(pos_hits + neg_hits)
    elif neg_hits > pos_hits:
        label = "NEGATIVE"
        score = float(neg_hits) / float(pos_hits + neg_hits)
    else:
        label = "NEUTRAL"
        score = 0.0

    return {
        "ok": True,
        "label": label,
        "score": score,
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
    }


def op_map_route(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Routing helper op for CPU agent.

    Uses local limits & size classes to decide whether this payload
    should be treated as:

      - "local_l2"  : small, L2-ish
      - "local_l3"  : medium, L3-ish
      - "local_ram" : large but allowed on this CPU agent
      - "heavy"     : better for server/GPU tier

    This returns hints; controller decides the actual follow-up task.
    """
    limits = WORKER_PROFILE.get("limits", {}) or {}
    max_bytes = limits.get("max_payload_bytes", SAFE_MAX_BYTES)
    max_tokens = limits.get("max_tokens", MAX_TOKENS)

    text = str(payload.get("text", "") or "")
    explicit_bytes = payload.get("payload_bytes")
    explicit_tokens = payload.get("tokens")

    if not text and explicit_bytes is None and explicit_tokens is None:
        return {
            "ok": True,
            "route": "discard",
            "reason": "empty_payload",
        }

    # Approx size
    approx_bytes = explicit_bytes
    if approx_bytes is None:
        try:
            approx_bytes = len(text.encode("utf-8"))
        except Exception:
            approx_bytes = len(text)

    approx_tokens = explicit_tokens
    if approx_tokens is None:
        approx_tokens = len(text.split()) if text else 0

    # Size classification
    if approx_bytes <= L2_TARGET_BYTES:
        size_class = "l2"
    elif approx_bytes <= L3_TARGET_BYTES:
        size_class = "l3"
    else:
        size_class = "ram"

    # Decide routing:
    #  - If above HARD_MAX_BYTES: clearly heavy
    #  - If above safe limit but under HARD_MAX, prefer heavy unless we *know* the box is comfy
    exceeds_hard = approx_bytes > HARD_MAX_BYTES
    exceeds_safe = approx_bytes > max_bytes

    if exceeds_hard:
        route = "heavy"
        reason = "exceeds_hard_limit"
    elif exceeds_safe and size_class == "ram":
        route = "heavy"
        reason = "exceeds_soft_cpu_limit"
    else:
        if size_class == "l2":
            route = "local_l2"
        elif size_class == "l3":
            route = "local_l3"
        else:
            route = "local_ram"
        reason = "within_cpu_limits"

    return {
        "ok": True,
        "route": route,
        "size_class": size_class,
        "reason": reason,
        "estimated_payload_bytes": approx_bytes,
        "estimated_tokens": approx_tokens,
        "limits": {
            "safe_max_bytes": max_bytes,
            "hard_max_bytes": HARD_MAX_BYTES,
            "max_tokens": max_tokens,
            "l2_target_bytes": L2_TARGET_BYTES,
            "l3_target_bytes": L3_TARGET_BYTES,
        },
    }


OPS = {
    "echo": op_echo,
    "map_tokenize": op_map_tokenize,
    "map_classify": op_map_classify,
    "map_route": op_map_route,
}

# =========================
#   METRICS
# =========================

_metrics_lock = threading.Lock()
_tasks_completed = 0
_tasks_failed = 0
_task_durations: List[float] = []
_max_duration_samples = 100


def _record_task_result(duration_ms: float, ok: bool) -> None:
    global _tasks_completed, _tasks_failed, _task_durations
    with _metrics_lock:
        if ok:
            _tasks_completed += 1
        else:
            _tasks_failed += 1

        _task_durations.append(duration_ms)
        if len(_task_durations) > _max_duration_samples:
            _task_durations.pop(0)


def _collect_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if psutil is not None:
        try:
            metrics["cpu_util"] = psutil.cpu_percent(interval=0.0) / 100.0
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()
            metrics["ram_mb"] = int(vm.used / (1024 * 1024))
            metrics["ram_available_mb"] = int(vm.available / (1024 * 1024))
        except Exception:
            pass

        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                metrics["on_battery"] = not battery.power_plugged
                metrics["battery_percent"] = battery.percent
        except Exception:
            pass

    with _metrics_lock:
        metrics["tasks_completed"] = _tasks_completed
        metrics["tasks_failed"] = _tasks_failed
        if _task_durations:
            metrics["avg_task_ms"] = sum(_task_durations) / len(_task_durations)

    return metrics

# =========================
#   SYSTEM GUARD
# =========================

def system_allows_work() -> bool:
    if psutil is None:
        return True

    try:
        cpu = psutil.cpu_percent(interval=0.3)
    except Exception:
        cpu = 0.0

    if cpu >= BUSY_CPU_THRESHOLD:
        return False

    if DISABLE_ON_BATTERY:
        try:
            battery = psutil.sensors_battery()
        except Exception:
            battery = None
        if battery is not None and not battery.power_plugged:
            return False

    return True

# =========================
#   HTTP HELPERS
# =========================

def _post_json(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None
    except Exception as e:
        print(f"[agent] POST {url} failed: {e}")
        return None


def _get_json(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None
    except Exception as e:
        print(f"[agent] GET {url} failed: {e}")
        return None

# =========================
#   REGISTER / HEARTBEAT
# =========================

def register_agent() -> None:
    payload: Dict[str, Any] = {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics(),
        "agent_config": {
            "agent_type": AGENT_TYPE,
            "version": AGENT_VERSION,
        },
    }
    print(f"[agent] registering with controller as {AGENT_NAME}")
    _post_json("/agents/register", payload)


def heartbeat_loop() -> None:
    while _running:
        payload: Dict[str, Any] = {
            "agent": AGENT_NAME,
            "labels": BASE_LABELS,
            "capabilities": CAPABILITIES,
            "worker_profile": WORKER_PROFILE,
            "metrics": _collect_metrics(),
        }
        _post_json("/agents/heartbeat", payload)
        time.sleep(HEARTBEAT_SEC)

# =========================
#   PAYLOAD FILTERING
# =========================

def _approx_payload_bytes(payload: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(payload).encode("utf-8"))
    except Exception:
        return len(str(payload))


def _payload_policy(payload: Dict[str, Any]) -> (bool, Dict[str, Any], str, int):
    """
    Decide if payload is allowed, and classify it into size classes.
    Returns: (allowed, error_info, size_class, approx_bytes)
    """
    approx_bytes = _approx_payload_bytes(payload)

    # Too big under any circumstances
    if approx_bytes > HARD_MAX_BYTES:
        return False, {
            "ok": False,
            "error": "payload_exceeds_hard_limit",
            "hard_max_bytes": HARD_MAX_BYTES,
            "actual_bytes": approx_bytes,
        }, "too_large", approx_bytes

    # Size class
    if approx_bytes <= L2_TARGET_BYTES:
        size_class = "l2"
    elif approx_bytes <= L3_TARGET_BYTES:
        size_class = "l3"
    else:
        size_class = "ram"

    # Always OK if under soft safe limit
    if approx_bytes <= SAFE_MAX_BYTES:
        return True, {}, size_class, approx_bytes

    # Between safe and hard: only allow if RAM is clearly comfortable
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            # require at least 3x payload in free memory
            if vm.available < approx_bytes * 3:
                return False, {
                    "ok": False,
                    "error": "insufficient_ram_for_large_payload",
                    "safe_max_bytes": SAFE_MAX_BYTES,
                    "hard_max_bytes": HARD_MAX_BYTES,
                    "actual_bytes": approx_bytes,
                    "available_ram_bytes": vm.available,
                }, size_class, approx_bytes
        except Exception:
            pass

    return True, {}, size_class, approx_bytes

# =========================
#   TASK EXECUTION
# =========================

def _execute_op(op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = OPS.get(op)
    if fn is None:
        return {
            "ok": False,
            "error": f"Unknown op '{op}'",
        }
    try:
        return fn(payload)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Exception in op '{op}': {e}",
        }


def worker_loop() -> None:
    global _running
    print(f"[agent] worker loop starting for {AGENT_NAME}")
    while _running:
        if not system_allows_work():
            time.sleep(CPU_CHECK_INTERVAL)
            continue

        task = _get_json("/task", {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS})
        if not task:
            continue

        job_id = task.get("id")
        op = task.get("op")
        payload = task.get("payload") or {}

        allowed, error_info, size_class, approx_bytes = _payload_policy(payload)
        if not allowed:
            duration_ms = 0.0
            ok = False
            _record_task_result(duration_ms, ok)
            result_payload: Dict[str, Any] = {
                "id": job_id,
                "agent": AGENT_NAME,
                "op": op,
                "ok": ok,
                "result": None,
                "error": error_info,
                "duration_ms": duration_ms,
                "size_class": size_class,
                "approx_payload_bytes": approx_bytes,
            }
            _post_json("/result", result_payload)
            continue

        start_ts = time.time()
        result_data = _execute_op(op, payload)
        duration_ms = (time.time() - start_ts) * 1000.0

        ok = bool(result_data.get("ok", True))
        error_str = result_data.get("error")

        _record_task_result(duration_ms, ok)

        result_payload: Dict[str, Any] = {
            "id": job_id,
            "agent": AGENT_NAME,
            "op": op,
            "ok": ok,
            "result": result_data if ok else None,
            "error": error_str if not ok else None,
            "duration_ms": duration_ms,
            "size_class": size_class,
            "approx_payload_bytes": approx_bytes,
        }
        _post_json("/result", result_payload)

# =========================
#   SIGNALS & MAIN
# =========================

def _stop(*_args, **_kwargs):
    global _running
    print("[agent] stop signal received, shutting down...")
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def main():
    register_agent()
    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    worker_loop()


if __name__ == "__main__":
    main()
