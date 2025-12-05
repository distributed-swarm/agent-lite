import os
import time
import socket
import signal
import threading
from typing import Optional, List, Dict, Any

import requests

# Optional metrics
try:
    import psutil
except ImportError:
    psutil = None

from worker_sizing import build_worker_profile

# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True

# Lite agent CPU behavior
BUSY_CPU_THRESHOLD = float(os.getenv("LITE_BUSY_CPU_THRESHOLD", "30.0"))
CPU_CHECK_INTERVAL = float(os.getenv("LITE_CPU_CHECK_INTERVAL", "2.0"))
DISABLE_ON_BATTERY = os.getenv("LITE_DISABLE_ON_BATTERY", "1") not in ("0", "false", "False")

# ---------------- Local metrics tracking ----------------

_metrics_lock = threading.Lock()
_tasks_completed = 0
_tasks_failed = 0
_task_durations: List[float] = []
_max_duration_samples = 100  # Rolling window size

# ---------------- worker profile / labels ----------------

WORKER_PROFILE = build_worker_profile()

BASE_LABELS: Dict[str, Any] = {}

# Parse AGENT_LABELS="key=value,key2=value2"
if AGENT_LABELS_RAW.strip():
    for item in AGENT_LABELS_RAW.split(","):
        if not item.strip():
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            BASE_LABELS[k.strip()] = v.strip()
        else:
            BASE_LABELS[item.strip()] = True

# Always include worker_profile in labels for the controller
BASE_LABELS["worker_profile"] = WORKER_PROFILE

CAPABILITIES: Dict[str, Any] = {
    # Lite agent: cheap text ops only + routing
    "ops": ["map_tokenize", "map_classify", "map_route"],
}

# ---------------- ops: tokenize + lightweight classify + route ----------------


def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple "tokenization" op.

    Payload:
      { "text": "some text" }

    Result:
      { "ok": true, "tokens": [...], "length": int }
    """
    text = str(payload.get("text", ""))
    tokens = text.split()
    return {
        "ok": True,
        "tokens": tokens,
        "length": len(tokens),
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
    Lightweight rules-based sentiment classifier for agent-lite.

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
        # tie
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
    Routing helper op for agent-lite.

    Uses local limits from WORKER_PROFILE to decide how this
    payload *should* be handled:

      - "local"  : safe for ultra-lite agent
      - "heavy"  : should be pushed to server/GPU tier
      - "discard": empty or obviously useless payload

    This doesn't actually move work; it returns *hints* the controller
    can use to schedule the real follow-up task.
    """
    limits = WORKER_PROFILE.get("limits", {}) or {}
    max_bytes = limits.get("max_payload_bytes")
    max_tokens = limits.get("max_tokens")

    text = str(payload.get("text", "") or "")
    explicit_bytes = payload.get("payload_bytes")
    explicit_tokens = payload.get("tokens")

    if not text and explicit_bytes is None and explicit_tokens is None:
        return {
            "ok": True,
            "route": "discard",
            "reason": "empty_payload",
        }

    # Approximate size
    approx_bytes = explicit_bytes
    if approx_bytes is None:
        try:
            approx_bytes = len(text.encode("utf-8"))
        except Exception:
            approx_bytes = len(text)

    approx_tokens = explicit_tokens
    if approx_tokens is None:
        approx_tokens = len(text.split()) if text else 0

    exceeds_bytes = bool(max_bytes is not None and approx_bytes is not None and approx_bytes > max_bytes)
    exceeds_tokens = bool(max_tokens is not None and approx_tokens is not None and approx_tokens > max_tokens)

    if exceeds_bytes or exceeds_tokens:
        route = "heavy"
        suggested_tier = "server-or-gpu"
        reason = "exceeds_ultra_lite_limits"
    else:
        route = "local"
        suggested_tier = WORKER_PROFILE.get("tier", "ultra-lite")
        reason = "within_ultra_lite_limits"

    return {
        "ok": True,
        "route": route,
        "suggested_tier": suggested_tier,
        "reason": reason,
        "estimated_payload_bytes": approx_bytes,
        "estimated_tokens": approx_tokens,
        "limits": {
            "max_payload_bytes": max_bytes,
            "max_tokens": max_tokens,
        },
    }


OPS = {
    "map_tokenize": op_map_tokenize,
    "map_classify": op_map_classify,
    "map_route": op_map_route,
}

# ---------------- metrics helpers ----------------


def _record_task_result(duration_ms: float, ok: bool) -> None:
    """
    Record the result of a task execution for local metrics tracking.
    Thread-safe.
    """
    global _tasks_completed, _tasks_failed, _task_durations

    with _metrics_lock:
        if ok:
            _tasks_completed += 1
        else:
            _tasks_failed += 1

        _task_durations.append(duration_ms)
        # Keep only the last N samples for rolling average
        if len(_task_durations) > _max_duration_samples:
            _task_durations.pop(0)


def _collect_metrics() -> Dict[str, Any]:
    """
    Collect lightweight agent metrics for autonomic decisions.
    Includes both system metrics (CPU, RAM) and task performance metrics.
    """
    metrics: Dict[str, Any] = {}

    # System metrics via psutil
    if psutil is not None:
        try:
            metrics["cpu_util"] = psutil.cpu_percent(interval=0.0) / 100.0
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()
            metrics["ram_mb"] = int(vm.used / (1024 * 1024))
        except Exception:
            pass

        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                metrics["on_battery"] = not battery.power_plugged
                metrics["battery_percent"] = battery.percent
        except Exception:
            pass

    # Task performance metrics
    with _metrics_lock:
        metrics["tasks_completed"] = _tasks_completed
        metrics["tasks_failed"] = _tasks_failed

        if _task_durations:
            avg_ms = sum(_task_durations) / len(_task_durations)
            metrics["avg_task_ms"] = avg_ms

    return metrics


# ---------------- system load guard ----------------


def system_allows_work() -> bool:
    """
    Decide whether the agent should lease/execute work, based on
    system CPU load and battery status.
    """
    if psutil is None:
        # No visibility; assume it's fine.
        return True

    try:
        cpu = psutil.cpu_percent(interval=0.3)
    except Exception:
        cpu = 0.0

    if cpu >= BUSY_CPU_THRESHOLD:
        # System is already busy; don't add more work.
        return False

    # Battery guardrail (mainly laptops)
    if DISABLE_ON_BATTERY:
        try:
            battery = psutil.sensors_battery()
        except Exception:
            battery = None

        if battery is not None and not battery.power_plugged:
            # On battery and policy says "do not run".
            return False

    return True


# ---------------- HTTP helpers ----------------


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


# ---------------- register / heartbeat ----------------


def register_agent() -> None:
    payload: Dict[str, Any] = {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
    }
    # send metrics as a dedicated sub-dict
    payload["metrics"] = _collect_metrics()
    print(f"[agent] registering with controller as {AGENT_NAME}")
    _post_json("/agents/register", payload)


def heartbeat_loop() -> None:
    while _running:
        payload: Dict[str, Any] = {
            "agent": AGENT_NAME,
            "labels": BASE_LABELS,
            "capabilities": CAPABILITIES,
            "worker_profile": WORKER_PROFILE,
        }
        # send metrics as a dedicated sub-dict
        payload["metrics"] = _collect_metrics()
        _post_json("/agents/heartbeat", payload)
        time.sleep(HEARTBEAT_SEC)


# ---------------- task execution ----------------


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
        # Be polite: don't work if the system is busy or on battery (per policy)
        if not system_allows_work():
            time.sleep(CPU_CHECK_INTERVAL)
            continue

        # Ask for a task
        task = _get_json("/task", {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS})
        if not task:
            # No task right now
            continue

        job_id = task.get("id")
        op = task.get("op")
        payload = task.get("payload") or {}

        start_ts = time.time()
        result_data = _execute_op(op, payload)
        duration_ms = (time.time() - start_ts) * 1000.0

        ok = bool(result_data.get("ok", True))
        error_str = result_data.get("error")

        # Record metrics locally
        _record_task_result(duration_ms, ok)

        result_payload: Dict[str, Any] = {
            "id": job_id,
            "agent": AGENT_NAME,
            "op": op,
            "ok": ok,
            "result": result_data if ok else None,
            "error": error_str if not ok else None,
            "duration_ms": duration_ms,
        }

        _post_json("/result", result_payload)


# ---------------- signal handling ----------------


def _stop(*_args, **_kwargs):
    global _running
    print("[agent] stop signal received, shutting down...")
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# ---------------- main ----------------


def main():
    register_agent()

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    worker_loop()


if __name__ == "__main__":
    main()
