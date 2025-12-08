import os
import time
import socket
import signal
import threading
import json
import platform
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

try:
    import psutil
except ImportError:
    psutil = None

# =========================
#   INTERNAL CONFIG GENERATOR
# =========================

def generate_agent_config():
    """
    Auto-detects hardware and generates the 'Stealth' configuration.
    Merged directly into app.py for portability.
    """
    hostname = socket.gethostname()
    
    # Hardware Detection
    try:
        cpu_count = psutil.cpu_count(logical=False) or 1
        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except:
        cpu_count = 1
        total_ram_gb = 4.0

    return {
        "agent_id": f"agent-lite-{hostname}",
        "agent_type": "desktop-lite",
        "version": "1.0.0",
        
        "capabilities": {
            # CRITICAL: Added 'echo' here so the Controller sends speed tests
            "ops": ["echo", "map_tokenize", "map_classify", "map_route"],
            "features": ["idle_harvesting", "auto_update", "health_reporting"]
        },
        
        "worker_profile": {
            "tier": "ultra-lite",
            "cpu": {
                "physical_cores": cpu_count,
                "usable_cores": 1, # Stealth: Only use 1 core
            },
            # FORCE CPU ONLY
            "gpu": { 
                "gpu_present": False, 
                "vram_gb": 0 
            },
            "memory": {
                "total_gb": total_ram_gb,
                "max_job_memory_mb": 256
            },
            "limits": {
                "max_payload_bytes": 1024 * 1024, # 1MB limit for lite agents
                "max_tokens": 2048,
                "max_concurrent_jobs": 1
            }
        },
        
        "harvesting_config": {
            "enabled": True,
            "cpu_threshold_percent": 30.0, # Work only if CPU < 30%
            "check_interval_seconds": 2.0,
            "respect_battery": True
        }
    }

# =========================
#   GLOBAL SETUP
# =========================

# 1. Generate Base Config
CONFIG = generate_agent_config()
WORKER_PROFILE = CONFIG["worker_profile"]
HARVEST_CONFIG = CONFIG["harvesting_config"]

# 2. Allow Environment Overrides (Docker/Power Users)
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:8080")
AGENT_NAME = os.getenv("AGENT_NAME", CONFIG["agent_id"])

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000")) # Matches Controller Async Wait
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True

# Harvesting Settings (Env overrides Config)
BUSY_CPU_THRESHOLD = float(os.getenv("CPU_BUSY_CPU_THRESHOLD", str(HARVEST_CONFIG["cpu_threshold_percent"])))
CPU_CHECK_INTERVAL = float(os.getenv("CPU_CPU_CHECK_INTERVAL", str(HARVEST_CONFIG["check_interval_seconds"])))
DISABLE_ON_BATTERY = os.getenv("CPU_DISABLE_ON_BATTERY", "1" if HARVEST_CONFIG["respect_battery"] else "0") not in ("0", "false", "False")

# Payload Limits
SAFE_MAX_BYTES = int(WORKER_PROFILE["limits"]["max_payload_bytes"])
HARD_MAX_BYTES = int(os.getenv("CPU_HARD_MAX_PAYLOAD_BYTES", str(120 * 1024 * 1024))) # 120MB Hard Cap

# Cache Targets for Routing Logic
L2_TARGET_BYTES = int(os.getenv("CPU_L2_TARGET_BYTES", str(256 * 1024)))      
L3_TARGET_BYTES = int(os.getenv("CPU_L3_TARGET_BYTES", str(8 * 1024 * 1024))) 
CHUNK_BYTES = int(os.getenv("CPU_TOKEN_CHUNK_BYTES", "1024"))

# Capabilities
CAPABILITIES = CONFIG["capabilities"]

# Labels
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
    """Required for Speed Tests"""
    return {"ok": True, "echo": payload}

def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    tokens = text.split()
    encoded = text.encode("utf-8", errors="ignore")
    chunks: List[str] = []
    for i in range(0, len(encoded), CHUNK_BYTES):
        chunk_bytes = encoded[i : i + CHUNK_BYTES]
        chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))
    return {
        "ok": True, "tokens": tokens, "length": len(tokens),
        "chunks": chunks, "chunk_bytes": CHUNK_BYTES,
        "total_bytes": len(encoded), "num_chunks": len(chunks)
    }

_POSITIVE_WORDS = {"good", "great", "excellent", "awesome", "love", "like"}
_NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "dislike"}

def op_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text: return {"ok": True, "label": "NEUTRAL", "score": 0.0}
    tokens = [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]
    pos_hits = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_hits = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    
    if pos_hits > neg_hits: label = "POSITIVE"
    elif neg_hits > pos_hits: label = "NEGATIVE"
    else: label = "NEUTRAL"
    
    return {"ok": True, "label": label, "pos_hits": pos_hits, "neg_hits": neg_hits}

def op_map_route(payload: Dict[str, Any]) -> Dict[str, Any]:
    limits = WORKER_PROFILE.get("limits", {})
    max_bytes = limits.get("max_payload_bytes", SAFE_MAX_BYTES)
    
    # Calculate Size
    text = str(payload.get("text", "") or "")
    approx_bytes = payload.get("payload_bytes") or len(text.encode("utf-8"))

    # Size Class
    if approx_bytes <= L2_TARGET_BYTES: size_class = "l2"
    elif approx_bytes <= L3_TARGET_BYTES: size_class = "l3"
    else: size_class = "ram"

    # Routing Logic
    if approx_bytes > HARD_MAX_BYTES:
        return {"ok": True, "route": "heavy", "reason": "exceeds_hard_limit"}
    elif approx_bytes > max_bytes and size_class == "ram":
         return {"ok": True, "route": "heavy", "reason": "exceeds_soft_cpu_limit"}
    
    return {"ok": True, "route": f"local_{size_class}", "reason": "within_cpu_limits"}

OPS = {
    "echo": op_echo,
    "map_tokenize": op_map_tokenize,
    "map_classify": op_map_classify,
    "map_route": op_map_route,
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
        if ok: _tasks_completed += 1
        else: _tasks_failed += 1
        _task_durations.append(duration_ms)
        if len(_task_durations) > 100: _task_durations.pop(0)

def _collect_metrics() -> Dict[str, Any]:
    metrics = {}
    if psutil:
        metrics["cpu_util"] = psutil.cpu_percent(interval=0.0) / 100.0
        try: metrics["ram_mb"] = int(psutil.virtual_memory().used / 1048576)
        except: pass
    with _metrics_lock:
        metrics["tasks_completed"] = _tasks_completed
        metrics["tasks_failed"] = _tasks_failed
    return metrics

def system_allows_work() -> bool:
    """Stealth Check: Returns False if user is busy or on battery"""
    if not psutil: return True
    
    # CPU Guard
    if psutil.cpu_percent(interval=0.1) >= BUSY_CPU_THRESHOLD: 
        return False
        
    # Battery Guard
    if DISABLE_ON_BATTERY:
        try:
            batt = psutil.sensors_battery()
            if batt and not batt.power_plugged: return False
        except: pass
        
    return True

# =========================
#   NETWORKING
# =========================

def _post_json(path, payload):
    try:
        r = requests.post(f"{CONTROLLER_URL}{path}", json=payload, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json() if r.content else None
    except Exception as e:
        # print(f"[agent] POST {path} error: {e}")
        return None

def _get_json(path, params):
    try:
        r = requests.get(f"{CONTROLLER_URL}{path}", params=params, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 204: return None
        r.raise_for_status()
        return r.json() if r.content else None
    except Exception as e:
        return None

def register_agent():
    print(f"[*] Registering {AGENT_NAME}...")
    _post_json("/agents/register", {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": CAPABILITIES,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics()
    })

def worker_loop():
    print(f"[*] Worker Loop Started -> {CONTROLLER_URL}")
    while _running:
        if not system_allows_work():
            time.sleep(CPU_CHECK_INTERVAL)
            continue

        task = _get_json("/task", {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS})
        if not task: continue

        op = task.get("op")
        payload = task.get("payload") or {}
        
        # Check if we support the op
        if op not in OPS:
            _post_json("/result", {"id": task["id"], "agent": AGENT_NAME, "ok": False, "error": f"Unknown op: {op}"})
            continue

        # Execute
        start = time.time()
        res = OPS[op](payload)
        duration = (time.time() - start) * 1000
        ok = res.get("ok", True)
        
        # Report
        _record_task_result(duration, ok)
        _post_json("/result", {
            "id": task["id"], "agent": AGENT_NAME, "ok": ok, 
            "result": res if ok else None, "error": res.get("error") if not ok else None, 
            "duration_ms": duration
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
    # Heartbeat
    threading.Thread(target=lambda: [
        (_post_json("/agents/heartbeat", {
            "agent": AGENT_NAME, "labels": BASE_LABELS, "capabilities": CAPABILITIES, 
            "worker_profile": WORKER_PROFILE, "metrics": _collect_metrics()
        }), time.sleep(HEARTBEAT_SEC)) 
        for _ in iter(int, 1) if _running
    ], daemon=True).start()
    
    worker_loop()

if __name__ == "__main__":
    main()
