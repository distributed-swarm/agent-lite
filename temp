"""
Enhanced agent configuration with monitoring and health capabilities
"""

import socket
import psutil
import platform
from datetime import datetime

def generate_agent_config():
    """Generate enhanced agent configuration with system detection"""
    
    hostname = socket.gethostname()
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    
    config = {
        "agent_id": f"agent-lite-{hostname}",
        "agent_type": "desktop-lite",
        "version": "1.0.0",  # Agent version for update tracking
        
        "capabilities": {
            "ops": ["map_tokenize", "map_classify", "map_route"],
            "features": [
                "idle_harvesting",      # Only works when system is idle
                "auto_update",          # Supports controller-push updates
                "health_reporting",     # Reports health metrics
                "graceful_shutdown"     # Can pause on user activity
            ]
        },
        
        "worker_profile": {
            "tier": "ultra-lite",
            "cpu": {
                "physical_cores": cpu_count,
                "logical_cores": psutil.cpu_count(logical=True),
                "usable_cores": 1,              # Only use 1 core to stay stealthy
                "max_cpu_workers": 1
            },
            "memory": {
                "total_gb": total_ram_gb,
                "reserved_mb": 512,              # Memory reserved for agent
                "max_job_memory_mb": 256         # Max memory per job
            },
            "limits": {
                "max_payload_bytes": 4096,
                "max_tokens": 1024,
                "max_concurrent_jobs": 1,        # One job at a time
                "job_timeout_seconds": 300       # 5 min timeout per job
            }
        },
        
        "system_info": {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": hostname,
            "python_version": platform.python_version()
        },
        
        "harvesting_config": {
            "enabled": True,
            "cpu_threshold_percent": 10,         # Only work when CPU < 10%
            "check_interval_seconds": 5,         # Check CPU every 5 seconds
            "min_idle_duration_seconds": 30,     # Must be idle for 30s before starting
            "pause_on_activity": True,           # Pause immediately on user activity
            "respect_battery": True,             # Don't work on battery power (laptops)
            "quiet_hours": {
                "enabled": False,                # Can enable business hours only
                "start_time": "09:00",
                "end_time": "17:00"
            }
        },
        
        "health": {
            "report_interval_seconds": 60,       # Send health ping every 60s
            "metrics": {
                "uptime_seconds": 0,
                "jobs_completed": 0,
                "jobs_failed": 0,
                "last_job_timestamp": None,
                "avg_job_duration_seconds": 0,
                "current_cpu_percent": 0,
                "current_memory_percent": 0,
                "is_active": False               # Currently processing work
            }
        },
        
        "update": {
            "auto_update_enabled": True,
            "current_version": "1.0.0",
            "update_channel": "stable",          # stable, beta, dev
            "last_update_check": datetime.utcnow().isoformat(),
            "pending_restart": False
        },
        
        "registration": {
            "first_seen": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "controller_url": "http://localhost:8080"  # Set from config
        }
    }
    
    return config


# Example usage
if __name__ == "__main__":
    import json
    config = generate_agent_config()
    print(json.dumps(config, indent=2))
