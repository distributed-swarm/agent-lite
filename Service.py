import os
import sys
import time
import threading
import logging
from pathlib import Path

import win32serviceutil
import win32service
import win32event
import servicemanager

SERVICE_NAME = "AgentLite"

PROGRAM_DATA_DIR = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "AgentLite"
PROGRAM_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROGRAM_DATA_DIR / "service.log"
ENV_FILE = PROGRAM_DATA_DIR / "agent.env"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AgentLiteService")


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ (does not override existing env)."""
    if not path.exists():
        logger.error("Missing env file: %s", str(path))
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        logger.info("Loaded env from: %s", str(path))
    except Exception:
        logger.exception("Failed to load env file: %s", str(path))


class AgentLiteService(win32serviceutil.ServiceFramework):
    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = "Agent-Lite Task Processor"
    _svc_description_ = "Headless lightweight task agent for Neuro-Fabric (admin-managed)"

    def __init__(self, args):
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.is_running = False
        self.agent_thread = None

    def SvcStop(self):
        logger.info("Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.is_running = False
        win32event.SetEvent(self.stop_event)

        if self.agent_thread and self.agent_thread.is_alive():
            logger.info("Waiting for agent thread to terminate...")
            self.agent_thread.join(timeout=15)

        logger.info("Service stopped")

    def SvcDoRun(self):
        logger.info("=" * 60)
        logger.info("Agent-Lite Service Starting")
        logger.info("=" * 60)

        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )

        self.is_running = True
        self.main()

    def main(self):
        script_dir = Path(__file__).parent.resolve()

        try:
            # Load admin-managed config
            load_env_file(ENV_FILE)

            # Make service runtime deterministic
            os.chdir(script_dir)
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            logger.info("Working directory: %s", str(script_dir))
            logger.info("Python: %s", sys.executable)

            import app  # must exist next to Service.py

            logger.info("Controller URL: %s", getattr(app, "CONTROLLER_URL", ""))
            logger.info("Agent Name: %s", getattr(app, "AGENT_NAME", ""))

            crash_count = 0
            backoff_s = 2

            while self.is_running:
                exc_holder = {"exc": None}

                def run_agent():
                    try:
                        logger.info("Starting app.main()")
                        app.main()
                        logger.info("app.main() exited normally")
                    except Exception as e:
                        exc_holder["exc"] = e
                        logger.exception("Agent crashed")

                self.agent_thread = threading.Thread(target=run_agent, daemon=False)
                self.agent_thread.start()

                while self.is_running:
                    rc = win32event.WaitForSingleObject(self.stop_event, 3000)
                    if rc == win32event.WAIT_OBJECT_0:
                        logger.info("Stop event received")
                        break
                    if self.agent_thread and not self.agent_thread.is_alive():
                        break

                if not self.is_running:
                    try:
                        if hasattr(app, "_running"):
                            app._running = False
                    except Exception:
                        logger.exception("Failed to signal agent stop")
                    break

                if self.agent_thread and not self.agent_thread.is_alive():
                    crash_count += 1
                    logger.error("Agent stopped unexpectedly (crash_count=%d, err=%r)", crash_count, exc_holder["exc"])
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 60)
                    continue

            logger.info("Service main loop exiting")

        except Exception:
            logger.exception("Fatal error in service")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AgentLiteService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(AgentLiteService)
