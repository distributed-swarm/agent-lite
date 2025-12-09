"""
Windows Service wrapper for Agent-Lite
Keeps the agent running as a background service with proper lifecycle management
"""

import win32serviceutil
import win32service
import win32event
import servicemanager
import sys
import os
import time
import logging
import threading
from pathlib import Path

# =========================
#   LOGGING SETUP
# =========================

# Log to a file since services run headless
LOG_DIR = Path.home() / "AppData" / "Local" / "AgentLite"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "service.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AgentLiteService')


# =========================
#   SERVICE CLASS
# =========================

class AgentLiteService(win32serviceutil.ServiceFramework):
    """Windows Service for Agent-Lite distributed task processing"""
    
    _svc_name_ = "AgentLite"
    _svc_display_name_ = "Agent-Lite Task Processor"
    _svc_description_ = "Lightweight distributed task agent that harvests idle CPU cycles for Neuro Fabric"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.is_running = False
        self.agent_thread = None
        
    def SvcStop(self):
        """Called when Windows requests service stop"""
        logger.info("Service stop requested by Windows")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
        # Signal the agent to stop
        self.is_running = False
        win32event.SetEvent(self.stop_event)
        
        # Give the agent thread time to shut down gracefully
        if self.agent_thread and self.agent_thread.is_alive():
            logger.info("Waiting for agent thread to terminate...")
            self.agent_thread.join(timeout=10)
            
        logger.info("Service stopped")
    
    def SvcDoRun(self):
        """Called when service is started"""
        logger.info("=" * 60)
        logger.info("Agent-Lite Service Starting")
        logger.info("=" * 60)
        
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.is_running = True
        self.main()
    
    def main(self):
        """Main service loop - runs the agent in a separate thread"""
        try:
            # Import the agent app here (after service starts) to avoid import issues
            logger.info("Importing agent app module...")
            
            # Make sure we can find app.py - assume it's in the same directory as service.py
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            
            import app
            
            logger.info("Agent app imported successfully")
            logger.info(f"Controller URL: {app.CONTROLLER_URL}")
            logger.info(f"Agent Name: {app.AGENT_NAME}")
            logger.info(f"CPU Threshold: {app.BUSY_CPU_THRESHOLD}%")
            logger.info(f"Battery Policy: {'Disabled on battery' if app.DISABLE_ON_BATTERY else 'Enabled on battery'}")
            
            # Start the agent in a separate thread so we can monitor it
            def run_agent():
                try:
                    logger.info("Starting agent worker...")
                    app.main()  # This runs the agent's main loop
                except Exception as e:
                    logger.error(f"Agent crashed: {e}", exc_info=True)
                    self.is_running = False
            
            self.agent_thread = threading.Thread(target=run_agent, daemon=False)
            self.agent_thread.start()
            logger.info("Agent thread started")
            
            # Main service loop - just keep the service alive and monitor health
            while self.is_running:
                # Wait for stop event with 5 second timeout
                rc = win32event.WaitForSingleObject(self.stop_event, 5000)
                
                if rc == win32event.WAIT_OBJECT_0:
                    # Stop event signaled
                    logger.info("Stop event received")
                    break
                
                # Check if agent thread died unexpectedly
                if not self.agent_thread.is_alive():
                    logger.error("Agent thread died unexpectedly!")
                    # Could implement auto-restart here, but for now just exit
                    break
            
            # Cleanup: tell the agent to stop
            logger.info("Shutting down agent...")
            app._running = False  # Signal the agent to stop
            
            # Wait for agent thread to finish
            if self.agent_thread.is_alive():
                self.agent_thread.join(timeout=15)
            
            logger.info("Service main loop exited normally")
            
        except ImportError as e:
            logger.error(f"Failed to import agent app: {e}", exc_info=True)
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Current directory: {os.getcwd()}")
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)


# =========================
#   INSTALLATION HELPERS
# =========================

def install_service():
    """Install the service with proper configuration"""
    print("\n" + "="*60)
    print("Installing Agent-Lite Windows Service")
    print("="*60)
    
    try:
        # Install the service
        win32serviceutil.HandleCommandLine(AgentLiteService, argv=['service.py', 'install'])
        
        print("\n✓ Service installed successfully!")
        print("\nNext steps:")
        print("1. Configure auto-restart on failure:")
        print("   sc failure AgentLite reset= 86400 actions= restart/60000/restart/60000/restart/60000")
        print("\n2. Set service to auto-start:")
        print("   sc config AgentLite start= auto")
        print("\n3. Start the service:")
        print("   net start AgentLite")
        print("\n4. Check service status:")
        print("   sc query AgentLite")
        print("\n5. View logs:")
        print(f"   {LOG_FILE}")
        
    except Exception as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nMake sure you're running as Administrator!")


def configure_recovery():
    """Configure service recovery options via sc.exe"""
    import subprocess
    
    print("\nConfiguring auto-restart on failure...")
    
    try:
        # Reset failure count after 24 hours, restart after 1 minute on each failure
        result = subprocess.run(
            ['sc', 'failure', 'AgentLite', 'reset=', '86400', 
             'actions=', 'restart/60000/restart/60000/restart/60000'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Auto-restart configured")
        else:
            print(f"✗ Failed to configure auto-restart: {result.stderr}")
    except Exception as e:
        print(f"✗ Error configuring recovery: {e}")


def set_auto_start():
    """Set service to start automatically on boot"""
    import subprocess
    
    print("\nConfiguring auto-start on boot...")
    
    try:
        result = subprocess.run(
            ['sc', 'config', 'AgentLite', 'start=', 'auto'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Auto-start configured")
        else:
            print(f"✗ Failed to configure auto-start: {result.stderr}")
    except Exception as e:
        print(f"✗ Error configuring auto-start: {e}")


# =========================
#   COMMAND LINE HANDLER
# =========================

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Started by Windows Service Manager
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AgentLiteService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        command = sys.argv[1].lower()
        
        if command == 'install':
            install_service()
            configure_recovery()
            set_auto_start()
        elif command == 'remove' or command == 'uninstall':
            print("\nRemoving Agent-Lite service...")
            win32serviceutil.HandleCommandLine(AgentLiteService)
        elif command == 'start':
            print("\nStarting Agent-Lite service...")
            win32serviceutil.HandleCommandLine(AgentLiteService)
            print(f"\n✓ Service started. Check logs at: {LOG_FILE}")
        elif command == 'stop':
            print("\nStopping Agent-Lite service...")
            win32serviceutil.HandleCommandLine(AgentLiteService)
        elif command == 'restart':
            print("\nRestarting Agent-Lite service...")
            win32serviceutil.HandleCommandLine(AgentLiteService, argv=['service.py', 'restart'])
        elif command == 'debug':
            # Run in console for debugging
            print("\n" + "="*60)
            print("Running in DEBUG mode (console)")
            print("="*60)
            print("Press Ctrl+C to stop\n")
            
            # Add console handler for debug mode
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            logger.addHandler(console)
            
            service = AgentLiteService(['service.py', 'debug'])
            service.SvcDoRun()
        else:
            print("\nAgent-Lite Service Management")
            print("="*60)
            print("\nUsage:")
            print("  python service.py install   - Install the service")
            print("  python service.py start     - Start the service")
            print("  python service.py stop      - Stop the service")
            print("  python service.py restart   - Restart the service")
            print("  python service.py remove    - Uninstall the service")
            print("  python service.py debug     - Run in console (for testing)")
            print("\nStatus & Logs:")
            print("  sc query AgentLite         - Check service status")
            print(f"  type {LOG_FILE}            - View logs")
