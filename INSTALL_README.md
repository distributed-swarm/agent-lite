# Agent-Lite Installation (Windows)

Agent-Lite is a lightweight CPU-only worker that runs quietly in the background and connects to a Neuro-Fabric controller.

---

## Quick Install

1. Right-click `install.bat`
2. Select **Run as administrator**
3. When prompted, enter your **Controller URL**
   - Example: `http://controller:8080`
4. Installation completes automatically.
5. Agent-Lite starts immediately as a Windows service.

---

## Requirements

- Windows 10 or Windows 11
- Python 3.8+ (Python 3.11 recommended)
- Administrator privileges

---

## Verify Installation

Open **Command Prompt (Admin)** and run:

```bat
sc query AgentLite
