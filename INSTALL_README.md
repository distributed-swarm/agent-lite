# Agent-Lite (Windows) — Headless Service

Agent-Lite runs as a Windows Service. There is no UI and no user interaction.
Install and management are intended for administrators only.

## Requirements
- Windows 10/11
- Python 3.8+ (3.11 recommended)
- Administrator rights

## Configuration
Agent-Lite reads configuration from:

- `%ProgramData%\AgentLite\agent.env`

Installer behavior:
- If `%ProgramData%\AgentLite\agent.env` does not exist, `install.bat` will copy `agent.env.template` from the install folder.
- If neither exists, install fails (no prompts).

## Install (Admin)
Right-click `install.bat` → **Run as administrator**

## Verify
```bat
sc query AgentLite
