Bootstrapping Guide

Ensure Docker Desktop is running.

From PowerShell:

```powershell
cd C:\Users\austi\otrep-x-prime
PowerShell -ExecutionPolicy Bypass -File .\scripts\windows_bootstrap.ps1
```

Verify containers:

```powershell
docker ps
```

Start the Web UI:

```powershell
python src\otrep_x_prime\webui\main.py
```

Visit http://localhost:8000
