# Installs the prompt-refiner subagent + UserPromptSubmit hook into this project's .claude folder.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File .\claude_setup\install.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$root\.claude\agents", "$root\.claude\hooks" | Out-Null
Copy-Item "$PSScriptRoot\agents\prompt-refiner.md" "$root\.claude\agents\prompt-refiner.md" -Force
Copy-Item "$PSScriptRoot\hooks\refine_prompt.py" "$root\.claude\hooks\refine_prompt.py" -Force
Copy-Item "$PSScriptRoot\settings.json" "$root\.claude\settings.json" -Force
Write-Host "Installed: .claude\agents\prompt-refiner.md, .claude\hooks\refine_prompt.py, .claude\settings.json (hook + existing deny rules)"
