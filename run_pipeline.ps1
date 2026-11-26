# Runs the full pipeline in the project venv and writes one log per step to .\logs\
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1          (full, ~6-10 min on a laptop CPU)
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Fast    (quicker smoke run)
param([switch]$Fast, [switch]$Framed)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "logs" | Out-Null
$extra = @(); if ($Fast) { $extra = @("--fast") }
$steps = @(
    @{ name = "1_data_generator"; args = @("-m", "src.data_generator") },
    @{ name = "2_train";          args = @("-m", "src.train") + $extra },
    @{ name = "3_optimizer";      args = @("-m", "src.optimizer") + $extra },
    @{ name = "4_report";         args = @("-m", "src.report") },
    @{ name = "5_pytest";         args = @("-m", "pytest", "-q") }
)
if ($Framed) {  # FRAMED benchmark on published data (docs/FRAMED_PROJECT.md) instead of the V1 steps
    $steps = @(@{ name = "framed_all"; args = @("-m", "src.framed.run", "--stage", "all") + $extra })
}
$summary = @()
foreach ($s in $steps) {
    Write-Host "=== $($s.name) ===" -ForegroundColor Cyan
    $t0 = Get-Date
    & $py @($s.args) 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "logs\$($s.name).log"
    $code = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $summary += "$($s.name): exit $code ($secs s)"
    if ($code -ne 0) { Write-Host "Step $($s.name) failed with exit code $code" -ForegroundColor Red; break }
}
$summary | Out-File -Encoding utf8 "logs\summary.txt"
Write-Host ""; $summary | ForEach-Object { Write-Host $_ }
