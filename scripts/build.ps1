<#
    Betaal Windows build script.

    Produces a one-directory PyInstaller bundle at dist/Betaal/.
    Run from the Betaal/ folder in an activated project venv, or let this
    script create/use .venv itself.

    Usage:
        pwsh -File scripts/build.ps1              # build the app bundle
        pwsh -File scripts/build.ps1 -Installer   # also compile the Inno Setup installer

    Requirements:
        - Python 3.10+ and uv (https://docs.astral.sh/uv/)
        - For -Installer: Inno Setup 6 (iscc.exe on PATH or default install path)

    Models are NOT bundled. They download + optimize into ~/.cache/betaal on
    first run (or via "Betaal.exe setup", which the installer triggers).
#>
[CmdletBinding()]
param(
    [switch]$Installer
)

$ErrorActionPreference = "Stop"

# Resolve the project root (parent of this script's folder).
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "[build] Project root: $ProjectRoot"

# 1. Create / sync the virtual environment with runtime deps + PyInstaller.
if (-not (Test-Path ".venv")) {
    Write-Host "[build] Creating virtual environment (.venv)..."
    uv venv
}

Write-Host "[build] Installing runtime + build dependencies (incl. PyInstaller)..."
# --extra build pulls PyInstaller as a managed dependency so subsequent syncs
# don't uninstall it (which previously corrupted altgraph mid-build).
uv sync --extra build

# 2. Clean previous build output.
foreach ($dir in @("build", "dist")) {
    if (Test-Path $dir) {
        Write-Host "[build] Removing stale $dir/"
        Remove-Item -Recurse -Force $dir
    }
}

# 3. Build the one-directory bundle.
Write-Host "[build] Running PyInstaller..."
uv run pyinstaller betaal.spec --noconfirm

$exePath = Join-Path $ProjectRoot "dist\Betaal\Betaal.exe"
if (-not (Test-Path $exePath)) {
    throw "[build] Build failed: $exePath not found."
}
Write-Host "[build] Bundle ready: $exePath"

# 4. Optionally compile the installer.
if ($Installer) {
    $iss = Join-Path $ProjectRoot "installer\betaal.iss"
    $iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($iscc) {
        $iscc = $iscc.Source
    } else {
        # Inno Setup 6 install locations: machine-wide (x86/x64) and per-user (winget).
        $candidates = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        )
        $iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $iscc) {
            throw "[build] Inno Setup (ISCC.exe) not found. Install Inno Setup 6 to build the installer."
        }
    }
    Write-Host "[build] Compiling installer with $iscc"
    & $iscc $iss
    Write-Host "[build] Installer written to installer\Output\"
}

Write-Host "[build] Done."
