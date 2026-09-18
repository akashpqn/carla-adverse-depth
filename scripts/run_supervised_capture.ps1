# Repeatedly (re)launches CARLA headless and resumes the dataset generator until the target frame
# count is reached, since CARLA sporadically dies with a GPU device-hung/removed crash on this
# machine's 4GB-VRAM laptop GPU (see README). The generator itself is resumable: it counts existing
# metadata/*.json files in -Out and only captures the deficit, using dataset-sequential filenames
# rather than CARLA's own per-session tick counter, so a restart never collides with prior output.
param(
    [string]$Town = "Town01",
    [string]$Weather = "hard_rain_night",
    [int]$Frames = 10000,
    [int]$Width = 960,
    [int]$Height = 540,
    [string]$Out = "$PSScriptRoot\..\dataset\capture",
    [int]$MaxAttempts = 500,
    # 40 traffic vehicles crashed the simulator during warm-up on a 4 GB GPU with the four-camera
    # rig; 25 has been stable.
    [int]$Traffic = 25,
    [int]$Walkers = 10,
    # Point these at your own install and Python environment, or pass them in.
    [string]$CarlaRoot = "C:\CARLA_0.9.12\WindowsNoEditor",
    [string]$Python = "$env:USERPROFILE\miniconda3\envs\carla\python.exe"
)

$ErrorActionPreference = "Stop"
# Keep the CARLA install on an internal drive: map streaming during load_world() fails on drives
# that stall under load, which kills the server rather than simply being slow.
$carlaExe = Join-Path $CarlaRoot "CarlaUE4.exe"
$carlaDir = $CarlaRoot
$py = $Python
$script = Join-Path $PSScriptRoot "generate_adver_city_depth_dataset.py"
# Per-run log folder, so many runs (one per town/weather pair) don't overwrite each other's
# capture_attemptN logs.
$logDir = Join-Path $Out "_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-CapturedCount {
    $metaDir = Join-Path $Out "metadata"
    if (-not (Test-Path $metaDir)) { return 0 }
    return (Get-ChildItem -Path $metaDir -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Stop-AllCarla {
    Get-Process | Where-Object { $_.ProcessName -match "Carla" } | ForEach-Object {
        Stop-Process -Id $_.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    }
    # D: is an external USB drive that measurably struggles (disk I/O retry events in Event
    # Viewer) when CARLA restarts back-to-back with only a couple seconds between attempts -
    # each restart re-reads a lot of map assets from it. This cooldown gives it a breather.
    Start-Sleep -Seconds 8
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $have = Get-CapturedCount
    if ($have -ge $Frames) {
        Write-Host "[supervisor] Target reached: $have/$Frames frames. Done."
        exit 0
    }
    Write-Host "[supervisor] Attempt $attempt/$MaxAttempts - $have/$Frames frames so far."

    Stop-AllCarla
    Write-Host "[supervisor] Launching CARLA headless..."
    Start-Process -FilePath $carlaExe -ArgumentList "-d3d11 -quality-level=Epic -RenderOffscreen -ResX=800 -ResY=600" `
        -WorkingDirectory $carlaDir | Out-Null

    $portOpen = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $conn = New-Object System.Net.Sockets.TcpClient
            $conn.Connect("127.0.0.1", 2000)
            $conn.Close()
            $portOpen = $true
            break
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $portOpen) {
        Write-Host "[supervisor] CARLA never opened port 2000, retrying..."
        continue
    }
    Write-Host "[supervisor] CARLA is up, running generator..."

    $stdout = Join-Path $logDir "capture_attempt$attempt.stdout.log"
    $stderr = Join-Path $logDir "capture_attempt$attempt.stderr.log"
    $argList = @(
        "-u",  # unbuffered: otherwise Python's stdout buffering hides all progress/errors until exit
        "`"$script`"",
        "--town", $Town,
        "--weather", $Weather,
        "--frames", $Frames,
        "--width", $Width, "--height", $Height,
        "--traffic", $Traffic, "--walkers", $Walkers,
        "--out", "`"$Out`""
    ) -join " "

    $proc = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
    $null = $proc.Handle  # without touching Handle, ExitCode stays empty for a non -Wait process
    # Watchdog: when CARLA crashes, the generator otherwise sits out its full RPC timeout (~2 min)
    # before exiting. Kill it as soon as the CARLA server process is gone so the restart is immediate.
    # A crashed CARLA often lingers behind its error dialog, so check the RPC port, not the process.
    # Require 3 consecutive failed checks (15s) so a momentary hiccup doesn't kill a healthy run.
    $misses = 0
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 5
        $alive = $false
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $c.Connect("127.0.0.1", 2000)
            $c.Close()
            $alive = $true
        } catch {}
        if ($alive) { $misses = 0 } else { $misses++ }
        if ($misses -ge 3) {
            Write-Host "[supervisor] CARLA stopped answering - stopping the generator now."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
    }
    $proc.WaitForExit()
    Write-Host "[supervisor] Generator exited with code $($proc.ExitCode)."
}

$have = Get-CapturedCount
Write-Host "[supervisor] Stopping after $MaxAttempts attempts: $have/$Frames frames captured."
if ($have -ge $Frames) { exit 0 } else { exit 1 }
