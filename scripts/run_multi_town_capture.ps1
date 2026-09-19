# Runs the full dataset across multiple towns sequentially, splitting each town's frame budget
# evenly across all weather presets (day/night x clear/soft rain/hard rain/fog/fog+rain, plus glare),
# using the crash-resilient per-run supervisor (run_supervised_capture.ps1) for each town/weather
# pair. Output: <OutRoot>\<Town>\<weather>\...
#
# Town03 and Town05 are deliberately excluded so they stay available as an unseen validation split.
# Town04 is included for its underpass/overpass (the closest thing to tunnel driving outside Town03).
# Town01 was captured earlier with the older pipeline and is not re-run here.
#
# Everything is resumable: re-running this skips any town/weather pair that already has its frames.
param(
    [string[]]$Towns = @("Town10HD", "Town07", "Town04", "Town06"),
    # Town02 is left out: with the 4-camera rig it crashed CARLA during warmup on every attempt,
    # even with zero traffic and zero walkers, while Town06 ran the same config cleanly.
    [string[]]$Weathers = @(
        "clear_day", "clear_night",
        # soft_rain_night dropped: at night its 30% precipitation isn't visible as rain.
        "soft_rain_day",
        "hard_rain_day", "hard_rain_night",
        "foggy_day", "foggy_night",
        "foggy_hard_rain_day", "foggy_hard_rain_night",
        "glare_day"
    ),
    [int]$FramesPerWeather = 182,
    [int]$Width = 960,
    [int]$Height = 540,
    [string]$OutRoot = "$PSScriptRoot\..\dataset",
    [int]$MaxAttemptsPerRun = 100,
    [int]$Traffic = 25,
    [int]$Walkers = 10,
    # Town04 is the only highway town in training, so it also records the 50 deg forward
    # camera: long-range depth is what a following-distance task leans on, and the 100 deg
    # surround rig cannot resolve a lead vehicle at that range. Elsewhere four cameras suffice.
    [hashtable]$CamerasPerTown = @{ "Town04" = "front,right,left,back,front_narrow" },
    [string]$DefaultCameras = "front,right,left,back"
)

$total = $Towns.Count * $Weathers.Count
$n = 0
foreach ($town in $Towns) {
    foreach ($weather in $Weathers) {
        $n++
        $out = Join-Path (Join-Path $OutRoot $town) $weather
        $cameras = $DefaultCameras
        if ($CamerasPerTown.ContainsKey($town)) { $cameras = $CamerasPerTown[$town] }
        Write-Host "=== [$n/$total] $town / $weather : target $FramesPerWeather frames -> $out ==="
        & "$PSScriptRoot\run_supervised_capture.ps1" `
            -Town $town -Weather $weather -Frames $FramesPerWeather `
            -Width $Width -Height $Height -Out $out -MaxAttempts $MaxAttemptsPerRun `
            -Traffic $Traffic -Walkers $Walkers `
            -Cameras $cameras
    }
}

Write-Host "=== All towns/weathers complete ==="
