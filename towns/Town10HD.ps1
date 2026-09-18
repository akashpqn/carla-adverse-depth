# Town10HD - dense urban downtown: high-rise canyons, wide avenues, heavy signage.
# Captures all weather presets for this town only.
param(
    [int]$FramesPerWeather = 182,
    [int]$Width = 960,
    [int]$Height = 540,
    [string]$OutRoot = "D:\adver_city_depth_dataset\dataset",
    [int]$Traffic = 25,
    [int]$Walkers = 10
)

& "$PSScriptRoot\..\scripts\run_multi_town_capture.ps1" `
    -Towns @("Town10HD") -FramesPerWeather $FramesPerWeather `
    -Width $Width -Height $Height -OutRoot $OutRoot -Traffic $Traffic -Walkers $Walkers
