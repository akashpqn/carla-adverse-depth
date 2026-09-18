# Town04 - highway loop with an underpass/overpass section, mountain backdrop.
# Included for covered-road ("tunnel-like") driving, which otherwise only exists in the held-out
# Town03.
param(
    [int]$FramesPerWeather = 182,
    [int]$Width = 960,
    [int]$Height = 540,
    [string]$OutRoot = "D:\adver_city_depth_dataset\dataset",
    [int]$Traffic = 25,
    [int]$Walkers = 10
)

& "$PSScriptRoot\..\scripts\run_multi_town_capture.ps1" `
    -Towns @("Town04") -FramesPerWeather $FramesPerWeather `
    -Width $Width -Height $Height -OutRoot $OutRoot -Traffic $Traffic -Walkers $Walkers
