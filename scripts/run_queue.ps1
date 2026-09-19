# Runs the training capture, then the held-out validation capture, in one unattended queue.
#
# The two passes differ in more than their town list, which is why they are separate invocations:
#
#   training   -> dataset\      182 frames per weather, the towns the model may learn from
#   validation -> dataset_val\   60 frames per weather, towns it must never see
#
# The validation pass writes to a DIFFERENT ROOT on purpose. Dataset loaders discover runs by
# walking for any directory containing an rgb\ folder, so a held-out town sitting inside the
# training root is one careless glob away from becoming training data - and an evaluation number
# that quietly became a training number looks completely normal in a report. A separate root cannot
# be included by accident.
#
# Validation needs coverage, not volume: 60 frames across each of the ten weathers samples every
# condition without spending days of capture on a set nothing trains on.
#
# Everything downstream is resumable, so re-running this after a crash or a reboot picks up wherever
# it stopped and skips whatever is already complete.
param(
    [string]$Root = "D:\adver_city_depth_dataset",
    [int]$ValidationFramesPerWeather = 60,
    # Town05, not Town03: Town06 is the held-out set for following-distance work, and Town05 adds an
    # unseen urban grid with multiple lanes per direction. Town03 stays uncaptured and fully unseen.
    [string[]]$ValidationTowns = @("Town05")
)

$multiTown = Join-Path $Root "run_multi_town_capture.ps1"

Write-Host "=== [queue] training capture ==="
& $multiTown

Write-Host "=== [queue] validation capture: $($ValidationTowns -join ', ') at $ValidationFramesPerWeather frames/weather ==="
& $multiTown -Towns $ValidationTowns -FramesPerWeather $ValidationFramesPerWeather `
    -OutRoot (Join-Path $Root "..\dataset_val")

Write-Host "=== [queue] all captures complete ==="
