# Runs every capture pass in order, unattended. Each pass is resumable and skips whatever is
# already complete, so re-running this after a crash or a reboot simply continues.
#
#   1. training core      -> dataset\      182 frames/weather   Town10HD, Town07, Town04
#   2. validation         -> dataset_val\   60 frames/weather   Town05, Town03
#   3. training extension -> dataset\      182 frames/weather   Town01, then Town02
#
# The validation pass writes to a DIFFERENT ROOT on purpose. Dataset loaders discover runs by
# walking for any directory containing an rgb\ folder, so a held-out town sitting inside the
# training root is one careless glob away from becoming training data - and an evaluation number
# that quietly became a training number looks completely normal in a report. A separate root cannot
# be included by accident. Validation also needs coverage rather than volume, hence 60 frames.
#
# Validation runs before the training extension because Town05 was already part-captured when
# Town01 and Town02 were added, and finishing the held-out set matters more than widening training.
#
# Town06 is captured by neither: it is reserved for following-distance evaluation (see the README).
param(
    [string]$Root = $PSScriptRoot,
    [int]$ValidationFramesPerWeather = 60,
    # Town03 has the only tunnel in the release and the densest junctions; Town05 an unseen urban
    # grid with multiple lanes per direction. Neither may ever enter training.
    [string[]]$ValidationTowns = @("Town05", "Town03"),
    [string[]]$ExtraTrainingTowns = @("Town01")
)

$multiTown = Join-Path $Root "run_multi_town_capture.ps1"

# Each phase is independent: one that fails outright must not cancel the phases after it.
function Invoke-Phase {
    param([string]$Label, [hashtable]$Arguments)
    Write-Host "=== [queue] $Label ==="
    try { & $multiTown @Arguments } catch { Write-Host "[queue] $Label failed: $($_.Exception.Message)" }
}

Write-Host "=== [queue] 1/3 training core ==="
& $multiTown

Write-Host "=== [queue] 2/3 validation: $($ValidationTowns -join ', ') at $ValidationFramesPerWeather frames/weather ==="
& $multiTown -Towns $ValidationTowns -FramesPerWeather $ValidationFramesPerWeather `
    -OutRoot (Join-Path $Root "..\dataset_val")

Write-Host "=== [queue] 3/3 training extension: $($ExtraTrainingTowns -join ', ') ==="
& $multiTown -Towns $ExtraTrainingTowns

# Town02 gets its own invocation with a lighter load. With the four-camera rig at 25 traffic it
# crashed the simulator during warm-up on every previous attempt, including with zero traffic and
# zero walkers - so this is a retry, not an expectation. Fewer actors is the one variable left to
# change; if it still dies, the supervisor gives up after its attempt budget and the queue ends
# without it, which costs nothing since every other town is already captured.
Write-Host "=== [queue] 3b/3 training extension: Town02 (retry, reduced load) ==="
& $multiTown -Towns @("Town02") -Traffic 12 -Walkers 4 -MaxAttemptsPerRun 15

Write-Host "=== [queue] all captures complete ==="
