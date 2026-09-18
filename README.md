# CARLA Adverse-Weather Depth Dataset

Generation code for a multi-town, multi-weather monocular-depth dataset recorded in
[CARLA 0.9.12](https://carla.org/). Each frame provides four camera views with aligned semantic
labels, sky masks, LiDAR-projected sparse and semi-dense depth, and per-frame metadata.

The sensor rig and the weather parameters follow the methodology of **Adver-City**
(Karvat & Givigi, 2024) — see [Citation](#citation). This repository is an independent
reimplementation for single-ego depth capture; it is not affiliated with the Adver-City authors and
does not use their OpenCDA multi-agent pipeline.

## What each frame contains

Per frame, for each of the four cameras (`front`, `right`, `left`, `back`):

| Folder | Contents |
| --- | --- |
| `rgb/` | RGB image, `<frame>_<camera>.png` |
| `semantic_label/` | CARLA semantic class ids (raw, not colourised) |
| `sky_mask/` | 255 where the class is sky, else 0 |
| `sparse_depth_u16/` | LiDAR projected into the camera, 16-bit PNG, **centimetres** (`value / 100 = metres`) |
| `semi_dense_depth_u16/` | As above, accumulated over the last 5 LiDAR sweeps, dynamic objects removed |
| `confidence/` | 255 = this-sweep LiDAR hit, 128 = accumulated hit, 0 = invalid or sky |

Shared per frame:

| Folder | Contents |
| --- | --- |
| `lidar/` | Full sweep in world coordinates, `.npy` of `(x, y, z, intensity)` |
| `metadata/` | JSON: weather parameters, camera intrinsics and transforms, scene seed |
| `segments.csv` | Continuous-segment index (see [Segments](#segments)) |

Depth is stored in centimetres rather than millimetres so that values up to 655 m fit in 16 bits;
millimetres overflow past 65.5 m, which ordinary street scenes exceed. Pass `--save-depth-npy` to
additionally write float32 metre-valued `.npy` depth (roughly 8 MB per image, so off by default).

## Weather presets

Ten presets, taken from Adver-City's published `WeatherParameters` values:

`clear_day`, `clear_night`, `soft_rain_day`, `hard_rain_day`, `hard_rain_night`,
`foggy_day`, `foggy_night`, `foggy_hard_rain_day`, `foggy_hard_rain_night`, `glare_day`

Three deviations from the published values, each made because the original value did not render:

- **Fog:** `fog_falloff = 0` disables the exponential height-fog effect in CARLA/UE4 entirely. Set to
  CARLA's own default of `1.0`.
- **Glare:** the published `sun_azimuth_angle` is calibrated to a fixed scenario road heading. Here
  the azimuth is computed from the ego's actual heading at runtime, re-aimed periodically, and
  rotated through the four cameras so glare appears in each of them across a set. Sun altitude is
  raised from 8° to 15°, since at 8° the sun sits behind buildings.
- **Soft rain at night** is excluded: at 30 % precipitation it is not visually distinguishable from
  clear night.

Rain and wet-road effects require `-quality-level=Epic`. CARLA's `Low` preset silently disables
precipitation rendering, producing dry roads under "rain" weather.

Beyond CARLA's own weather, rain presets also receive a **lens-droplet overlay** applied to the saved
RGB only: water beads refract a blurred, inverted view of the scene. Depth, semantic and LiDAR
ground truth are left untouched, so inputs are degraded while targets stay clean.

## Towns

| Script | Town | Character |
| --- | --- | --- |
| `towns/Town10HD.ps1` | Town10HD | Dense urban downtown |
| `towns/Town07.ps1` | Town07 | Rural roads |
| `towns/Town04.ps1` | Town04 | Highway loop with an underpass |
| `towns/Town06.ps1` | Town06 | Multi-lane highways, open horizons |

**Town03 and Town05 are deliberately not captured**, so they remain an unseen validation split.
Town04's underpass provides covered-road driving, which otherwise exists only in the held-out Town03.

Town02 is also excluded: with the four-camera rig it crashed the simulator during warm-up on every
attempt, including with zero traffic and zero pedestrians, while other towns ran the same
configuration without issue.

## Usage

Start nothing by hand — the run scripts launch and supervise CARLA themselves.

```powershell
# one town, all weather presets
.\towns\Town10HD.ps1

# every town in sequence
.\scripts\run_multi_town_capture.ps1

# a single town/weather pair
.\scripts\run_supervised_capture.ps1 -Town Town07 -Weather foggy_night -Frames 182 `
    -Out D:\dataset\Town07\foggy_night
```

Requirements: CARLA 0.9.12 (packaged build) and a Python 3.7 environment with the matching `carla`
wheel, plus `numpy` and `Pillow`. Set `DEFAULT_CARLA_ROOT` in
`scripts/generate_adver_city_depth_dataset.py` and `$carlaExe` / `$py` in
`scripts/run_supervised_capture.ps1` to your own paths.

Every run is resumable: it counts existing frames in the output folder and captures only the
shortfall, so re-running after an interruption continues where it stopped.

## Capture behaviour

- **Traffic and pedestrians are placed relative to the ego**, on its own road ahead and on the
  pavements it is about to pass, rather than scattered across the map. Uniform placement leaves most
  frames empty of any road user.
- **A frame is saved only once the ego has travelled `--min-move` metres** (default 1 m). Without
  this, a car waiting at a red light contributes many near-identical frames.
- **If the ego stays stuck for `--max-stall-ticks`**, the run ends and restarts with a new scene.
- **The scene seed varies with town, weather and wall-clock time**, so different weather folders are
  different drives rather than the same route under a different sky.

### Segments

The simulator's state cannot be restored after a crash, so each restart begins a new scene. A folder
is therefore a sequence of continuous segments rather than one unbroken drive. Each frame's metadata
records its scene in `run_seed`, and `scripts/make_segment_index.py` writes a `segments.csv` per
folder listing every segment's id, seed, first and last frame, and length:

```bash
python scripts/make_segment_index.py <dataset_root>
```

For single-frame depth training the boundaries are irrelevant. For temporal models, use
`segments.csv` to select continuous clips.

## Citation

This work follows the dataset methodology introduced by Adver-City. If you use this code, please
cite their paper:

```bibtex
@article{Karvat_2024_AdverCity,
  title = {Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions},
  author = {Karvat, Mateus and Givigi, Sidney},
  journal = {arXiv preprint arXiv:2410.06380},
  note = {Available at https://arxiv.org/abs/2410.06380},
  year = {2024}
}
```

- Paper: <https://arxiv.org/abs/2410.06380>
- Adver-City project: <https://labs.cs.queensu.ca/quarrg/datasets/adver-city/>
- Adver-City code: <https://github.com/QUARRG/Adver-City>

Please also cite the simulator:

```bibtex
@inproceedings{Dosovitskiy17,
  title = {{CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
```
