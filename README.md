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
| `carla_depth_u16/` | Simulator z-buffer depth, same encoding - dense and pixel-sharp (`--save-carla-depth`) |
| `confidence/` | 255 = this-sweep LiDAR hit, 128 = accumulated hit, 0 = invalid or sky |
| `*_depth_view/` | 8-bit colour preview of each depth map above, for looking at rather than training |

Shared per frame:

| Folder | Contents |
| --- | --- |
| `lidar/` | Full sweep in world coordinates, `.npy` of `(x, y, z, intensity)` |
| `metadata/` | JSON: weather parameters, per-camera intrinsics and transforms, scene seed, ego state and lead-vehicle ground truth |
| `segments.csv` | Continuous-segment index (see [Segments](#segments)) |

Depth is stored in centimetres rather than millimetres so that values up to 655 m fit in 16 bits;
millimetres overflow past 65.5 m, which ordinary street scenes exceed. Pass `--save-depth-npy` to
additionally write float32 metre-valued `.npy` depth (roughly 8 MB per image, so off by default).

Each depth source carries its own range limit, and **anything past it is stored as 0** - the same
"no measurement" value as sky and the gaps between scan lines. Out-of-range pixels are never clamped
to the limit, which would write a distance that was never measured and teach a network that every
far surface sits at exactly that number; KITTI, DrivingStereo and nuScenes all leave such pixels
empty and mask them out of the loss.

| Source | Limit | Why |
| --- | --- | --- |
| `sparse_depth_u16/`, `semi_dense_depth_u16/` | `--max-depth`, 200 m | The LiDAR's own configured range, so nothing the sensor reports is discarded |
| `carla_depth_u16/` | `--camera-max-depth`, 655.35 m | A z-buffer is not range-limited; 655.35 m is the ceiling of the 16-bit centimetre encoding, the convention Virtual KITTI 2 uses for synthetic depth |

Cap the range at training time to suit the task rather than losing it here - KITTI and nuScenes
evaluate depth to 80 m, DDAD to 200 m.

The `_u16` maps are the training data, and an ordinary image viewer renders them nearly black: it
stretches the full 0-65535 range while a 200 m scene only reaches 20000, and most of the frame is 0.
That is expected. Each one therefore also gets a colour preview in the matching `_depth_view/` folder
- turbo on a square-root scale, near in blue through to far in red, black for no measurement, every
source sharing one `--depth-view-max` scale (200 m) so the previews stay comparable. Pass
`--no-depth-view` to skip them, or render any depth PNG on demand:

```powershell
python scripts/view_depth.py <depth.png> --max-depth 200
python scripts/view_depth.py <folder> --all          # a whole folder
```

`semantic_label/` is raw class ids (0-22) in every channel, so it looks almost black too - that is
the label map, not a picture. `confidence/` is a mask, so it is black with white LiDAR scan lines.

## Cameras

Adver-City's four-camera rig (`front`, `right`, `left`, `back`) at 100 deg, plus an optional
`front_narrow` at **50 deg** on the same mount as `front`. Select them with `--cameras`; the
default is the four.

The narrow camera exists because field of view, not resolution, decides whether a distant vehicle
is measurable. At 960 px wide the 100 deg rig resolves 9.6 px/deg, so a 1.5 m car at 80 m is 8 px
tall and one pixel of edge error is +-10.6 m of range. At 50 deg the same car is 19 px and one
pixel is +-4.1 m. Supplying K at evaluation time does not recover that difference - the detail was
never sampled. Each camera carries its own intrinsics, and the LiDAR is projected with the K of the
camera it is being projected into.

## Lead-vehicle ground truth

Every frame records what a following-distance task needs, alongside the depth:

```json
"sim_time_s": 412.35,
"ego": {"speed_mps": 8.19, "yaw_rate_deg_s": -0.00},
"lead_vehicle": {
  "actor_id": 149, "type_id": "vehicle.mini.cooper_s",
  "gap_m": 39.32, "centre_distance_m": 43.67, "lateral_offset_m": 0.00,
  "lead_speed_mps": 0.90, "closing_speed_mps": 7.30, "ttc_s": 5.39,
  "in_ego_path": true
}
```

`gap_m` is bumper-to-bumper along the ego's forward axis, not centre-to-centre. The lead is found
by walking the lane graph ahead rather than by a lateral distance test, because on a curve the
vehicle straight ahead is not the one being followed; `in_ego_path` records whether the lane graph
confirmed it, or whether it was matched by the lateral fallback. Each camera also stores
`lead_vehicle_box`, the lead's 2D box in that view, so depth error can be scored on the followed
vehicle rather than averaged over road and sky.

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

## The captured data

The dataset this code produced is published on Kaggle, CC BY 4.0:
**[brsimps/carla-adverse-weather-depth](https://www.kaggle.com/datasets/brsimps/carla-adverse-weather-depth)**
- 10,300 frames, 54.7 GB, one archive per town and weather so a single condition can be downloaded
without the rest.

## What was captured

10,300 frames, every town/weather pair complete at its target. Each frame carries four camera views
(five where noted), three depth ground truths per view, and per-frame ego and lead-vehicle state.

| Split | Town | Character | Cameras | Frames |
| --- | --- | --- | --- | --- |
| training | Town10HD | Dense urban downtown | 4 | 1,820 |
| training | Town07 | Rural roads, narrow lanes | 4 | 1,820 |
| training | Town04 | Highway loop with an underpass | **5** | 1,820 |
| training | Town01 | Basic T-junction town | **5** | 1,820 |
| training | Town02 | Smaller variant of Town01 | **5** | 1,820 |
| **held out** | Town05 | Squared grid, multiple lanes per direction | 4 | 600 |
| **held out** | Town03 | Roundabout, dense junctions, the only tunnel | 4 | 600 |
| *not captured* | Town06 | Multi-lane highways | - | reserved |

Training goes to `dataset/` (9,100 frames at 182 per weather), held-out towns to `dataset_val/`
(1,200 frames at 60 per weather). The five-camera towns add `front_narrow`; see [Cameras](#cameras).

**Town03, Town05 and Town06 never enter training.** Town03 and Town05 are captured into the
separate `dataset_val/` root as held-out validation; Town06 is not captured by these scripts at
all, since it is reserved for following-distance evaluation. Town04's underpass provides
covered-road driving for training, which otherwise exists only in the held-out Town03.

Town06 is held out specifically for following-distance evaluation. Measured from the OpenDRIVE
networks, only two towns carry freeway-class road:

| Town | Road total | >=55 mph | >=65 mph | Max lanes per direction |
| --- | --- | --- | --- | --- |
| Town06 | 8.5 km | 4.1 km | **3.9 km** | **5** |
| Town04 | 10.8 km | 4.7 km | **3.6 km** | 4 |
| Town05 | 9.2 km | 3.4 km | 1.4 km | 3 |
| Town03 | 11.3 km | 3.6 km | 0 | 2 |
| Town10HD, Town07, Town01, Town02 | - | **0** | **0** | 1-2 |

Training on Town06 would leave no unseen map with a highway on it. Town04 stays in training so the
model still sees freeway geometry, and it is the town that records the 50 deg `front_narrow`
camera; `towns/Town06.ps1` remains here for capturing the held-out set separately.

Town02 is captured last and on a reduced actor load (12 vehicles, 4 pedestrians, a short attempt
budget): with the four-camera rig at the standard load it crashed the simulator during warm-up on
every previous attempt, including with zero traffic and zero pedestrians, while other towns ran
the same configuration without issue. If it fails again the queue simply ends without it.

Town08 and Town09 do not exist in any public CARLA release - the numbering jumps from Town07 to
Town10, and those two are kept unreleased for Leaderboard evaluation.

## Usage

Start nothing by hand — the run scripts launch and supervise CARLA themselves.

```powershell
# one town, all weather presets
.\towns\Town10HD.ps1

# every town in sequence
.\scripts\run_multi_town_capture.ps1

# training capture, then the held-out validation capture, unattended
.\scripts\run_queue.ps1

# a single town/weather pair
.\scripts\run_supervised_capture.ps1 -Town Town07 -Weather foggy_night -Frames 182 `
    -Out D:\dataset\Town07\foggy_night
```

Requirements: CARLA 0.9.12 (packaged build) and a Python 3.7 environment with the matching `carla`
wheel, plus `numpy` and `Pillow`. `matplotlib` is optional: it supplies the turbo colour map for the
depth previews, which fall back to a plain red-to-blue ramp without it. Set `DEFAULT_CARLA_ROOT` in
`scripts/generate_adver_city_depth_dataset.py` and `$carlaExe` / `$py` in
`scripts/run_supervised_capture.ps1` to your own paths.

Every run is resumable: it counts existing frames in the output folder and captures only the
shortfall, so re-running after an interruption continues where it stopped.

`run_queue.ps1` chains the two passes that make up the dataset:

| Pass | Output | Frames per weather | Towns |
| --- | --- | --- | --- |
| training core | `dataset/` | 182 | Town10HD, Town07, Town04 |
| validation | `dataset_val/` | 60 | Town05, Town03 |
| training extension | `dataset/` | 182 | Town01, then Town02 (retry, reduced load) |

The validation pass writes to a separate root deliberately. Loaders discover runs by walking for
any directory containing `rgb/`, so a held-out town inside the training root is one careless glob
away from becoming training data - and an evaluation number that quietly became a training number
looks entirely normal in a report. Validation also needs coverage rather than volume, hence the
smaller budget.

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
