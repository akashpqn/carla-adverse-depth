#!/usr/bin/env python
"""
Generate an adverse-weather CARLA dataset with dense depth (LiDAR + optional
CARLA depth camera), following Adver-City's own methodology:
    https://github.com/QUARRG/Adver-City  (arXiv:2410.06380)

This is NOT the OpenCDA multi-agent pipeline (no platoons/RSUs/V2X
annotations) - it is a single-ego CARLA 0.9.12 script that reuses
Adver-City's actual sensor rig and per-weather CARLA WeatherParameters
values (taken from Dataset/Configs/{default,Weather}/*.yaml in their repo),
and adds what Adver-City does NOT provide: a depth camera per viewpoint and
LiDAR-projected sparse/semi-dense depth with sky/dynamic-object masking.

Start CARLA first, then run for example:
    python generate_adver_city_depth_dataset.py --frames 500 --weather hard_rain_night
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import traceback
import zlib
from collections import deque
from queue import Empty, Queue

import numpy as np
from PIL import Image, ImageFilter


# C:, not D:. D: is an external USB drive that throws I/O retry events under load, and CARLA's
# map-asset streaming during load_world() was failing/crashing the server because of it - the
# server survives 90s idle but dies as soon as a client asks it to load a map from that drive.
DEFAULT_CARLA_ROOT = r"C:\CARLA_0.9.12\WindowsNoEditor"

# CARLA semantic segmentation tag ids (CARLA 0.9.12 default palette).
# https://carla.readthedocs.io/en/0.9.12/ref_sensors/#semantic-segmentation-camera
SKY_LABEL = 13
PEDESTRIAN_LABEL = 4
VEHICLE_LABEL = 10
DYNAMIC_LABEL = 20
DYNAMIC_LABELS = {PEDESTRIAN_LABEL, VEHICLE_LABEL, DYNAMIC_LABEL}

# Adver-City's 4-camera rig, Dataset/Configs/default.yaml -> vehicle_base.sensing.perception.camera
# positions are [x, y, z, yaw] relative to the vehicle, tuned for vehicle.lincoln.mkz_2017.
# NOTE: "front" is raised/moved forward from Adver-City's published [0.45, 0, 1.45, 0] - at that
# mount point with a 100 deg FOV, the ego's own hood filled ~17% of every single frame (measured:
# ~87,900 near-constant "vehicle" px in every frame, rows 268-539 of 540, regardless of real
# traffic). This is our own deliberate deviation for data quality, not part of their methodology.
CAMERA_RIG = [
    {"name": "front", "position": [0.7, 0.0, 1.75, 0.0]},
    {"name": "right", "position": [-0.28, 0.65, 1.52, 100.0]},
    {"name": "left", "position": [-0.28, -0.65, 1.52, -100.0]},
    {"name": "back", "position": [-2.42, 0.0, 1.10, 180.0]},
]

# Same source: vehicle_base.sensing.perception.camera (minus image size/fov, passed separately).
CAMERA_PARAMS = {
    "bloom_intensity": 0.675,
    "fstop": 1.4,
    "iso": 100.0,
    "gamma": 2.2,
    "lens_flare_intensity": 0.1,
    "shutter_speed": 200.0,
}

# LiDAR mount: not published by Adver-City (it comes from OpenCDA's LidarSensor default),
# so this is our own reasonable roof-center placement.
LIDAR_POSITION = [0.0, 0.0, 1.9, 0.0]

# Dataset/Configs/default.yaml -> vehicle_base.sensing.perception.lidar (clean, no sensor noise)
LIDAR_PROFILE_CLEAN = {
    "channels": 32,
    "range": 200.0,
    "points_per_second": 1200000,
    "upper_fov": 15.0,
    "lower_fov": -25.0,
    "dropoff_general_rate": 0.0,
    "dropoff_intensity_limit": 1.0,
    "dropoff_zero_intensity": 0.0,
    "noise_stddev": 0.0,
}

# Dataset/Configs/default.yaml -> rsu_base.sensing.perception.lidar (adds dropoff + noise)
LIDAR_PROFILE_NOISY = dict(LIDAR_PROFILE_CLEAN)
LIDAR_PROFILE_NOISY.update(
    {
        "dropoff_general_rate": 0.3,
        "dropoff_intensity_limit": 0.7,
        "dropoff_zero_intensity": 0.4,
        "noise_stddev": 0.02,
    }
)

LIDAR_PROFILES = {"clean": LIDAR_PROFILE_CLEAN, "noisy": LIDAR_PROFILE_NOISY}

# Exact CARLA WeatherParameters values from Adver-City's Dataset/Configs/default.yaml (base) and
# Dataset/Configs/Weather/*.yaml (per-condition overrides). glare_day's sun angles are scenario-
# dependent in their configs (they vary per road layout); the "ui" (urban intersection) values are
# used here as a representative default.
_WEATHER_BASE = {
    "rayleigh_scattering_scale": 0.02,
    "scattering_intensity": 1.0,
}

WEATHER_PRESETS = {
    "clear_day": dict(_WEATHER_BASE, cloudiness=3, fog_density=2, wetness=0, precipitation_deposits=0,
                       fog_distance=0.75, fog_falloff=0.1, precipitation=0, wind_intensity=10, sun_altitude_angle=60),
    "clear_night": dict(_WEATHER_BASE, cloudiness=3, fog_density=2, wetness=0, precipitation_deposits=0,
                         fog_distance=0.75, fog_falloff=0.1, precipitation=0, wind_intensity=10, sun_altitude_angle=-90),
    "soft_rain_day": dict(_WEATHER_BASE, cloudiness=70, fog_density=5, wetness=80, precipitation_deposits=20,
                           fog_distance=0.75, fog_falloff=0.1, precipitation=30, wind_intensity=30, sun_altitude_angle=60),
    "soft_rain_night": dict(_WEATHER_BASE, cloudiness=70, fog_density=7, wetness=90, precipitation_deposits=25,
                             fog_distance=0.75, fog_falloff=0.1, precipitation=30, wind_intensity=30, sun_altitude_angle=-90),
    "hard_rain_day": dict(_WEATHER_BASE, cloudiness=95, fog_density=7, wetness=100, precipitation_deposits=85,
                           fog_distance=0.75, fog_falloff=0.1, precipitation=100, wind_intensity=90, sun_altitude_angle=60),
    "hard_rain_night": dict(_WEATHER_BASE, cloudiness=95, fog_density=9, wetness=100, precipitation_deposits=90,
                             fog_distance=0.75, fog_falloff=0.1, precipitation=100, wind_intensity=90, sun_altitude_angle=-90),
    # fog_falloff: 0 is Adver-City's own published value, but in CARLA/UE4's exponential height-fog
    # shader that value effectively disables the visible effect (measured: no visible fog at 0).
    # Using CARLA's own API default of 1.0 instead - same density/distance philosophy, but actually
    # renders.
    "foggy_day": dict(_WEATHER_BASE, cloudiness=100, fog_density=100, wetness=5, precipitation_deposits=0,
                       fog_distance=0.0, fog_falloff=1.0, precipitation=0, wind_intensity=0, sun_altitude_angle=60),
    "foggy_night": dict(_WEATHER_BASE, cloudiness=100, fog_density=100, wetness=5, precipitation_deposits=0,
                         fog_distance=0.0, fog_falloff=1.0, precipitation=0, wind_intensity=0, sun_altitude_angle=-90),
    "foggy_hard_rain_day": dict(_WEATHER_BASE, cloudiness=100, fog_density=100, wetness=100, precipitation_deposits=85,
                                 fog_distance=0.0, fog_falloff=1.0, precipitation=100, wind_intensity=90, sun_altitude_angle=60),
    "foggy_hard_rain_night": dict(_WEATHER_BASE, cloudiness=100, fog_density=100, wetness=100, precipitation_deposits=90,
                                   fog_distance=0.0, fog_falloff=1.0, precipitation=100, wind_intensity=90, sun_altitude_angle=-90),
    # sun_azimuth_angle is a placeholder here - Adver-City's published 271 deg is calibrated to their
    # own scenario's fixed road heading ("set so that sunlight hits straight onto ego's path"), which
    # has no reason to match our own random spawn points. main() overrides this at runtime to the
    # ego's actual spawn yaw, replicating their stated intent generically instead of their constant.
    "glare_day": dict(_WEATHER_BASE, cloudiness=0, fog_density=8, wetness=0, precipitation_deposits=0,
                       fog_distance=0.75, fog_falloff=0.2, precipitation=0, wind_intensity=0,
                       # altitude 15, not Adver-City's 8: at 8 deg the sun sat behind buildings even
                       # on an open map (0 blown-out px); 15 clears them and still reads as low sun.
                       sun_altitude_angle=15, sun_azimuth_angle=271, mie_scattering_scale=0.0),
}


def add_carla_python_api(carla_root):
    python_api = os.path.join(carla_root, "PythonAPI")
    dist_dir = os.path.join(python_api, "carla", "dist")
    exact_pattern = os.path.join(
        dist_dir,
        "carla-*%d.%d-%s.egg"
        % (
            sys.version_info.major,
            sys.version_info.minor,
            "win-amd64" if os.name == "nt" else "linux-x86_64",
        ),
    )

    matches = glob.glob(exact_pattern)
    if not matches:
        matches = glob.glob(os.path.join(dist_dir, "carla-*.egg"))

    for path in matches:
        sys.path.append(path)
    sys.path.append(os.path.join(python_api, "carla"))


def make_dirs(root, save_depth_npy, depth_view):
    names = [
        "rgb",
        "semantic_label",
        "sky_mask",
        "lidar",
        "sparse_depth_u16",
        "semi_dense_depth_u16",
        "confidence",
        "metadata",
    ]
    if save_depth_npy:
        names += ["sparse_depth_m", "semi_dense_depth_m"]
    if depth_view:
        names += ["sparse_depth_view", "semi_dense_depth_view"]
    for name in names:
        os.makedirs(os.path.join(root, name), exist_ok=True)


def sensor_callback(data, queue):
    queue.put(data)


def retrieve_data(queue, frame, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = queue.get(True, max(0.01, deadline - time.time()))
        except Empty:
            break
        if data.frame == frame:
            return data
    return None


def set_weather(carla, world, preset_name, sun_azimuth_override=None):
    params = dict(WEATHER_PRESETS[preset_name])
    if sun_azimuth_override is not None and "sun_azimuth_angle" in params:
        params["sun_azimuth_angle"] = sun_azimuth_override
    weather = carla.WeatherParameters()
    for key, value in params.items():
        if hasattr(weather, key):
            setattr(weather, key, value)
    world.set_weather(weather)
    return params


def set_attr_if(bp, name, value):
    if bp.has_attribute(name):
        bp.set_attribute(name, str(value))


def spawn_point_estimation(carla, position):
    """Mirrors Adver-City's CameraSensor.spawn_point_estimation (single-ego case, no global_position)."""
    x, y, z, yaw = position
    location = carla.Location(x=x, y=y, z=z)
    rotation = carla.Rotation(roll=0.0, yaw=yaw, pitch=0.0)
    return carla.Transform(location, rotation)


def build_camera_bp(bp_lib, sensor_id, width, height, fov, sensor_tick, extra_params=None):
    bp = bp_lib.find(sensor_id)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(sensor_tick))
    for key, value in (extra_params or {}).items():
        set_attr_if(bp, key, value)
    return bp


def build_lidar_bp(bp_lib, args, profile):
    bp = bp_lib.find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", str(profile["channels"]))
    bp.set_attribute("range", str(profile["range"]))
    bp.set_attribute("rotation_frequency", str(args.fps))
    bp.set_attribute("points_per_second", str(profile["points_per_second"]))
    bp.set_attribute("upper_fov", str(profile["upper_fov"]))
    bp.set_attribute("lower_fov", str(profile["lower_fov"]))
    set_attr_if(bp, "noise_stddev", profile["noise_stddev"])
    set_attr_if(bp, "dropoff_general_rate", profile["dropoff_general_rate"])
    set_attr_if(bp, "dropoff_intensity_limit", profile["dropoff_intensity_limit"])
    set_attr_if(bp, "dropoff_zero_intensity", profile["dropoff_zero_intensity"])
    return bp


def camera_intrinsics(width, height, fov_degrees):
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    k = np.identity(3, dtype=np.float32)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1]


def apply_lens_droplets(rgb_image, precipitation, wind_intensity, seed):
    """
    Stylized wet-lens/windshield droplet overlay, composited onto the saved RGB frame only.
    CARLA has no built-in camera-lens rain shader (that needs a custom UE4 material in a source
    build, which isn't available here) - this approximates it in post: each droplet locally
    replaces the sharp image with a blurred, brightened patch (refraction/magnification), plus an
    optional gravity streak in windy rain. Depth/semantic/LiDAR ground truth is untouched, matching
    the degraded-input/clean-target philosophy in the README.
    """
    if precipitation <= 0:
        return rgb_image

    rng = np.random.RandomState(seed)
    h, w = rgb_image.shape[:2]
    short_side = min(h, w)

    blurred = np.array(
        Image.fromarray(rgb_image).filter(ImageFilter.GaussianBlur(radius=max(3, int(short_side * 0.02))))
    ).astype(np.float32)
    out = rgb_image.astype(np.float32)

    num_droplets = int(np.interp(precipitation, [0, 100], [4, 40]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    for _ in range(num_droplets):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        r = rng.uniform(short_side * 0.012, short_side * 0.04)

        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        px, py = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        # A drop on the lens is a tiny inverted lens: a solid disc with a soft edge showing a
        # heavily blurred, upside-down, slightly brighter view, a faint dark meniscus at the edge
        # and a small specular highlight. (A bright outline ring - the first version - read as
        # hollow circles, not water.)
        mask = np.clip((r - dist) / (0.25 * r), 0.0, 1.0)
        edge = np.clip(1.0 - np.abs(dist / r - 0.92) * 10, 0.0, 1.0) * 0.3
        hx, hy, hs = cx - 0.35 * r, cy - 0.35 * r, 0.12 * r
        spec = np.exp(-((px - hx) ** 2 + (py - hy) ** 2) / (2 * hs * hs)) * 0.45

        patch = out[y0:y1, x0:x1]
        refracted = blurred[y0:y1, x0:x1][::-1] * 1.08
        for c in range(3):
            drop = refracted[:, :, c] * (1 - edge) + spec * 255.0
            patch[:, :, c] = patch[:, :, c] * (1 - mask) + drop * mask

        # gravity streak trailing below the droplet, more likely/longer in windier rain
        if wind_intensity > 20 and rng.random() < 0.5:
            streak_len = int(r * rng.uniform(3, 7))
            for s in range(streak_len):
                sy = int(cy + r + s)
                if sy >= h:
                    break
                streak_r = max(1.0, r * 0.5 * (1 - s / streak_len))
                sx0, sx1 = max(0, int(cx - streak_r)), min(w, int(cx + streak_r))
                if sx1 <= sx0:
                    continue
                sdist = np.abs(xx[sy, sx0:sx1] - cx)
                smask = np.clip(1.0 - sdist / streak_r, 0.0, 1.0) * 0.22 * (1 - s / streak_len)
                for c in range(3):
                    out[sy, sx0:sx1, c] = out[sy, sx0:sx1, c] * (1 - smask) + blurred[sy, sx0:sx1, c] * smask

    return np.clip(out, 0, 255).astype(np.uint8)


def semantic_to_labels(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, 2].copy()


def depth_image_to_meters(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4)).astype(np.float32)
    normalized = (
        array[:, :, 2]
        + array[:, :, 1] * 256.0
        + array[:, :, 0] * 256.0 * 256.0
    ) / (256.0 * 256.0 * 256.0 - 1.0)
    return normalized * 1000.0


def lidar_to_local_points(lidar_data):
    data = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(data, (-1, 4))
    return points


def local_lidar_to_world(lidar_data, lidar_actor):
    points = lidar_to_local_points(lidar_data)
    local = points[:, :3].T
    local_h = np.r_[local, [np.ones(local.shape[1])]]
    lidar_to_world = np.array(lidar_actor.get_transform().get_matrix(), dtype=np.float32)
    world_points = np.dot(lidar_to_world, local_h)
    return world_points, points[:, 3]


def project_world_points(world_points, camera_actor, k, width, height, max_depth):
    world_to_camera = np.array(camera_actor.get_transform().get_inverse_matrix(), dtype=np.float32)
    sensor_points = np.dot(world_to_camera, world_points)

    # CARLA/UE coordinates to standard camera coordinates: (x, y, z) -> (y, -z, x)
    camera_points = np.array(
        [sensor_points[1], -sensor_points[2], sensor_points[0]],
        dtype=np.float32,
    )

    z = camera_points[2]
    valid_z = (z > 0.1) & (z < max_depth)
    camera_points = camera_points[:, valid_z]
    z = z[valid_z]

    if camera_points.shape[1] == 0:
        return np.zeros((height, width), dtype=np.float32)

    projected = np.dot(k, camera_points)
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]

    in_canvas = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_canvas].astype(np.int32)
    v = v[in_canvas].astype(np.int32)
    z = z[in_canvas]

    flat_index = v * width + u
    flat_depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flat_depth, flat_index, z)
    depth = flat_depth.reshape((height, width))
    depth[~np.isfinite(depth)] = 0.0
    return depth


# 16-bit PNGs store depth in CENTIMETERS (value / 100.0 = meters), giving a 0-655.35m range.
# (Millimeters would overflow uint16 past 65.535m, which is well inside typical scene depth.)
DEPTH_PNG_UNITS_PER_METER = 100.0


def turbo_lut():
    """
    256x3 uint8 colour table, cached. matplotlib's turbo if it is installed, otherwise a red->blue
    ramp so the previews still render on a bare install.
    """
    if turbo_lut.cache is None:
        try:
            from matplotlib import cm
            turbo_lut.cache = (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        except Exception:
            ramp = np.linspace(0, 1, 256)
            turbo_lut.cache = np.stack(
                [255 * (1 - ramp), 255 * ramp * 0.6, 255 * ramp], axis=1
            ).astype(np.uint8)
    return turbo_lut.cache


turbo_lut.cache = None


def colourise_depth(depth_m, max_depth):
    """
    Depth -> an 8-bit RGB image a viewer can actually show. The u16 PNGs hold centimetres, so an
    ordinary viewer stretches 0..65535 and renders a 200m scene as near-black; this maps depth
    through turbo on a SQUARE-ROOT scale (most pixels are close, so a linear ramp crushes all the
    near detail into one colour). Unmeasured pixels stay black: no data, not zero distance.
    """
    valid = depth_m > 0
    scaled = np.sqrt(np.clip(depth_m, 0, max_depth) / max_depth)
    idx = np.clip((scaled * 255).astype(np.int32), 0, 255)
    out = np.zeros(depth_m.shape + (3,), dtype=np.uint8)
    out[valid] = turbo_lut()[idx[valid]]
    return out


def save_depth(root, subdir_npy, subdir_png, stem, depth_m, max_depth, save_npy, subdir_view=None):
    if save_npy:
        np.save(os.path.join(root, subdir_npy, stem + ".npy"), depth_m.astype(np.float32))
    depth_units = np.clip(
        depth_m * DEPTH_PNG_UNITS_PER_METER, 0, min(max_depth * DEPTH_PNG_UNITS_PER_METER, 65535.0)
    ).astype(np.uint16)
    Image.fromarray(depth_units).save(os.path.join(root, subdir_png, stem + ".png"))
    if subdir_view:
        Image.fromarray(colourise_depth(depth_m, max_depth)).save(
            os.path.join(root, subdir_view, stem + ".png")
        )


def save_ply(path, world_points, intensity):
    xyz = world_points[:3, :].T
    with open(path, "w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("element vertex %d\n" % xyz.shape[0])
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float intensity\n")
        handle.write("end_header\n")
        for point, value in zip(xyz, intensity):
            handle.write("%.4f %.4f %.4f %.6f\n" % (point[0], point[1], point[2], value))


def save_lidar_npy(path, world_points, intensity):
    xyz_i = np.column_stack((world_points[:3, :].T, intensity.astype(np.float32)))
    np.save(path, xyz_i.astype(np.float32))


def choose_vehicle_bps(bp_lib):
    bps = []
    for bp in bp_lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4:
            bps.append(bp)
    return bps


def spawn_vehicle(world, bp, spawn_points, autopilot=False):
    for transform in spawn_points:
        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            actor.set_autopilot(autopilot)
            return actor
    return None


def sort_points_near_first(spawn_points, center, radius):
    """
    Splits spawn_points into "near" (within radius of center, sorted closest-first) and "far"
    (everything else, left shuffled), near first. Matches Adver-City's own spawning_distance
    concept (Dataset/Configs/default.yaml: spawning points generated within a radius of the ego)
    instead of scattering traffic uniformly across the whole town, which left most captured frames
    with no traffic/pedestrians actually in view.
    """
    if center is None:
        return spawn_points
    near, far = [], []
    for point in spawn_points:
        (near if point.location.distance(center) <= radius else far).append(point)
    near.sort(key=lambda p: p.location.distance(center))
    return near + far


def glare_azimuth_for(ego_yaw, index=0):
    """
    Sun azimuth that actually puts the sun in the cameras' view, given the ego's heading.

    CARLA's sun_azimuth_angle does NOT share vehicle yaw's zero direction: setting azimuth = yaw
    puts the sun BEHIND the car (measured on an open map: 0 blown-out pixels at every altitude).
    Sweeping the full circle showed the sun lands dead ahead at yaw+180, so a camera mounted at
    relative yaw `c` sees it at roughly yaw + 180 + c.

    Measured sweep of blown-out px against offset from ego yaw:
        +180: 2246   +210: 6059   +240: 4139   +270: 19   +300 and beyond: 0
    i.e. the sun lights the FRONT camera at roughly yaw+180..+240. A camera mounted at relative yaw
    `c` therefore needs about yaw + 210 + c.

    `index` walks deterministically through the rig (front, right, left, back) on successive calls,
    so a set gets glare in every camera across its frames rather than only straight ahead - the sun
    is one light source, so no single frame can have it in all four at once.
    """
    targets = [cam["position"][3] for cam in CAMERA_RIG]
    target = targets[index % len(targets)]
    return (ego_yaw + 210.0 + target + random.uniform(-25.0, 25.0)) % 360.0


def road_ahead_transforms(carla, carla_map, location, distances, own_lane_min=40.0):
    """
    Spawn transforms on the road AHEAD of `location`: the same lane plus any neighbouring driving
    lanes (including oncoming ones) at each distance. "Near the ego" alone wasn't enough - nearby
    spawn points are often on side streets or behind buildings, and a measured Town06 sample had
    zero traffic in any camera. Traffic placed along the ego's own road stays in the front view.
    """
    base = carla_map.get_waypoint(location)
    out = []
    for d in distances:
        for wp in base.next(d):
            # Keep the ego's own lane clear close in: a queue of cars spawned right in front of it
            # (first version: from 12 m at 8 m spacing) left the ego standing still for whole runs.
            own = [wp] if d >= own_lane_min else []
            for lane in own + [wp.get_left_lane(), wp.get_right_lane()]:
                if lane is None or lane.lane_type != carla.LaneType.Driving:
                    continue
                t = carla.Transform(lane.transform.location + carla.Location(z=0.5), lane.transform.rotation)
                out.append(t)
    return out


def find_navigation_location_near(world, center, radius, max_attempts=30):
    """Rejection-samples CARLA's random-nav-point API for a point within radius of center,
    falling back to an unrestricted point if the area is too sparse to find one in budget."""
    fallback = None
    for _ in range(max_attempts):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        fallback = fallback or location
        if center is None or location.distance(center) <= radius:
            return location
    return fallback


def apply_vehicle_lights(carla, vehicles):
    try:
        lights = (
            carla.VehicleLightState.Position
            | carla.VehicleLightState.LowBeam
            | carla.VehicleLightState.Fog
        )
    except AttributeError:
        return
    for vehicle in vehicles:
        if vehicle is not None and vehicle.is_alive:
            vehicle.set_light_state(carla.VehicleLightState(lights))


def spawn_walkers(carla, world, count, near_location=None, near_radius=None):
    """
    Spawns pedestrians with AI controllers, Adver-City's 6th object category alongside vehicles.

    near_location/near_radius bias both the spawn point and the walk destination toward the ego's
    area (rejection-sampled - CARLA has no direct "random point within radius" API). Unrestricted
    random.get_random_location_from_navigation() scatters walkers across the WHOLE town's sidewalk
    network, so in a short capture window they almost never end up near the ego's actual path -
    measured: 18px of pedestrian visible across 48 sampled frames with 30 unrestricted walkers.
    """
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        return [], []

    walkers = []
    for _ in range(count):
        if near_location is not None and near_radius is not None:
            location = find_navigation_location_near(world, near_location, near_radius)
        else:
            location = world.get_random_location_from_navigation()
        if location is None:
            continue
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = world.try_spawn_actor(bp, carla.Transform(location))
        if actor is not None:
            walkers.append(actor)

    if not walkers:
        return [], []
    world.tick()

    controller_bp = bp_lib.find("controller.ai.walker")
    controllers = []
    for walker in walkers:
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        controllers.append(controller)
    world.tick()

    for controller in controllers:
        controller.start()
        if near_location is not None and near_radius is not None:
            destination = find_navigation_location_near(world, near_location, near_radius)
        else:
            destination = world.get_random_location_from_navigation()
        if destination is not None:
            controller.go_to_location(destination)
        controller.set_max_speed(1.0 + random.random())

    return walkers, controllers


def move_spectator(carla, world, ego):
    transform = ego.get_transform()
    forward = transform.rotation.get_forward_vector()
    location = transform.location - (forward * 8.0) + carla.Location(z=3.5)
    rotation = carla.Rotation(pitch=-12.0, yaw=transform.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def parse_camera_names(value):
    names = [name.strip() for name in value.split(",") if name.strip()]
    valid = {cam["name"] for cam in CAMERA_RIG}
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise argparse.ArgumentTypeError("Unknown camera name(s) %s, choose from %s" % (unknown, sorted(valid)))
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--traffic-manager-port", default=8000, type=int)
    parser.add_argument("--carla-root", default=DEFAULT_CARLA_ROOT)
    parser.add_argument(
        "--town",
        default="Town01",
        help="Town11 is CARLA's heaviest map and crashed (DXGI device hung/removed) on this machine's "
        "4GB-VRAM GPU; defaulting to a lighter town instead",
    )
    parser.add_argument("--out", default=r"D:\adver_city_depth_dataset\output")
    parser.add_argument("--weather", default="hard_rain_night", choices=sorted(WEATHER_PRESETS))
    parser.add_argument("--frames", default=300, type=int)
    parser.add_argument("--warmup", default=30, type=int)
    parser.add_argument("--fps", default=10, type=float, help="also sets LiDAR rotation_frequency (Adver-City: 10)")
    parser.add_argument("--width", default=1920, type=int, help="Adver-City default: 1920")
    parser.add_argument("--height", default=1080, type=int, help="Adver-City default: 1080")
    parser.add_argument("--fov", default=100.0, type=float, help="Adver-City default: 100")
    parser.add_argument(
        "--cameras",
        default="front,right,left,back",
        type=parse_camera_names,
        help="comma-separated subset of Adver-City's 4-camera rig: front,right,left,back",
    )
    parser.add_argument("--traffic", default=40, type=int)
    parser.add_argument("--walkers", default=25, type=int)
    parser.add_argument(
        "--spawn-radius",
        default=60.0,
        type=float,
        help="traffic/walkers are spawned within this many meters of the ego first (Adver-City's own "
        "spawning_distance concept), falling back to farther points/nav locations only if not enough "
        "are found nearby - keeps them actually visible instead of scattered across the whole town",
    )
    parser.add_argument("--seed", default=13, type=int, help="Adver-City default: 13")
    parser.add_argument("--accumulate-sweeps", default=5, type=int)
    parser.add_argument("--min-move", default=1.0, type=float,
                        help="metres the ego must travel between saved frames (skips near-duplicates)")
    parser.add_argument("--max-stall-ticks", default=300, type=int,
                        help="end the run if the ego stays within --min-move for this many ticks")
    parser.add_argument("--save-carla-depth", action="store_true", help="also save CARLA's ideal z-buffer depth per camera")
    parser.add_argument(
        "--no-depth-view",
        dest="depth_view",
        action="store_false",
        help="skip the *_depth_view folders. On by default: each depth map also gets an 8-bit turbo "
        "colour preview, since the u16 depth PNGs themselves look black in an ordinary image viewer. "
        "The previews are for looking at - train on the u16 PNGs, which hold the actual centimetres.",
    )
    parser.set_defaults(depth_view=True)
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument(
        "--save-depth-npy",
        action="store_true",
        help="also save float32 .npy depth (meters) alongside the u16 PNG (centimeters). Off by default: "
        "each .npy is ~8MB uncompressed at 1920x1080 x2 per camera, dwarfing every other output file; the "
        "u16 PNG already has 1cm precision and compresses well since most of the frame is zero.",
    )
    parser.add_argument("--max-depth", default=120.0, type=float)
    parser.add_argument(
        "--lidar-profile",
        default="clean",
        choices=sorted(LIDAR_PROFILES),
        help="clean = Adver-City vehicle_base (no sensor noise), noisy = Adver-City rsu_base (dropoff + noise)",
    )
    args = parser.parse_args()

    make_dirs(args.out, args.save_depth_npy, args.depth_view)
    # Vary the seed with how many frames are already captured, not just args.seed: this script
    # restarts from scratch (fresh ego/traffic spawn) on every crash-recovery, and a fixed seed would
    # replay the exact same opening seconds of the exact same scenario every single restart - nearly
    # identical frames and no traffic diversity, which is exactly what happened before this fix.
    # Also mix in town/weather and the wall clock. Seeding only on frames-already-captured gave
    # every new weather folder the same seed (13 + 0), so clear_day, clear_night and soft_rain_day
    # were the identical drive with only the sky changed. The clock also keeps a restart after a
    # stuck scene from replaying that same stuck scene.
    already_captured_for_seed = len(glob.glob(os.path.join(args.out, "metadata", "*.json")))
    run_seed = (
        args.seed
        + zlib.crc32(("%s:%s" % (args.town, args.weather)).encode())
        + already_captured_for_seed * 7919
        + int(time.time())
    ) % (2 ** 31)
    random.seed(run_seed)
    np.random.seed(run_seed)
    if args.save_carla_depth:
        if args.save_depth_npy:
            os.makedirs(os.path.join(args.out, "carla_depth_m"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "carla_depth_u16"), exist_ok=True)
        if args.depth_view:
            os.makedirs(os.path.join(args.out, "carla_depth_view"), exist_ok=True)

    active_cameras = [cam for cam in CAMERA_RIG if cam["name"] in args.cameras]

    add_carla_python_api(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    # The traffic manager RPC service has been observed to hang indefinitely (never responds, even
    # given 90s+) while the main world connection works fine - a distinct failure mode from this
    # project's other CARLA instability. It's a convenience layer (lane-change/speed tuning) on top
    # of autopilot, not required for it - set_autopilot(True) still works without an explicit TM
    # handle, CARLA auto-connects its own default one server-side. So this is best-effort: a short
    # dedicated timeout, and every use below is guarded so a hung TM degrades tuning, not the run.
    traffic_manager = None
    try:
        client.set_timeout(10.0)
        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        # Dataset/Configs/default.yaml -> carla_traffic_manager
        traffic_manager.set_global_distance_to_leading_vehicle(5.0)
        traffic_manager.global_percentage_speed_difference(-10.0)
    except RuntimeError as exc:
        print("WARNING: traffic manager unavailable (%s) - continuing without lane-change/speed tuning." % exc)
        traffic_manager = None
    finally:
        # 120s, not 60s: load_world() is disk-bound reading map assets from an external USB drive
        # that measurably struggles (disk I/O retry events in Event Viewer) under the restart-heavy
        # workload this script's own crash-recovery produces - 60s wasn't always enough.
        client.set_timeout(120.0)

    # Initialise pedestrian navigation on the map CARLA booted into BEFORE switching towns. On this
    # machine, the first-ever nav query landing on a map switched in via load_world() crashes the
    # server (measured: every run with walkers died right after Nav/<Town>.bin loaded, while
    # --walkers 0 ran clean), but once nav has been initialised on the boot map, a subsequent
    # load_world() + walker spawn works (20 walkers on Town02 through 50 ticks).
    if args.walkers > 0:
        client.get_world().get_random_location_from_navigation()

    print("Loading %s..." % args.town)
    world = client.load_world(args.town)
    print("  [setup] map loaded")
    original_settings = world.get_settings()

    actors = []
    walkers = []
    walker_controllers = []
    sensors = []
    queues = {}
    sweep_buffers = {cam["name"]: deque(maxlen=max(1, args.accumulate_sweeps)) for cam in active_cameras}

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)

        # NOTE: there is deliberately no "wait for the pedestrian nav mesh" loop here. An earlier
        # version polled get_random_location_from_navigation() until it returned non-None, on the
        # theory that walkers were failing to spawn because the nav mesh loads lazily. That loop
        # never once succeeded (it burned its full timeout on every run, with and without ticking)
        # and the sustained hammering immediately after world load was itself killing the server.
        # CARLA's own PythonAPI/examples/generate_traffic.py just calls that API directly with no
        # readiness check, so we do the same and let individual walkers fail to spawn if the nav
        # query comes back empty - spawn_walkers() already skips those.

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Ego spawns first so traffic/walker placement can be biased toward it below - otherwise
        # everything scatters uniformly across the whole town and rarely ends up in view.
        vehicle_bps = choose_vehicle_bps(bp_lib)
        ego_bps = bp_lib.filter("vehicle.lincoln.mkz_2017") or bp_lib.filter("vehicle.tesla.model3") or vehicle_bps
        ego = spawn_vehicle(world, ego_bps[0], spawn_points, autopilot=True)
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle.")
        if traffic_manager is not None:
            traffic_manager.auto_lane_change(ego, False)
        actors.append(ego)
        # In synchronous mode a freshly spawned actor reports a zero transform until the world ticks.
        # Without this tick, ego_location was (0,0,0): traffic got placed around the MAP ORIGIN and
        # glare_day's sun was aimed at yaw 0, rather than at the ego.
        world.tick()
        ego_location = ego.get_transform().location
        print("  [setup] ego at (%.0f, %.0f)" % (ego_location.x, ego_location.y))
        print("  [setup] ego spawned")

        ahead = road_ahead_transforms(carla, world.get_map(), ego_location, range(12, 150, 12))
        traffic_spawn_points = ahead + sort_points_near_first(spawn_points, ego_location, args.spawn_radius)
        for _ in range(max(0, args.traffic)):
            if not traffic_spawn_points or not vehicle_bps:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            npc = spawn_vehicle(world, bp, traffic_spawn_points, autopilot=True)
            if npc is not None:
                if traffic_manager is not None:
                    traffic_manager.auto_lane_change(npc, False)
                actors.append(npc)

        # NB: walkers are deliberately NOT spawned here - see after the warmup ticks below.
        print("  [setup] %d traffic vehicles spawned" % (len(actors) - 1))

        apply_vehicle_lights(carla, actors)

        # Set after the ego spawns so glare_day can aim the sun relative to the ego's actual heading
        # (see WEATHER_PRESETS comment) rather than a scenario-specific constant that assumes a road
        # layout we don't have.
        ego_yaw = ego.get_transform().rotation.yaw % 360.0
        weather_params = set_weather(
            carla, world, args.weather,
            sun_azimuth_override=glare_azimuth_for(ego_yaw, already_captured_for_seed // 25),
        )
        print("  [setup] weather set")

        sensor_tick = 1.0 / args.fps
        lidar_profile = LIDAR_PROFILES[args.lidar_profile]
        lidar_tf = spawn_point_estimation(carla, LIDAR_POSITION)
        lidar = world.spawn_actor(build_lidar_bp(bp_lib, args, lidar_profile), lidar_tf, attach_to=ego)
        sensors.append(lidar)

        cameras = {}
        for cam in active_cameras:
            name = cam["name"]
            cam_tf = spawn_point_estimation(carla, cam["position"])

            rgb = world.spawn_actor(
                build_camera_bp(bp_lib, "sensor.camera.rgb", args.width, args.height, args.fov, sensor_tick, CAMERA_PARAMS),
                cam_tf,
                attach_to=ego,
            )
            semantic = world.spawn_actor(
                build_camera_bp(bp_lib, "sensor.camera.semantic_segmentation", args.width, args.height, args.fov, sensor_tick),
                cam_tf,
                attach_to=ego,
            )
            depth_camera = None
            if args.save_carla_depth:
                depth_camera = world.spawn_actor(
                    build_camera_bp(bp_lib, "sensor.camera.depth", args.width, args.height, args.fov, sensor_tick),
                    cam_tf,
                    attach_to=ego,
                )
                sensors.append(depth_camera)

            sensors.extend([rgb, semantic])
            cameras[name] = {"rgb": rgb, "semantic": semantic, "depth": depth_camera}

        for sensor_name, sensor in [("lidar", lidar)] + [
            ("%s_%s" % (kind, name), cameras[name][kind])
            for name in cameras
            for kind in ("rgb", "semantic", "depth")
        ]:
            if sensor is None:
                continue
            q = Queue()
            sensor.listen(lambda data, queue=q: sensor_callback(data, queue))
            queues[sensor_name] = q
        print("  [setup] %d sensors listening" % len(queues))

        k = camera_intrinsics(args.width, args.height, args.fov)
        intrinsics = {
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
            "width": args.width,
            "height": args.height,
            "fov": args.fov,
        }

        for _ in range(args.warmup):
            world.tick()
        print("  [setup] warmup done (%d ticks)" % args.warmup)

        # Walkers spawn HERE, after the warmup ticks, not alongside the vehicles. CARLA's pedestrian
        # navigation is not queryable immediately after load_world() - get_random_location_from_
        # navigation() returns None until the world has been ticked a while (measured on Town02:
        # None on 50/50 calls after 10 ticks, but working after 20), and a walker spawned from a
        # None location is silently skipped. Spawning after the warmup loop reuses ticks we already
        # do, so no extra polling/waiting is needed.
        # Bias to where the ego is NOW, not its spawn point - it has been driving on autopilot
        # throughout the warmup ticks above and has moved.
        # Centre them on the road ~25m AHEAD of the ego with a tight radius, so they're on the
        # sidewalks the front camera is about to pass, and let a share of them cross the road.
        ego_now = ego.get_transform().location
        ahead_wps = world.get_map().get_waypoint(ego_now).next(25.0)
        walker_center = ahead_wps[0].transform.location if ahead_wps else ego_now
        world.set_pedestrians_cross_factor(0.3)
        walkers, walker_controllers = spawn_walkers(
            carla, world, max(0, args.walkers),
            near_location=walker_center, near_radius=30.0,
        )
        if args.walkers > 0:
            print("Spawned %d/%d walkers." % (len(walkers), args.walkers))

        already_captured = len(glob.glob(os.path.join(args.out, "metadata", "*.json")))
        remaining = max(0, args.frames - already_captured)
        if already_captured:
            print(
                "Resuming: %d/%d frames already in %s, capturing %d more."
                % (already_captured, args.frames, args.out, remaining)
            )
        print("Recording %d frames (%s) to %s..." % (remaining, ", ".join(cameras), args.out))
        # Only save a frame once the ego has moved at least --min-move metres since the last saved
        # one. Before this, 99 of 182 frames in a folder were the ego standing still (waiting at a
        # light / behind traffic), i.e. near-duplicate images. If it stays stuck for
        # --max-stall-ticks, end the run; the supervisor restarts with a fresh scene.
        saved = 0
        last_saved_pos = None
        stall_ticks = 0
        while saved < remaining:
            frame = world.tick()

            lidar_data = retrieve_data(queues["lidar"], frame)
            per_cam_data = {}
            missing = lidar_data is None
            for name in cameras:
                rgb_data = retrieve_data(queues["rgb_%s" % name], frame)
                semantic_data = retrieve_data(queues["semantic_%s" % name], frame)
                depth_data = (
                    retrieve_data(queues["depth_%s" % name], frame) if cameras[name]["depth"] is not None else None
                )
                per_cam_data[name] = (rgb_data, semantic_data, depth_data)
                if rgb_data is None or semantic_data is None:
                    missing = True

            if missing:
                print("Skipping frame %s because one or more sensors missed the tick." % frame)
                continue

            ego_pos = ego.get_transform().location
            if last_saved_pos is not None and ego_pos.distance(last_saved_pos) < args.min_move:
                stall_ticks += 1
                if stall_ticks >= args.max_stall_ticks:
                    print("Ego barely moved for %d ticks - ending run so the supervisor restarts "
                          "with a new scene." % stall_ticks)
                    break
                continue
            stall_ticks = 0
            last_saved_pos = ego_pos
            i = saved

            # Re-aim the sun as the ego turns, otherwise glare drifts out of view after the first
            # corner and the rest of the run is just ordinary low sunlight. The fresh random offset
            # each time also moves the sun between the front, side and rear cameras.
            if "sun_azimuth_angle" in weather_params and saved and saved % 25 == 0:
                weather_params = set_weather(
                    carla, world, args.weather,
                    sun_azimuth_override=glare_azimuth_for(
                        ego.get_transform().rotation.yaw % 360.0, (already_captured + saved) // 25),
                )

            # A dataset-sequential index, not CARLA's own tick counter: the tick counter restarts near 0
            # every time the server process restarts, which would collide with (and silently overwrite)
            # filenames from a prior run when resuming after a crash.
            stem = "%06d" % (already_captured + i)
            world_points, intensity = local_lidar_to_world(lidar_data, lidar)
            save_lidar_npy(os.path.join(args.out, "lidar", stem + ".npy"), world_points, intensity)
            if args.save_ply:
                save_ply(os.path.join(args.out, "lidar", stem + ".ply"), world_points, intensity)

            frame_metadata = {
                "frame": int(frame),
                "weather": args.weather,
                "weather_parameters": weather_params,
                "map": world.get_map().name,
                "intrinsics": intrinsics,
                "ego_transform": ego.get_transform().__str__(),
                "lidar_transform": lidar.get_transform().__str__(),
                "lidar_profile": args.lidar_profile,
                "run_seed": run_seed,
                "cameras": {},
            }

            for name, (rgb_data, semantic_data, depth_data) in per_cam_data.items():
                rgb_camera = cameras[name]["rgb"]
                sweep_buffer = sweep_buffers[name]

                rgb_image = image_to_rgb(rgb_data)
                droplet_seed = (run_seed * 1000003 + (already_captured + i) * 101 + sum(ord(c) for c in name)) % (2**31)
                rgb_image = apply_lens_droplets(
                    rgb_image, weather_params.get("precipitation", 0), weather_params.get("wind_intensity", 0),
                    seed=droplet_seed,
                )
                labels = semantic_to_labels(semantic_data)
                sky_mask = labels == SKY_LABEL
                dynamic_or_sky = sky_mask | np.isin(labels, list(DYNAMIC_LABELS))

                sweep_buffer.append(world_points)
                sparse_depth = project_world_points(world_points, rgb_camera, k, args.width, args.height, args.max_depth)

                if len(sweep_buffer) > 1:
                    all_world_points = np.concatenate(list(sweep_buffer), axis=1)
                else:
                    all_world_points = world_points
                semi_dense_depth = project_world_points(
                    all_world_points, rgb_camera, k, args.width, args.height, args.max_depth
                )

                sparse_depth[sky_mask] = 0.0
                semi_dense_depth[dynamic_or_sky] = 0.0

                confidence = np.zeros((args.height, args.width), dtype=np.uint8)
                confidence[semi_dense_depth > 0.0] = 128
                confidence[sparse_depth > 0.0] = 255
                confidence[sky_mask] = 0

                file_stem = "%s_%s" % (stem, name)
                Image.fromarray(rgb_image).save(os.path.join(args.out, "rgb", file_stem + ".png"))
                Image.fromarray(labels).save(os.path.join(args.out, "semantic_label", file_stem + ".png"))
                Image.fromarray(sky_mask.astype(np.uint8) * 255).save(
                    os.path.join(args.out, "sky_mask", file_stem + ".png")
                )
                Image.fromarray(confidence).save(os.path.join(args.out, "confidence", file_stem + ".png"))
                save_depth(
                    args.out, "sparse_depth_m", "sparse_depth_u16", file_stem, sparse_depth, args.max_depth,
                    args.save_depth_npy, "sparse_depth_view" if args.depth_view else None,
                )
                save_depth(
                    args.out, "semi_dense_depth_m", "semi_dense_depth_u16", file_stem, semi_dense_depth,
                    args.max_depth, args.save_depth_npy,
                    "semi_dense_depth_view" if args.depth_view else None,
                )

                if depth_data is not None:
                    carla_depth = depth_image_to_meters(depth_data)
                    carla_depth[sky_mask] = 0.0
                    save_depth(
                        args.out, "carla_depth_m", "carla_depth_u16", file_stem, carla_depth, args.max_depth,
                        args.save_depth_npy, "carla_depth_view" if args.depth_view else None,
                    )

                frame_metadata["cameras"][name] = {
                    "camera_transform": rgb_camera.get_transform().__str__(),
                    "sparse_valid_pixels": int(np.count_nonzero(sparse_depth)),
                    "semi_dense_valid_pixels": int(np.count_nonzero(semi_dense_depth)),
                }

            with open(os.path.join(args.out, "metadata", stem + ".json"), "w", encoding="utf-8") as handle:
                json.dump(frame_metadata, handle, indent=2)

            move_spectator(carla, world, ego)
            saved += 1
            if saved % 10 == 0:
                print("Saved %d/%d frames this run (%d/%d total)" % (saved, remaining, already_captured + saved, args.frames))

    finally:
        print("Cleaning up...")
        for controller in walker_controllers:
            if controller is not None and controller.is_alive:
                controller.stop()
                controller.destroy()
        for walker in walkers:
            if walker is not None and walker.is_alive:
                walker.destroy()
        for sensor in sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)
        if traffic_manager is not None:
            traffic_manager.set_synchronous_mode(False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
