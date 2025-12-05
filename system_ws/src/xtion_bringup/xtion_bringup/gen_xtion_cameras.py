#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def run_enumerator() -> list[str]:
    """
    Run the C++ xtion_enumerate tool and return the list of device URIs.
    Uses the OpenNI2 build living under cpp/OpenNI2.
    """
    enumerator_path = Path.home() / "Projects" / "teleoperation_spot" / "cpp" / "xtion_multi" / "xtion_enumerate"
    lib_dir = Path.home() / "Projects" / "teleoperation_spot" / "cpp" / "OpenNI2" / "Bin" / "x64-Release"

    if not enumerator_path.exists():
        print(f"[gen_xtion_cameras] ERROR: {enumerator_path} not found. Did you build xtion_enumerate?", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(lib_dir) + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(
            [str(enumerator_path)],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[gen_xtion_cameras] ERROR running xtion_enumerate:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    uris: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines like: "URI:           1d27/0601@1/9"
        if line.startswith("URI:"):
            uri = line.split("URI:", 1)[1].strip()
            if uri:
                uris.append(uri)

    return uris


def load_mapping() -> dict[str, str]:
    """
    Load persistent mapping from URI -> role (logical name), if it exists.
    File format (xtion_mapping.yaml):

    mappings:
      - uri: "1d27/0601@1/9"
        role: "front_left"
      - uri: "1d27/0601@1/10"
        role: "front_right"
    """
    share_dir = Path(get_package_share_directory("xtion_bringup"))
    mapping_path = share_dir / "config" / "xtion_mapping.yaml"

    if not mapping_path.exists():
        return {}

    try:
        with mapping_path.open("r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[gen_xtion_cameras] WARNING: Failed to read mapping file {mapping_path}: {e}", file=sys.stderr)
        return {}

    mappings_list = data.get("mappings", [])
    uri_to_role: dict[str, str] = {}
    for item in mappings_list:
        uri = item.get("uri")
        role = item.get("role")
        if uri and role:
            uri_to_role[uri] = role

    return uri_to_role


def write_yaml(uris: list[str]) -> Path:
    """
    Write xtion_cameras.yaml into xtion_bringup's share/config directory.
    Uses role mapping when available; falls back to cam0, cam1, ...
    """
    share_dir = Path(get_package_share_directory("xtion_bringup"))
    config_dir = share_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = config_dir / "xtion_cameras.yaml"

    if not uris:
        print("[gen_xtion_cameras] WARNING: No Xtion devices found. Writing empty cameras list.", file=sys.stderr)

    mapping = load_mapping()

    cameras = []
    for idx, uri in enumerate(uris):
        default_name = f"cam{idx}"
        name = mapping.get(uri, default_name)
        cameras.append(
            {
                "name": name,
                "namespace": name,
                "device_id": uri,
            }
        )

    data = {"cameras": cameras}

    with yaml_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    print(f"[gen_xtion_cameras] Wrote {yaml_path}")
    for cam in cameras:
        print(f"  - {cam['name']}: {cam['device_id']}")

    return yaml_path


def main():
    uris = run_enumerator()
    write_yaml(uris)


if __name__ == "__main__":
    main()
