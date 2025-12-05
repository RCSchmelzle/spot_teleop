#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import yaml
from PIL import Image, ImageTk
from ament_index_python.packages import get_package_share_directory

from .gen_xtion_cameras import run_enumerator, load_mapping

HOME = Path.home()
XTION_MULTI_DIR = HOME / "Projects" / "teleoperation_spot" / "cpp" / "xtion_multi"
SNAPSHOT_BIN = XTION_MULTI_DIR / "xtion_snapshot"
OPENNI_LIB_DIR = HOME / "Projects" / "teleoperation_spot" / "cpp" / "OpenNI2" / "Bin" / "x64-Release"


def get_mapping_path() -> Path:
    share_dir = Path(get_package_share_directory("xtion_bringup"))
    config_dir = share_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "xtion_mapping.yaml"


def save_mapping(uri_to_role: dict[str, str]) -> None:
    mappings_list = [
        {"uri": uri, "role": role}
        for uri, role in uri_to_role.items()
        if role.strip()
    ]
    mapping_path = get_mapping_path()
    data = {"mappings": mappings_list}
    with mapping_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"[xtion_mapper_gui] Wrote mapping file {mapping_path}")


def capture_preview(uri: str) -> Image.Image | None:
    """
    Run the C++ xtion_snapshot tool for a given URI and return a PIL Image.
    xtion_snapshot writes a PPM file to disk; we load that.
    """
    import subprocess
    import traceback

    if not SNAPSHOT_BIN.exists():
        print(f"[xtion_mapper_gui] Snapshot binary not found: {SNAPSHOT_BIN}", file=sys.stderr)
        return None

    cache_dir = HOME / ".cache" / "xtion_previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_uri = uri.replace("/", "_").replace("@", "_")
    ppm_path = cache_dir / f"{safe_uri}.ppm"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(OPENNI_LIB_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        subprocess.run(
            [str(SNAPSHOT_BIN), uri, str(ppm_path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"[xtion_mapper_gui] Snapshot failed for {uri}", file=sys.stderr)
        return None
    except Exception:
        traceback.print_exc()
        return None

    if not ppm_path.exists():
        return None

    try:
        img = Image.open(ppm_path)
        img = img.convert("RGB")
        img.thumbnail((240, 180))
        return img
    except Exception as e:
        print(f"[xtion_mapper_gui] Failed to decode snapshot for {uri}: {e}", file=sys.stderr)
        return None


def main():
    # Discover current devices
    uris = run_enumerator()
    if not uris:
        print("[xtion_mapper_gui] No devices found. Plug in Xtions and try again.", file=sys.stderr)
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Xtion Mapper", "No Xtion devices found.\nPlug in cameras and try again.")
        root.destroy()
        return

    existing_mapping = load_mapping()

    uri_roles: dict[str, str] = {}
    for idx, uri in enumerate(uris):
        default_role = f"cam{idx}"
        uri_roles[uri] = existing_mapping.get(uri, default_role)

    suggested_roles = set(existing_mapping.values()) | set(uri_roles.values())
    suggested_roles |= {"front_left", "front_right", "overhead", "hand", "unused"}
    role_options = sorted(suggested_roles)

    root = tk.Tk()
    root.title("Xtion Camera Role Mapper")

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)

    ttk.Label(main_frame, text="URI", font=("TkDefaultFont", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=5, pady=5
    )
    ttk.Label(main_frame, text="Role / Name", font=("TkDefaultFont", 10, "bold")).grid(
        row=0, column=1, sticky="w", padx=5, pady=5
    )
    ttk.Label(main_frame, text="Preview", font=("TkDefaultFont", 10, "bold")).grid(
        row=0, column=2, sticky="w", padx=5, pady=5
    )

    row_widgets = []  # (uri, role_var, preview_label)
    for row_idx, uri in enumerate(uris, start=1):
        ttk.Label(main_frame, text=uri).grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)

        var = tk.StringVar(value=uri_roles[uri])
        combo = ttk.Combobox(main_frame, textvariable=var, values=role_options)
        combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        combo.configure(state="normal")

        preview_label = ttk.Label(main_frame, text="(click Refresh)")
        preview_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=2)

        row_widgets.append((uri, var, preview_label))

    def update_previews_once():
        for uri, _var, label in row_widgets:
            img = capture_preview(uri)
            if img is None:
                continue
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo, text="")
            label.image = photo  # keep reference

    def on_save():
        updated_mapping: dict[str, str] = {}
        for uri, var, _label in row_widgets:
            role = var.get().strip()
            if role:
                updated_mapping[uri] = role

        if not updated_mapping:
            if not messagebox.askyesno(
                "Confirm",
                "No roles specified. This will clear the mapping file.\nContinue?"
            ):
                return

        try:
            save_mapping(updated_mapping)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mapping:\n{e}")
            return

        messagebox.showinfo(
            "Saved",
            "Mapping saved.\n\nNow re-run:\n  python3 -m xtion_bringup.gen_xtion_cameras\nbefore launching multi_xtion."
        )
        root.destroy()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=len(uris) + 1, column=0, columnspan=3, sticky="e", pady=(10, 0))

    refresh_btn = ttk.Button(button_frame, text="Refresh previews", command=update_previews_once)
    refresh_btn.grid(row=0, column=0, padx=5)

    save_btn = ttk.Button(button_frame, text="Save & Close", command=on_save)
    save_btn.grid(row=0, column=1, padx=5)

    cancel_btn = ttk.Button(button_frame, text="Cancel", command=root.destroy)
    cancel_btn.grid(row=0, column=2, padx=5)

    # Grab one set of previews at startup
    update_previews_once()

    root.mainloop()


if __name__ == "__main__":
    main()
