"""Build an OpenStudio model (.osm) from fit_planes.py's planes.json.

Standalone geometry->OpenStudio step: one Space (+ ThermalZone) per run, one
Surface per plane. No thermal/material data, no labeling beyond what
fit_planes.py already computed (orientation, synthetic).

OpenStudio does not require a fully closed solid to import -- surfaces with
no matching neighbour (open faces) are simply imported as-is (usually still
Outdoors-facing). --cap-open-faces in fit_planes.py is only needed if you
specifically want EnergyPlus's "is this an enclosed volume" checks to pass.

Winding: fit_planes.py's corner order isn't reliably outward-facing (the
--close-geometry box-stretch step reuses the same corner order for both
sides of an axis, so it can't be right for both). This script recomputes
each plane's actual implied normal from its corner order and reverses it
when it points into the room instead of out of it, using the room's overall
centroid as the inside reference -- OpenStudio needs correctly outward-
wound vertices for surface type/boundary-condition defaults to make sense.

Windows (--windows): fit_window_polygons.py's output, one SubSurface
(FixedWindow) per entry on its plane_id's Surface. Wound the same way as its
host wall (same outward_ref) so its normal agrees with the parent's -- an
inward-facing subsurface confuses EnergyPlus's view-factor/shading logic even
though it imports without error. Windows measured (or inferred) as holes in
the real voxel data, not guessed -- see that script's docstring for how.

Usage:
    python to_openstudio.py --planes planes.json [--out model.osm]
        [--name Corridor] [--exclude-synthetic] [--windows windows.json]

Venv: C:\\venvs\\planefit (same as fit_planes.py; `pip install openstudio`
added there -- openstudio 3.9+ has cp312/cp313 wheels, unlike open3d).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openstudio


def fix_winding(corners, outward_ref):
    """Reverse the vertex order if the polygon's implied normal (from its own
    point order) points opposite to `outward_ref` (room center -> plane
    center, i.e. "away from the room's inside")."""
    c = np.array(corners, dtype=float)
    implied = np.cross(c[1] - c[0], c[3] - c[0])
    if np.dot(implied, outward_ref) < 0:
        return corners[::-1]
    return corners


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=Path, default=Path("planes.json"))
    ap.add_argument("--out", type=Path, default=Path("model.osm"))
    ap.add_argument("--name", default="Corridor", help="Space/ThermalZone name")
    ap.add_argument("--exclude-synthetic", action="store_true",
                    help="skip synthetic/adiabatic cap faces from --cap-open-faces "
                         "(included by default -- OpenStudio doesn't need them, but "
                         "they don't hurt and make the volume calculable)")
    ap.add_argument("--windows", type=Path, default=None,
                    help="fit_window_polygons.py output -- adds one FixedWindow "
                         "SubSurface per entry on its plane_id's Surface")
    args = ap.parse_args()

    data = json.loads(args.planes.read_text())
    planes = data["planes"]
    if args.exclude_synthetic:
        before = len(planes)
        planes = [p for p in planes if not p.get("synthetic")]
        if len(planes) != before:
            print(f"excluded {before - len(planes)} synthetic cap face(s)")
    if not planes:
        raise SystemExit("no planes to build a model from")

    # translate to a local origin near (0,0,0) -- purely cosmetic, real-world
    # coordinates are recoverable by adding `origin` back
    allc = np.array([c for p in planes for c in p["corners_3d"]])
    origin = allc.min(axis=0)
    print(f"translating geometry by {(-origin).round(3).tolist()} "
          f"(original min corner {origin.round(3).tolist()})")

    room_center = np.mean([p["centroid_3d"] for p in planes], axis=0)

    # floor vs ceiling: among floor_ceiling planes, lower half of the z-range = Floor
    fc_z = [p["centroid_3d"][2] for p in planes if p["orientation"] == "floor_ceiling"]
    z_split = (min(fc_z) + max(fc_z)) / 2 if fc_z else 0.0

    model = openstudio.model.Model()
    space = openstudio.model.Space(model)
    space.setName(args.name)
    zone = openstudio.model.ThermalZone(model)
    zone.setName(f"{args.name} Zone")
    space.setThermalZone(zone)

    n_wall = n_floor = n_roof = n_adiabatic = 0
    surface_by_plane = {}      # plane id -> (Surface, outward_ref) for --windows below
    for p in planes:
        centroid = np.array(p["centroid_3d"])
        outward_ref = centroid - room_center
        corners = fix_winding(p["corners_3d"], outward_ref)
        pts = openstudio.Point3dVector(
            [openstudio.Point3d(*(np.array(c) - origin)) for c in corners])

        surface = openstudio.model.Surface(pts, model)
        surface.setSpace(space)
        surface.setName(f"plane_{p['id']}_{p['orientation']}")
        surface_by_plane[p["id"]] = (surface, outward_ref)

        if p["orientation"] == "wall":
            surface.setSurfaceType("Wall")
            n_wall += 1
        elif centroid[2] <= z_split:
            surface.setSurfaceType("Floor")
            n_floor += 1
        else:
            surface.setSurfaceType("RoofCeiling")
            n_roof += 1

        if p.get("synthetic"):
            surface.setOutsideBoundaryCondition("Adiabatic")
            surface.setSunExposure("NoSun")
            surface.setWindExposure("NoWind")
            n_adiabatic += 1
        elif surface.surfaceType() == "Floor":
            surface.setOutsideBoundaryCondition("Ground")
            surface.setSunExposure("NoSun")
            surface.setWindExposure("NoWind")
        else:
            surface.setOutsideBoundaryCondition("Outdoors")
            surface.setSunExposure("SunExposed")
            surface.setWindExposure("WindExposed")

    print(f"{len(planes)} surface(s): {n_wall} wall, {n_floor} floor, {n_roof} roof/ceiling "
          f"({n_adiabatic} adiabatic/synthetic)")

    if args.windows:
        win_data = json.loads(args.windows.read_text())["windows"]
        n_win = n_skipped = 0
        by_source = {}
        for w in win_data:
            pid = w["plane_id"]
            if pid not in surface_by_plane:
                print(f"WARNING: window on plane {pid} has no matching Surface -- skipped")
                n_skipped += 1
                continue
            parent, outward_ref = surface_by_plane[pid]
            corners = fix_winding(w["corners_3d"], outward_ref)
            pts = openstudio.Point3dVector(
                [openstudio.Point3d(*(np.array(c) - origin)) for c in corners])
            sub = openstudio.model.SubSurface(pts, model)
            sub.setSurface(parent)
            sub.setSubSurfaceType("FixedWindow")
            sub.setName(f"plane_{pid}_window_{n_win}_{w['source']}")
            n_win += 1
            by_source[w["source"]] = by_source.get(w["source"], 0) + 1
        print(f"{n_win} window(s) added from {args.windows.name} "
              f"({', '.join(f'{v} {k}' for k, v in by_source.items())})"
              + (f", {n_skipped} skipped (no matching plane)" if n_skipped else ""))

    try:
        print(f"floor area: {space.floorArea():.2f} m2")
    except Exception as e:
        print(f"floor area: n/a ({e})")
    try:
        print(f"volume: {space.volume():.2f} m3 (0 if the solid isn't enclosed -- "
              f"see --cap-open-faces in fit_planes.py)")
    except Exception as e:
        print(f"volume: n/a ({e})")

    model.save(openstudio.path(str(args.out)), True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
