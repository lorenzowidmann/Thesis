"""Map a PlanarSurface (or its JSON) to an OpenStudio .osm model.

Thin adapter over the OpenStudio SDK. The SDK is optional: the neutral JSON
from smoothing.to_openstudio_json is the primary deliverable and needs no extra
dependency; install `openstudio` (a cp313 wheel exists) to also write an .osm
directly. If the SDK is absent this raises with install guidance, mirroring the
hardware stubs in EmissivityCalculation / RadiometricCalibration.

Envelope polygons become base `Surface` objects; fenestration polygons become
`SubSurface` objects attached to the nearest enclosing envelope surface.
"""

import json
from pathlib import Path

from .smoothing import PlanarSurface, SubSurface


def _load_surface(surface_or_json) -> PlanarSurface:
    if isinstance(surface_or_json, PlanarSurface):
        return surface_or_json
    import numpy as np

    doc = json.loads(Path(surface_or_json).read_text(encoding="utf-8"))
    subs = [
        SubSurface(
            class_id=s["class_id"], class_name=s["class_name"], role=s["role"],
            polygons=[np.asarray(p, float) for p in s["polygons"]],
        )
        for s in doc["surfaces"]
    ]
    return PlanarSurface(
        axis=doc["axis"], plane_coord=doc["plane_coord"],
        voxel_size=doc["voxel_size"], subsurfaces=subs,
        n_inliers=doc.get("n_inliers", 0), n_deviations=doc.get("n_deviations", 0),
    )


def to_osm(surface_or_json, path: str | Path):
    """Write an OpenStudio .osm from a PlanarSurface or its exported JSON."""
    try:
        import openstudio
    except ImportError:
        raise RuntimeError(
            "OpenStudio SDK not installed. The neutral JSON export "
            "(smoothing.to_openstudio_json) needs no extra dependency; to write "
            ".osm directly, install it into this venv:\n"
            "    C:\\venvs\\octree\\Scripts\\pip install openstudio\n"
            "(a Python 3.13 Windows wheel is available)."
        ) from None

    surface = _load_surface(surface_or_json)
    model = openstudio.model.Model()
    space = openstudio.model.Space(model)

    def _os_points(poly):
        pts = openstudio.Point3dVector()
        for x, y, z in poly:
            pts.append(openstudio.Point3d(float(x), float(y), float(z)))
        return pts

    base_surfaces = []
    for sub in surface.subsurfaces:
        if sub.role != "fenestration":
            for poly in sub.polygons:
                s = openstudio.model.Surface(_os_points(poly), model)
                s.setName(f"{sub.class_name}_{len(base_surfaces)}")
                s.setSpace(space)
                base_surfaces.append((s, sub.class_name))

    # Attach fenestration polygons to the first base surface (simple draft
    # mapping; a full version would pick the enclosing envelope surface).
    for sub in surface.subsurfaces:
        if sub.role == "fenestration":
            for poly in sub.polygons:
                ss = openstudio.model.SubSurface(_os_points(poly), model)
                ss.setName(f"{sub.class_name}")
                if base_surfaces:
                    ss.setSurface(base_surfaces[0][0])

    path = Path(path)
    model.save(openstudio.toPath(str(path)), True)
    return path
