# SolarShadowVoxel — VOSTOK shadow mask per surface voxel

Self-contained (no cross-import -- the pieces it reuses are copied in, see
"Provenance" below). Two parts:

- **VOSTOK** (`../../../vostok`, i.e. `Thesis/../../vostok` =
  `Desktop/Measurment_v2/ClaudeCode/vostok`, sibling to `Easy3D`) -- built
  from source, see "Building VOSTOK" below. Does the actual raycast/octree
  shadow computation; this folder doesn't reimplement any of that.
- **`solar_shadow_voxel.py`** -- prepares VOSTOK's inputs from
  `AlignedOctree`'s output, runs `vostok.exe`, and parses the result back
  into a per-voxel shadow mask.

## Why

Gets a proper shadow-aware sunlit/occluded mask per surface voxel for
Session 9 at its actual capture time, to later combine with the existing
pvlib/Erbs/Perez irradiance pipeline (`OpenStudioModel/sun_incidence.py`,
`parse_arpav.py`) instead of trusting VOSTOK's own built-in clear-sky
irradiance model (a cruder Linke-turbidity-fixed-to-3 approximation than
what's already validated there). **This script stops at the shadow mask --
it does not multiply by irradiance magnitude.** That combination (irradiance
x mask -> sol-air correction) is a separate follow-up, once this output
exists and has been sanity-checked.

## Building VOSTOK

No C++ toolchain existed on this machine going in. Installed via winget
(confirmed via VOSTOK's own `CMakeLists.txt`/README that it needs g++, not
MSVC -- the CMakeLists uses GCC-only flags like `-std=c++0x -m64` that
`cl.exe` doesn't understand):

```powershell
winget install --id Kitware.CMake -e
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e
```

Then, from a shell with both on PATH (a fresh terminal picks this up
automatically once winget's PATH changes take effect; refresh manually in
an existing session with
`$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`):

```powershell
cd C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\vostok
mkdir build; cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
  "-DCMAKE_EXE_LINKER_FLAGS=-static -static-libgcc -static-libstdc++" `
  "-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs" ..
cmake --build . -j 4
```

Three things needed working around, none of them VOSTOK source bugs:

1. **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`** -- CMake 4.x dropped support for
   `cmake_minimum_required(VERSION 3.0.2)` (what VOSTOK's `CMakeLists.txt`
   declares). This flag tells CMake 4.x "configure it anyway, using policies
   from 3.5 onward" without editing the vendored `CMakeLists.txt`.
2. **`-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs`** -- `main.cpp` uses
   `std::experimental::filesystem` (the pre-C++17 TS API), which even on a
   modern GCC still needs an explicit link against `libstdc++fs` (unlike
   the mainlined `<filesystem>`, folded into libstdc++ itself since GCC 9).
   `CMakeLists.txt` doesn't link it. Using `CMAKE_CXX_STANDARD_LIBRARIES`
   specifically (not `CMAKE_EXE_LINKER_FLAGS`) matters: GNU `ld` needs a
   static library listed *after* the object files that reference its
   symbols, and `CMAKE_CXX_STANDARD_LIBRARIES` is what CMake appends at the
   very end of the link line.
3. **`-static -static-libgcc -static-libstdc++`** -- not required to build,
   but without it the resulting `vostok.exe` depends on `libgomp-1.dll`
   (OpenMP runtime) being on `PATH` at runtime, including whenever this
   folder's script later invokes it via `subprocess`. Static linking makes
   it a single self-contained binary with no MinGW DLL dependencies at all
   -- confirmed by running it with a stripped-down `PATH`.

Verify: `vostok\build\vostok.exe` with no arguments should print
`ERROR: Please specify a .sol project file.` (its usage message).

## Usage

```
C:\venvs\planefit\Scripts\python.exe solar_shadow_voxel.py --north-bearing-deg <measured value>
```

**`--north-bearing-deg` is required, no default -- this is the open
"resolve south wall / spatial mask" item.** It's the true compass bearing
(degrees from North, clockwise) of the *aligned* building frame's +X axis
(`aligned_octree.py`'s "forward", not +Y/"right"). This is deliberately
**not** the same number as `EmissivityCalculation/voxel_solar_ns.py`'s
`--north-offset-deg` (default 193, documented there as "bearing of the SLAM
+Y axis"): that's the bearing of the *raw, pre-alignment* SLAM frame, and
`aligned_octree.py`'s `compute_building_frame` applies its own extra
rotation (a measured ~2.6 deg yaw correction, see `AlignedOctree/README.md`)
on top of it, so the aligned frame's own axis bearing is close to, but not
exactly, that number. Reusing 193 (or 193-90) directly here without
re-deriving it for the *aligned* frame specifically would silently be wrong
by roughly that residual rotation -- hence this script fails loudly instead
of guessing.

All other flags have defaults (site lat/lon, capture time, timezone, etc.
-- see `--help`). See each function's docstring for the pipeline's math
(`north_alignment_rotation`, `assign_surface_voxels`).

### Verified

The full pipeline (transform loading, north-rotation math, surface-voxel
face matching, VOSTOK invocation, shadow-cloud parsing, row-order matching)
was run end-to-end against the real Session 9 data with a placeholder
`--north-bearing-deg 193` (explicitly *not* the correct aligned-frame value,
see above -- this was a mechanics-only check, the numeric result is not
meant to be trusted). Result was physically plausible: 2,966 / 5,014 voxels
matched to a surface, VOSTOK ran in ~11s, the closest available shadow file
was 2 minutes from the 18:12 target (VOSTOK only evaluates sunrise-to-sunset
minutes, not midnight-aligned, so an exact match isn't guaranteed even at a
5-minute step), the position sanity check matched to 0.0005 m, and the
per-face illumination pattern made sense (floor 2.7% lit -- interior,
mostly occluded; roof/ceiling 100% -- open sky, no surrounding buildings
are modeled as shadow points; walls 24-46%, varying by orientation as
expected for a single afternoon sun position). None of that run's output
files are committed -- they used the placeholder bearing and would be
misleading if mistaken for a real result; rerun with the real
`--north-bearing-deg` to get one.

## Output

`shadow_mask.json` (or `--out`): `voxel_index` (row into `voxels.npz`'s
`centers`/`counts`), `center_aligned` (aligned frame, NOT north-rotated --
consistent with the rest of the `AlignedOctree`/`TemperatureToVoxel`
pipeline), `face_id` (0-5, `planes_aligned.json` index), `normal_aligned`
(outward-pointing, aligned frame), `illuminated` (bool). Plus
`shadow_file`/`capture_datetime`/`north_bearing_deg` for provenance.

Also written to `--workdir` (default: this folder): `shadow_points.xyz`,
`query_points.xyz`, `run.sol` (VOSTOK's inputs), and `shadow_clouds/`
(VOSTOK's own output -- one `<year>_<day>_<hour>-<minute>_shadow.txt` per
evaluated minute step across the whole day, sunrise to sunset; only the one
closest to the requested capture time is parsed, the rest are left in place
in case another time of day is wanted later). None of these are meant to be
committed (`.gitignore`) -- they're regenerated by every run.

## Provenance (self-contained copies, adapted where noted)

- `load_merged_cloud`/`read_pointcloud2` in `solar_shadow_voxel.py` --
  copied from `AlignedOctree/fit_closed_planes.py` (same convention: raw,
  unaligned bag points in the SLAM/map frame).
- Everything else (`north_alignment_rotation`, `compose_transform`,
  `face_bounds`, `outward_normal`, `assign_surface_voxels`, the `.sol`
  writer, the VOSTOK runner/parser) is new, written directly against
  VOSTOK's own source (`IrradianceCalc.cpp`, `ProjectConfig.cpp`,
  `main.cpp`, `solpos00.h`) rather than trusting its README prose alone --
  see the module docstring for exactly which source lines back which
  design decision (the +Y=North/+X=East/Z=up convention, the SOLPOS
  timezone sign, the shadow-file row-order guarantee, the `.sol` line
  format).
