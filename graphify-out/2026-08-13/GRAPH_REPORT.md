# Graph Report - Thesis  (2026-08-09)

## Corpus Check
- 126 files · ~509,678 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1031 nodes · 1724 edges · 60 communities (58 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 90 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8d51df86`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- detect_board_poses.py
- __init__.py
- smoothing.py
- main
- _CdrReader
- flir_intrinsic_calib.py
- viewer.py
- diagnose_flir_calib.py
- SharedZedSource
- zed_intrinsic_calib.py
- _merge_adjacent_rectangles
- sync_manifest.py
- __init__.py
- target_hole_analysis.py
- Calibration
- sources.py
- check_bag_rate.py
- main.py
- octree.py
- __init__.py
- export_livox_cloud.py
- zed_frame_publisher.py
- zed_record.py
- capture_zed_right.py
- generate_board_template.py
- smooth_surface
- ThermalData.py
- flir_frame_publisher.py
- PlanarSurface
- ZedSource
- PointCloudView
- sensors.py
- `zed_record.py` — ZED 2i recorder
- Modules
- flir_frame_publisher
- LidarDistance
- OcTree — point-cloud octree voxel sampling + GUI
- Radiometric Calibration
- CameraServer
- Emissivity Calculation
- zed_frame_publisher
- extract_planes
- drive_view.py
- DriveView
- view_pcd.py
- convert_livox_bag.py
- split_flir_poses.py
- .classify
- extract_sample.py
- recompute_pose_windows.py
- make_demo_data.py
- main
- merge_planar_surface
- camera_server.py
- ffprobe_frame_count

## God Nodes (most connected - your core abstractions)
1. `VoxelGrid` - 25 edges
2. `_smooth_surface_ransac()` - 21 edges
3. `project_axis_aligned()` - 20 edges
4. `_CdrReader` - 17 edges
5. `main()` - 16 edges
6. `smooth_surface()` - 16 edges
7. `_smooth_surface_core()` - 16 edges
8. `main()` - 15 edges
9. `merge_planar_surface()` - 14 edges
10. `render_screenshot()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `flir_fov_bbox_in_zed()`  [INFERRED]
  EmissivityCalculation/classify_session.py → Calibration/projection.py
- `main()` --calls--> `load_rig_calibration()`  [INFERRED]
  EmissivityCalculation/classify_session.py → Calibration/rig_calibration.py
- `main()` --calls--> `WebcamSource`  [INFERRED]
  CameraServer/camera_server.py → EmissivityCalculation/emissivity/sources.py
- `main()` --calls--> `SharedZedSource`  [INFERRED]
  DriveView/drive_view.py → CameraServer/shared_frame.py
- `main()` --calls--> `SharedZedSource`  [INFERRED]
  EmissivityCalculation/main.py → CameraServer/shared_frame.py

## Import Cycles
- None detected.

## Communities (60 total, 2 thin omitted)

### Community 0 - "detect_board_poses.py"
Cohesion: 0.06
Nodes (51): build_time_index(), collect_images(), compare_sensors(), _custom_msg_xyz(), extract_index(), fmt_clock(), frame_difference(), image_profile() (+43 more)

### Community 1 - "__init__.py"
Cohesion: 0.06
Nodes (37): build_arm_data(), build_frame(), crc16(), _ip_bytes(), _kv(), LivoxController, Livox SDK2 control channel -- arm the Mid-360 so it streams point cloud.  The po, Program the point-cloud destination and put the sensor into SAMPLING.          R (+29 more)

### Community 2 - "smoothing.py"
Cohesion: 0.10
Nodes (41): _any_perpendicular(), _axis_aligned_basis(), _axis_target(), _drop_small_components(), fill_enclosed_cells(), _footprint_cells(), _greedy_rectangles(), _label_components() (+33 more)

### Community 3 - "main"
Cohesion: 0.07
Nodes (36): candidate_list(), main(), parse_args(), Batch radiometric correction over a whole synced session, with a physical-plaus, Materials to try for one segment, best first.      The accepted call comes fir, load_scalar_or_map(), main(), parse_args() (+28 more)

### Community 4 - "_CdrReader"
Cohesion: 0.11
Nodes (23): _extract_labels(), load_las(), PointCloud, ndarray, Path, Load a TUM-FACADE .las point cloud into numpy arrays.  Returns the XYZ coordin, Best-effort per-point class id from a laspy LasData object., Load a .las file into a PointCloud (points in the file's own CRS). (+15 more)

### Community 5 - "flir_intrinsic_calib.py"
Cohesion: 0.10
Nodes (31): _apply_clahe(), calibrate(), CalibrationResult, checkerboard_object_points(), detect_all(), detect_corners(), find_images(), main() (+23 more)

### Community 6 - "viewer.py"
Cohesion: 0.11
Nodes (28): class_name(), _color_lut(), colorize(), ndarray, TUM-FACADE semantic classes: id -> name and id -> RGB color.  Class ids and name, (MAX_CLASS_ID+1, 3) float array mapping class id -> RGB., Map an (N,) array of class ids to an (N,3) float RGB array., _add_points() (+20 more)

### Community 7 - "diagnose_flir_calib.py"
Cohesion: 0.13
Nodes (31): _aggregate(), Fit, fit_model(), _fit_subset(), _fmt_dist(), FoldStats, _initial_guess(), main() (+23 more)

### Community 8 - "SharedZedSource"
Cohesion: 0.12
Nodes (13): _cleanup_stale(), FrameReader, FrameWriter, _now_ms(), ndarray, Seqlock-protected shared-memory frame buffer: one writer, many readers.  Windows, Return the latest published frame, retrying on a torn read., Drop-in replacement for ZedUvcSource when another process (camera_server     .py (+5 more)

### Community 9 - "zed_intrinsic_calib.py"
Cohesion: 0.12
Nodes (25): calibrate(), CalibrationResult, checkerboard_object_points(), detect_all(), detect_corners(), find_images(), main(), parse_args() (+17 more)

### Community 10 - "_merge_adjacent_rectangles"
Cohesion: 0.16
Nodes (17): cluster_labels(), declutter(), main(), Open and view a point cloud from a rosbag with PyVista.  Usage:     python view_, Remove disconnected islands far from the main body.      - keep_dist > 0: keep t, Keep one point per voxel of edge `size` (metres)., Return the centre of every occupied voxel of edge `size` (metres)., Build one cube mesh of edge `size` at each voxel centre. (+9 more)

### Community 11 - "sync_manifest.py"
Cohesion: 0.12
Nodes (27): build_triplets(), compute_offset(), flir_timestamp(), list_flir_frames(), load_lidar_poses(), load_zed_frames(), main(), nearest_index() (+19 more)

### Community 12 - "__init__.py"
Cohesion: 0.17
Nodes (20): main(), Provenance for one output rectangle, parallel to a SubSurface's `polygons`., RectMergeInfo, cube_origin(), filter_by_count(), _grid_from_index(), ndarray, Fast voxel sampling of a point cloud (numpy).  Each point is mapped to an inte (+12 more)

### Community 13 - "target_hole_analysis.py"
Cohesion: 0.14
Nodes (21): ArgumentParser, build_parser(), _count_messages(), detect_holes(), knn_spacing(), main(), _make_plot(), match_centres() (+13 more)

### Community 14 - "Calibration"
Cohesion: 0.10
Nodes (20): 1. Capture the board, 2. Solve the intrinsics, 3. Hand off to LVT2Calib, Bag rate check — `check_bag_rate.py`, Calibration, Correspondence, Detection, Getting the board numbers right (+12 more)

### Community 15 - "sources.py"
Cohesion: 0.12
Nodes (12): ABC, FrameSource, ImageSource, _open_capture(), ndarray, Path, Frame sources: still image, webcam, and ZED 2i stereo camera.  All sources retur, ZED 2i single-eye RGB frames via plain UVC (OpenCV), no ZED SDK/GPU needed. (+4 more)

### Community 16 - "check_bag_rate.py"
Cohesion: 0.18
Nodes (18): format_row(), main(), parse_args(), parse_window(), ndarray, Path, rate_stats(), RateStats (+10 more)

### Community 17 - "main.py"
Cohesion: 0.17
Nodes (17): classify_frame(), classify_grid(), crop_roi(), default_center_roi(), draw_grid_overlay(), grid_boxes(), main(), parse_args() (+9 more)

### Community 18 - "octree.py"
Cohesion: 0.18
Nodes (16): parse_args(), print_info(), OcTree — sample a TUM-FACADE point cloud into voxels and visualize it.  Loads, Occupied octree leaves at depth d == voxelizer voxels at the matching size., selftest(), build_octree(), _cube_root(), leaf_voxels() (+8 more)

### Community 19 - "__init__.py"
Cohesion: 0.06
Nodes (40): apply_low_emissivity_gate(), draw_overlay(), load_zed_frames_dir(), main(), parse_args(), ndarray, Path, Material + emissivity per region, for every frame of a recorded ZED session (dr (+32 more)

### Community 20 - "export_livox_cloud.py"
Cohesion: 0.24
Nodes (14): _align(), main(), parse_args(), parse_custom_msg(), ndarray, Path, Export raw Livox `CustomMsg` scans from a ROS 2 bag, for MATLAB or CloudCompare., One point per occupied voxel: the first seen, not the cell average.      Cheaper (+6 more)

### Community 21 - "zed_frame_publisher.py"
Cohesion: 0.18
Nodes (14): companion_camera_info_topic(), gen_frames(), gen_mp4(), load_camera_info(), load_metadata(), main(), parse_iso_epoch(), publish_pass() (+6 more)

### Community 22 - "zed_record.py"
Cohesion: 0.24
Nodes (14): build_metadata(), main(), parse_args(), prepare_session(), Record-only capture utility for the ZED 2i (SVO2 + mp4 + still frames).  Two cap, ISO-8601 UTC timestamp with microsecond resolution     (e.g. 2026-07-28T14:03:11, Provenance-style session sidecar, matching this codebase's flat     json.dumps(d, Create the timestamped session folder and resolve output paths shared by     bot (+6 more)

### Community 23 - "capture_zed_right.py"
Cohesion: 0.22
Nodes (13): capture_loop(), main(), next_index(), open_camera(), parse_args(), ndarray, Path, Grab ZED 2i right-eye frames on a keypress, to feed zed_intrinsic_calib.py.  Ste (+5 more)

### Community 24 - "generate_board_template.py"
Cohesion: 0.21
Nodes (13): hole_centers(), main(), Path, Genera i template PCD per LVT2Calib con la geometria della board reale.  Board:, Scrive un PCD ASCII con campi x y z intensity range (5 campi),     stesso schema, I 4 centri, ai vertici di un quadrato di lato 2*offset., Punti lungo il perimetro del rettangolo, passo `spacing`., Punti lungo la circonferenza, passo `spacing` misurato sull'arco. (+5 more)

### Community 25 - "smooth_surface"
Cohesion: 0.14
Nodes (13): PlaneAnchor, principal_yaw(), A previously computed RANSAC plane, reusable to keep a surface put.      Pass th, Capture the plane a RANSAC-fitted surface was built on, or None (legacy fit / no, Flatten `grid` onto a single plane, preserving class zoning.      offset_method:, Dominant horizontal direction of the voxels, in degrees (0-180).      PCA on the, Rotate `grid`'s centers by -yaw_deg about their horizontal centroid.      Keeps, Inverse of _rotate_grid_horizontal's rotation, for (..., 2) xy arrays. (+5 more)

### Community 26 - "ThermalData.py"
Cohesion: 0.25
Nodes (13): consensus_temperature(), list_session_frames(), main(), parse_args(), ndarray, Path, Read apparent-temperature data out of recorded FLIR radiometric JPEGs.  The ther, Radiometric JPEGs in a session folder, in capture order. (+5 more)

### Community 27 - "flir_frame_publisher.py"
Cohesion: 0.23
Nodes (11): companion_camera_info_topic(), exif_epoch(), list_images(), load_camera_info(), load_embedded(), load_raw(), main(), CameraInfo from a ROS camera_calibration-style YAML (optional). (+3 more)

### Community 28 - "PlanarSurface"
Cohesion: 0.23
Nodes (10): _load_surface(), Path, Map a PlanarSurface (or its JSON) to an OpenStudio .osm model.  Thin adapter ove, Write an OpenStudio .osm from a PlanarSurface or its exported JSON., to_osm(), PlanarSurface, Path, Write the planar surface as OpenStudio-friendly JSON (polygons + roles).      Sc (+2 more)

### Community 29 - "ZedSource"
Cohesion: 0.08
Nodes (40): flir_fov_bbox_in_zed(), project_lidar_to_camera(), ndarray, quat_to_rotation_matrix(), Project LiDAR points into a camera's pixel space.  Generic: works for either F, xyzw quaternion (ROS convention, as stored in sync_manifest.json's     triplet[, Undo the SLAM pose (lidar-local -> world) to bring world points back     into t, Project world-frame LiDAR points into one camera's pixel space.      Returns: (+32 more)

### Community 30 - "PointCloudView"
Cohesion: 0.18
Nodes (10): Appearance, Examples, Flags, How declutter works, How SOR works, PointCloudView, Processing pipeline, Requirements (+2 more)

### Community 31 - "sensors.py"
Cohesion: 0.18
Nodes (7): HygrometerSource, LidarSource, Hardware input stubs for the field setup (not yet available on this PC).  Mirror, Apparent-temperature map from the thermal camera (model t.b.d.)., Per-pixel distance map from the LiDAR, projected onto the thermal image., Relative humidity and air temperature from the weather sensor., ThermalCameraSource

### Community 32 - "`zed_record.py` — ZED 2i recorder"
Cohesion: 0.18
Nodes (10): Errors, Feeding EmissivityCalculation, Output, Sensor Fusion, Setup, Structure, `sync_manifest.py` — FLIR/ZED/LiDAR triplet manifest, Usage (+2 more)

### Community 33 - "Modules"
Cohesion: 0.20
Nodes (9): 1. `EmissivityCalculation/` — what material am I looking at?, 2. `RadiometricCalibration/` — from apparent to true temperature, 3. `PointCloudElaboration/OcTree/` — point cloud → planar building surfaces, 4. `DriveView/` — live view from the ZED 2i's first lens, 5. `CameraServer/` — shared access to one physical camera, Current status, How the modules connect, Modules (+1 more)

### Community 34 - "flir_frame_publisher"
Cohesion: 0.22
Nodes (8): Deploy into the running container, Empty test, Flags, flir_frame_publisher, Image modes (`--image-mode`), Run + detection, What to confirm on real data, Why these defaults (verified against lvt2calib source)

### Community 35 - "LidarDistance"
Cohesion: 0.25
Nodes (7): How it works, LidarDistance, Options, Output, Requirements, The central square, Usage

### Community 36 - "OcTree — point-cloud octree voxel sampling + GUI"
Cohesion: 0.25
Nodes (7): Data, How it works, Next steps (not in this draft), OcTree — point-cloud octree voxel sampling + GUI, Setup, Structure, Usage

### Community 37 - "Radiometric Calibration"
Cohesion: 0.25
Nodes (7): Hardware (later), Physics, Radiometric Calibration, Setup, Structure, Synchronization (design note — not yet implemented), Usage

### Community 38 - "CameraServer"
Cohesion: 0.29
Nodes (6): Auto-close when idle, CameraServer, How it works, Structure, Usage, When you don't need this

### Community 39 - "Emissivity Calculation"
Cohesion: 0.25
Nodes (7): Adding materials, Emissivity Calculation, Session pipeline (real FLIR pixels, via LiDAR fusion), Setup, Structure, Usage, ZED 2i camera

### Community 40 - "zed_frame_publisher"
Cohesion: 0.29
Nodes (6): Deploy into the running container (no mount / image rebuild), Empty test (no target in scene), Flags, Run, Why these defaults (verified against lvt2calib source), zed_frame_publisher

### Community 41 - "extract_planes"
Cohesion: 0.33
Nodes (7): extract_planes(), fit_plane_ransac(), Plane, Fit the dominant plane by RANSAC (inlier count) or MSAC (truncated L2).      `th, Iteratively RANSAC-fit planes: fit, strip inliers, repeat.      Returns up to `m, Pick the plane whose normal best matches a *stable* target direction.      Withi, _select_plane_for_axis()

### Community 42 - "drive_view.py"
Cohesion: 0.40
Nodes (5): _load_module(), main(), parse_args(), Namespace, Path

### Community 43 - "DriveView"
Cohesion: 0.40
Nodes (4): DriveView, Options, Structure, Usage

### Community 44 - "view_pcd.py"
Cohesion: 0.50
Nodes (4): main(), Path, Visualizzatore rapido per file .pcd (formato ASCII), senza dipendenze PCL. Parsa, read_pcd_ascii()

### Community 45 - "convert_livox_bag.py"
Cohesion: 0.50
Nodes (4): main(), Converte un bag ROS2 contenente livox_ros_driver2/msg/CustomMsg in formato ROS1., Registra CustomPoint e CustomMsg nel typestore sotto il package `pkg`., register_livox_types()

### Community 46 - "split_flir_poses.py"
Cohesion: 0.60
Nodes (4): file_seconds(), hhmmss(), main(), Estrae HHMMSS dal nome file e lo converte in secondi. None se non combacia.

### Community 48 - ".classify"
Cohesion: 0.07
Nodes (39): canonical_normal_offset(), close_geometry(), cluster_labels(), crop_roi(), declutter(), dedupe_planes(), fit_oriented_rect(), largest_cluster_mask() (+31 more)

### Community 49 - "extract_sample.py"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Extract a single .las point cloud from a TUM-FACADE .7z archive.  The dataset sh

### Community 50 - "recompute_pose_windows.py"
Cohesion: 0.83
Nodes (3): file_seconds(), hhmmss(), main()

### Community 57 - "merge_planar_surface"
Cohesion: 0.22
Nodes (8): ClassMergeStats, merge_planar_surface(), MergeSummary, _quad(), Planar quad (4,3) for in-plane cell-corner spans [u0,u1] x [v0,v1] (world axes)., Per-class diagnostic counts from one merge_planar_surface() call., Diagnostics for one merge_planar_surface() call, one entry per class., Merge touching same-class rectangles across an already-smoothed surface.      Po

### Community 58 - "camera_server.py"
Cohesion: 0.29
Nodes (7): _load_module(), main(), parse_args(), Namespace, Path, find_v4l2_capture_index(), Linux: return the integer index N of the /dev/videoN node that     advertises th

### Community 59 - "ffprobe_frame_count"
Cohesion: 0.60
Nodes (5): extract_frames(), ffprobe_frame_count(), main(), Path, Exact decoded frame count (ffprobe -count_frames), not the container's     (some

## Knowledge Gaps
- **82 isolated node(s):** `Setup`, `Options`, `Shooting a good set`, `Options`, `Getting the board numbers right` (+77 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EmissivityTable` connect `__init__.py` to `main.py`, `main`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `__init__.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `VoxelGrid` (e.g. with `ClassMergeStats` and `MergeSummary`) actually correct?**
  _`VoxelGrid` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Setup`, `Options`, `Shooting a good set` to the rest of the system?**
  _82 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `detect_board_poses.py` be split into smaller, more focused modules?**
  _Cohesion score 0.059506531204644414 - nodes in this community are weakly interconnected._
- **Should `__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0641025641025641 - nodes in this community are weakly interconnected._
- **Should `smoothing.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10104529616724739 - nodes in this community are weakly interconnected._