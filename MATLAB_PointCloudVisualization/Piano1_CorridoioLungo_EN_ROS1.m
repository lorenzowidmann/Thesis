%% Accumulated point cloud visualization from a pre-built .pcd map
% Loads an already-accumulated point cloud (e.g. the FAST-LIO-SAM-SC-QN
% loop-closure-corrected map) from a .pcd file and displays it. No bag
% reading involved: the accumulation across frames already happened
% upstream (SLAM pipeline), this script only visualizes the result.

clear
close all
clc

%% 1. Parameters
pcdPath    = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\loop_closed_map_ros1.pcd";

voxelSize  = 0.01;   % m, downsampling of the GEOMETRIC cloud (Figure 1, sec. 8).
                     % No LiDAR<->camera cross-registration constraint applies here
                     % (these are just XYZ coordinates from SLAM): it can stay fine.
                     % The temperature-colored views (sec. 9-10) use a different,
                     % coarser voxel, see voxelSizeTemp below.
markerSize = 20;     % on-screen point size. pcshow's default is small

%% 2. Point cloud loading
pc = pcread(pcdPath);
nRaw = pc.Count;

% Remove invalid returns (NaN/Inf), if any survived the SLAM pipeline
xyz = pc.Location;
xyz = xyz(all(isfinite(xyz), 2), :);
fprintf('Points read: %d, valid: %d (discarded %d)\n', ...
    nRaw, size(xyz,1), nRaw - size(xyz,1));

pc = pointCloud(xyz);

%% 5. Outlier filtering
% Two complementary filters, can be used together or separately.

% --- 5a. Geometric crop (ROI) ---
% Removes everything outside a box. This is the right filter for
% spurious but COHERENT clusters (e.g. a line of points detached from
% the corridor), which statistical denoising does not remove because
% their points are close to each other and therefore not "isolated".
% Set useROI = false to disable it and see the whole cloud.
useROI = true;
roi = [12 Inf, ...    % X min max
       -2 2, ...    % Y min max
       -Inf Inf];       % Z min max, cut above 4 m

if useROI
    inIdx  = findPointsInROI(pc, roi);
    nBefore = pc.Count;
    pc = select(pc, inIdx);
    fprintf('ROI crop: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

% --- 5b. Statistical denoise ---
% Removes points whose average distance from the k nearest neighbors
% deviates from the global mean by more than 'threshold' standard
% deviations. Effective against diffuse scatter and single spurious
% returns.
% Raising threshold = more permissive filter, lowering it = more
% aggressive.
useDenoise   = true;
denoiseK     = 20;    % number of neighbors considered
denoiseThres = 2.0;   % threshold in standard deviations

if useDenoise
    nBefore = pc.Count;
    pc = pcdenoise(pc, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    fprintf('Denoise: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

%% 6. Optional downsampling
% Consecutive FAST-LIO frames overlap heavily, so most points are nearly
% duplicates. The voxel grid reduces the load without losing real
% coverage.
if voxelSize > 0
    pcView = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('After %.0f cm downsampling: %d points (%.1f%% of original)\n', ...
        voxelSize*100, pcView.Count, 100*pcView.Count/pc.Count);
else
    pcView = pc;
end

%% 7. Cloud extent
fprintf('\nExtent [min max] in meters:\n');
fprintf('  X: %7.2f  %7.2f\n', pcView.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcView.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcView.ZLimits);

%% 8. Visualization
figure('Name', sprintf('%d points', pcView.Count), 'Color', 'k');

pcshow(pcView, 'MarkerSize', markerSize);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d points', pcView.Count), 'Color', 'w');
axis equal;
grid on;
colormap(gca, turbo);   % color as a function of Z height

%% 9. FLIR temperature coloring (WHOLE session, multi-pose)
% /cloud_registered is already in the world frame and this frame does
% not change over time, so any point of the accumulated map can be
% reprojected "to where the FLIR was" at a specific pose along the
% session (static frustum check). A single pose only covers a narrow
% stretch of corridor (FLIR FOV ~32x26 degrees): to color the WHOLE
% corridor, the projection is repeated for EVERY triplet in the sync
% manifest and, for each point, the average of the temperatures observed
% by all the poses that saw it is accumulated (stretches seen by several
% consecutive poses are averaged, not overwritten). Same
% extrinsics/intrinsics/z-buffer pipeline as
% ProjectFlirOnZed_Session9.m / FlirZedViewer_Session9.m (see those
% scripts for the Obsidian sources of the parameters).
%
% Uses the same pc (post ROI+denoise, BEFORE downsampling) as the main
% view. Each point's color comes from a LiDAR->camera reprojection
% (T_lidar_to_flir), which has a compound error of ~9 cm RMSE (5.8 cm
% lidar->flir + 6.8 cm lidar->zed, see rig_calibration.yaml, BEFORE SLAM
% drift) on WHICH pixel a given point is actually sampling. A per-point
% value finer than that error is not reliable: it would show
% cross-correspondence noise as if it were structure. For this reason,
% AFTER the temperature computation (multi-pose loop below, unchanged),
% a second averaging pass is done -- this time across NEARBY points, not
% across poses of the same point -- on the voxelSizeTemp grid (15 cm,
% ~1.6x the RMSE), and the stabilized result is broadcast back to every
% point for the fine-grained visualization (sec. 9-10 below).
% voxelSizeTemp is therefore NO LONGER the display density: it is only
% the granularity of the trust value. The displayed density is decided
% by voxelSize (Figure 1/2, points) or voxelSizeCubes (Figure 3, cubes),
% independently.
%
% useCorrectedTemp = true reads the CORRECTED temperature produced by
% RadiometricCalibration/correct_session.py (emissivity + atmosphere,
% with the multi-view consensus materials from voxel_consensus.py
% --stage vote) instead of the sensor's raw apparent temperature. Same
% .npy format (float32, same shape), so the rest of the reprojection
% pipeline does not change: only WHICH file is read per pose changes.
% correct_session.py can write NaN for a segment with no physically
% plausible candidate (see its plausibility retry): these points are
% excluded from the average instead of propagating NaN into the
% accumulator, which would otherwise permanently zero out that point
% even for all subsequent poses that observe it correctly.

useCorrectedTemp = true;    % false = raw apparent temperature (original behavior)
correctedName    = 'corrected_temperature_consensus.npy';
voxelSizeTemp    = 0.15;    % m, granularity of the statistical POOLING (no longer
                             % display): the per-point (multi-pose) temperature is
                             % here also averaged across nearby points to stabilize
                             % it, then the result is broadcast to the fine
                             % points/cubes shown in Figures 2 and 3 (see sec. 9-10).
                             % Do not go below ~0.10-0.12 without also tightening
                             % zBufferTol_m: they are two comparable error sources
                             % that add up, not independent.

sessionRootSlam  = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM';
sessionDirZed    = fullfile(sessionRootSlam, 'ZED', '20260730_161223', 'fullrate');
flirRot180Dir    = fullfile(sessionRootSlam, 'Flir', 'session9_only_rot180');
emissivityMapDir = fullfile(sessionDirZed, 'emissivity_map');
syncManifestPath = fullfile(sessionDirZed, 'sync_manifest.json');
zBufferTol_m     = 0.08;   % see ProjectFlirOnZed_Session9.m
rangeMax_m       = 20;     % coarse prefilter for speed: points beyond this distance from the current pose are not even projected

if useCorrectedTemp
    fprintf('Temperature source: CORRECTED (%s)\n', correctedName);
else
    fprintf('Temperature source: raw apparent (sensor, not corrected)\n');
end

% LiDAR -> FLIR extrinsics, 6 clean poses, min3d (risultato_calibrazione_estrinseca_lidar_flir.md)
R_lidar2flir = [ 0.048992   -0.998798   -0.00174722;
                 0.0621242   0.00479317 -0.998057;
                 0.996865    0.0487882   0.0622843 ];
t_lidar2flir = [-0.107859; -0.0426556; -0.0135286];

% FLIR intrinsics, no-skew model (risultato_calibrazione_intrinseca_flir_vue_pro_r.md)
Kf = [570.4796        0  149.1501;
             0  545.4275  117.0047;
             0         0         1];
kFlir = [-0.4241, -0.1241];
pFlir = [-0.0053,  0.0025];
flirW = 336; flirH = 256;

manifest = jsondecode(fileread(syncManifestPath));
allTriplets = manifest.triplets;
nT = numel(allTriplets);
fprintf('\n--- FLIR temperature coloring, WHOLE session (%d poses) ---\n', nT);

xyzFilt = pc.Location;   % same points (post ROI+denoise) feeding pcView
nPts = size(xyzFilt, 1);
sumTemp = zeros(nPts, 1);
cntTemp = zeros(nPts, 1, 'uint16');

ticId = tic;
for i = 1:nT
    tr = allTriplets(i);

    t_wb = tr.lidar.position(:);
    q_xyzw = tr.lidar.orientation(:)';
    q_wxyz = [q_xyzw(4), q_xyzw(1), q_xyzw(2), q_xyzw(3)];
    R_wb = quat2rotm(q_wxyz);

    % coarse prefilter: only points close enough to the current pose
    d2 = sum((xyzFilt - t_wb').^2, 2);
    nearIdx = find(d2 <= rangeMax_m^2);
    if isempty(nearIdx)
        continue
    end

    ptsBody = (R_wb' * (xyzFilt(nearIdx,:)' - t_wb))';
    ptsFlir = (R_lidar2flir * ptsBody' + t_lidar2flir)';

    [uFlir, vFlir, validFlir] = projectPinholeTemp(ptsFlir, Kf, kFlir, pFlir, flirW, flirH);
    if ~any(validFlir)
        continue
    end
    okFlir = false(size(validFlir));
    okFlir(validFlir) = zBufferMaskTemp(uFlir(validFlir), vFlir(validFlir), ptsFlir(validFlir,3), ...
        flirW, flirH, zBufferTol_m);
    validFlir = validFlir & okFlir;
    if ~any(validFlir)
        continue
    end

    [~, flirStemFull, ~] = fileparts(tr.flir.file);   % e.g. 20250906_233144_R
    if useCorrectedTemp
        npyPath = fullfile(emissivityMapDir, flirStemFull, correctedName);
        if ~isfile(npyPath)
            continue   % frame without correction (classify_session/correct_session not run on this frame)
        end
    else
        flirBase = erase(flirStemFull, '_R');         % the raw .npy has no _R suffix
        npyPath = fullfile(flirRot180Dir, [flirBase '.npy']);
    end
    flirRaw = readNpyFloat32Temp(npyPath);

    uF = round(uFlir(validFlir)); vF = round(vFlir(validFlir));
    linIdx = sub2ind([flirH, flirW], vF, uF);
    vals = double(flirRaw(linIdx));

    % correct_session.py can write NaN where no candidate material gave a
    % physically plausible temperature: that point is excluded from THIS
    % observation only, without corrupting its accumulation for
    % subsequent poses (the raw apparent temperature is never NaN, but
    % the check costs nothing and keeps the code correct in both cases).
    globalIdx = nearIdx(validFlir);
    okVal = isfinite(vals);
    globalIdx = globalIdx(okVal);
    vals = vals(okVal);

    sumTemp(globalIdx) = sumTemp(globalIdx) + vals;
    cntTemp(globalIdx) = cntTemp(globalIdx) + 1;

    if mod(i, 20) == 0 || i == nT
        fprintf('  pose %d/%d, points covered so far: %d, %.0fs elapsed\n', ...
            i, nT, sum(cntTemp > 0), toc(ticId));
    end
end

hasObs = cntTemp > 0;
temperature = nan(nPts, 1);
temperature(hasObs) = sumTemp(hasObs) ./ double(cntTemp(hasObs));

fprintf('Points with valid temperature (>=1 pose): %d / %d (%.1f%%)\n', ...
    sum(hasObs), nPts, 100*sum(hasObs)/nPts);
fprintf('Observations per covered point: mean=%.1f  max=%d\n', ...
    mean(cntTemp(hasObs)), max(cntTemp));

if useCorrectedTemp
    tempLabel = 'CORRECTED temperature (deg C, consensus emissivity + atmosphere)';
else
    tempLabel = 'Temperature (deg C, raw FLIR radiometric data)';
end

% --- Statistical pooling on coarse voxel + broadcast ---
% temperature (above) is already the multi-pose average for a single
% point and must not be touched. Here a SECOND averaging level is added,
% this time across different nearby points that fall in the same
% voxelSizeTemp voxel (15 cm, trust threshold explained in the
% voxelSizeTemp comment above), to further stabilize the value before
% displaying it. Same floor+unique+accumarray idiom already used in sec.
% 10 for the cubes.
%
% The broadcast is valid because floor(xyzFilt/voxelSize) and
% floor(xyzFilt/voxelSizeTemp) are computed on the SAME xyzFilt, same
% origin, no shift: since voxelSizeTemp/voxelSize = 15 is an exact
% integer, every fine voxelSize cell falls entirely inside a single
% coarse voxelSizeTemp cell (coarse boundaries always coincide with fine
% boundaries, never mid-cell). All points of a fine cell therefore share
% by construction the same stabilized value: the gridAverage of Figure 2
% on those values is a color no-op (see below).
validT = isfinite(temperature);
ivTrust = floor(xyzFilt(validT,:) / voxelSizeTemp);
[ivTrustU, ~, icTrust] = unique(ivTrust, 'rows');
meanTrust = accumarray(icTrust, temperature(validT), [], @mean);
nVoxTrust = size(ivTrustU, 1);
fprintf('Statistical pooling voxels (%.0f cm): %d, points pooled: %d\n', ...
    voxelSizeTemp*100, nVoxTrust, sum(validT));

temperatureStable = nan(nPts, 1);
temperatureStable(validT) = meanTrust(icTrust);

% pcTemp uses the ALREADY stabilized color (temperatureStable, pooled
% average over the voxelSizeTemp voxel), but the DISPLAY downsample
% reuses voxelSize (same value as Figure 1, sec. 8): fine density,
% coarse-trust color. Inside every voxelSize cell, all points share by
% construction the same temperatureStable (see note above): the
% gridAverage below averages identical values, so it thins the points
% out as in Figure 1 without altering the color.
pcTemp = pointCloud(xyzFilt, 'Intensity', temperatureStable);
pcViewTemp = pcdownsample(pcTemp, 'gridAverage', voxelSize);

hasTemp = isfinite(pcViewTemp.Intensity);
fprintf('Points shown (%.0f cm voxel) with valid stabilized temperature: %d / %d\n', ...
    voxelSize*100, sum(hasTemp), pcViewTemp.Count);

if any(hasTemp)
    pcViewTempValid = select(pcViewTemp, find(hasTemp));

    figure('Name', sprintf('FLIR temperature - %.0f cm points, %.0f cm stabilized color - %d poses', ...
        voxelSize*100, voxelSizeTemp*100, nT), 'Color', 'k');
    pcshow(pcViewTempValid.Location, pcViewTempValid.Intensity, 'MarkerSize', markerSize);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Points at %.0f cm, color stabilized on %.0f cm voxel - %d poses fused', ...
        voxelSize*100, voxelSizeTemp*100, nT), 'Color', 'w', 'Interpreter', 'none');
    axis equal;
    grid on;
    colormap(gca, hot);
    cb = colorbar;
    cb.Color = 'w';
    cb.Label.String = [tempLabel, sprintf(' - stabilized on %.0f cm voxel', voxelSizeTemp*100)];
    cb.Label.Color = 'w';

    tVals = pcViewTempValid.Intensity;
    fprintf('Stabilized temperature shown: min=%.1f  mean=%.1f  max=%.1f  [degrees, .npy data units]\n', ...
        min(tVals), mean(tVals), max(tVals));
else
    warning('No point with a valid stabilized temperature over the whole session.');
end

%% 10. Voxels as transparent cubes (not just a point at the center)
% Draws every occupied voxel as an actual cube. As with Figure 2 (sec.
% 9), the GEOMETRY (cube size) is now decoupled from the STATISTICAL
% TRUST of the color: each cube takes its color from the average of
% temperatureStable of the points inside it, and temperatureStable has
% already been stabilized on the coarse voxelSizeTemp voxel (15 cm, see
% sec. 9 above) BEFORE arriving here. Adjacent cubes that fall in the
% same voxelSizeTemp voxel therefore legitimately show the same color:
% that is the point of this view, not an artifact.
%
% voxelSizeCubes governs ONLY the geometric density of the cubes shown,
% no longer the noise threshold (that role now belongs to
% voxelSizeTemp, above). The limit that matters here is renderability
% with transparency enabled: at 1 cm the occupied voxels would be ~570k
% and MATLAB chokes; ~40-50k cubes (5 cm) is heavy but feasible; ~10-15k
% cubes (10 cm) is smooth.
%
% Optimization: shared faces between two adjacent occupied voxels are
% discarded (face culling). Needed both for performance and for
% rendering: with transparency enabled, hidden internal faces would
% visually stack up, making everything opaque and confusing.

voxelSizeCubes = 0.05;   % m, cube side for VISUALIZATION (geometric
                          % density). No longer tied to voxelSizeTemp: that
                          % remains the color trust threshold (see above),
                          % this is only a rendering/detail trade-off.
cubeAlpha      = 1;   % 0 = invisible, 1 = opaque
cubeEdges      = false;  % true = draw edges (readable only with few voxels)

fprintf('\n--- Voxels as transparent cubes: %.0f cm geometry, %.0f cm stabilized color ---\n', ...
    voxelSizeCubes*100, voxelSizeTemp*100);

% validT was already computed in sec. 9 (isfinite(temperature)): the
% exact same point set, reused here without recomputation. The only
% difference from before is the SOURCE of the values: tAll now comes
% from temperatureStable (already averaged on the voxelSizeTemp voxel),
% not from the raw per-point temperature -- so every cube shows the
% stabilized color, at the fine geometric density of voxelSizeCubes. A
% cube straddling two voxelSizeTemp voxels (possible only if
% voxelSizeTemp is not an exact multiple of voxelSizeCubes) would simply
% take the average of the already-stabilized values of its points: no
% crash or visual discontinuity, just a local blend between two nearby
% averages.
ivAll = floor(xyzFilt(validT,:) / voxelSizeCubes);      % integer voxel coordinates
tAll  = temperatureStable(validT);

[ivU, ~, ic] = unique(ivAll, 'rows');
meanT = accumarray(ic, tAll, [], @mean);                % average (already stabilized values) per display voxel
nVox = size(ivU, 1);
fprintf('Occupied voxels (%.0f cm geometry, %.0f cm stabilized color): %d\n', ...
    voxelSizeCubes*100, voxelSizeTemp*100, nVox);

% --- geometry: 8 vertices and 6 faces per voxel, vectorized ---
% Local corners of the unit cube, then scaled and translated onto the voxel
cornerOffsets = [0 0 0; 1 0 0; 1 1 0; 0 1 0; ...
                 0 0 1; 1 0 1; 1 1 1; 0 1 1];
% Faces as quads over the 8 local vertices, one row per direction
faceDefs = [1 2 3 4;   % -Z
            5 6 7 8;   % +Z
            1 2 6 5;   % -Y
            4 3 7 8;   % +Y
            1 4 8 5;   % -X
            2 3 7 6];  % +X
% Neighbor direction that hides each face, same order
faceNeighborDir = [0 0 -1; 0 0 1; 0 -1 0; 0 1 0; -1 0 0; 1 0 0];

% Vertices: 8 per voxel (with duplicates between adjacent voxels, acceptable)
vertsAll = zeros(nVox * 8, 3);
for c = 1:8
    vertsAll(c:8:end, :) = (ivU + cornerOffsets(c,:)) * voxelSizeCubes;
end

% Visible faces: discard those facing an occupied neighbor
baseIdx = (0:nVox-1)' * 8;
facesVis = cell(6, 1);
colorVis = cell(6, 1);
for f = 1:6
    hidden = ismember(ivU + faceNeighborDir(f,:), ivU, 'rows');
    keep = ~hidden;
    facesVis{f} = baseIdx(keep) + faceDefs(f,:);
    colorVis{f} = meanT(keep);
end
faces = vertcat(facesVis{:});
faceColors = vertcat(colorVis{:});

fprintf('Total faces: %d, visible after culling: %d (%.0f%% discarded)\n', ...
    nVox*6, size(faces,1), 100*(1 - size(faces,1)/(nVox*6)));

figure('Name', sprintf('Cubes %.0f cm, %.0f cm stabilized color - %d poses', ...
    voxelSizeCubes*100, voxelSizeTemp*100, nT), 'Color', 'k');
if cubeEdges
    edgeArg = {'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.1};
else
    edgeArg = {'EdgeColor', 'none'};
end
patch('Vertices', vertsAll, 'Faces', faces, ...
    'FaceVertexCData', faceColors, 'FaceColor', 'flat', ...
    'FaceAlpha', cubeAlpha, edgeArg{:});

ax = gca;
ax.Color = 'k';
ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Cubes at %.0f cm, color stabilized on %.0f cm voxel (alpha %.2f) - %d poses fused', ...
    voxelSizeCubes*100, voxelSizeTemp*100, cubeAlpha, nT), 'Color', 'w');
axis equal; grid on; view(3);
colormap(ax, hot);
cbc = colorbar;
cbc.Color = 'w';
cbc.Label.String = [tempLabel, sprintf(' - stabilized on %.0f cm voxel', voxelSizeTemp*100)];
cbc.Label.Color = 'w';
camlight headlight; lighting none;   % no shading: the color is the data, not the light

fprintf('Stabilized temperature on cubes (%.0f cm geometry, %.0f cm stabilized color): min=%.1f  mean=%.1f  max=%.1f\n', ...
    voxelSizeCubes*100, voxelSizeTemp*100, min(meanT), mean(meanT), max(meanT));

%% --- Local functions ---

function [u, v, valid] = projectPinholeTemp(P, K, k, p, W, H)
    z = P(:,3);
    valid = z > 0.05;
    xn = P(:,1) ./ z;
    yn = P(:,2) ./ z;
    r2 = xn.^2 + yn.^2;
    radial = 1 + k(1)*r2 + k(2)*r2.^2;
    xd = xn .* radial + 2*p(1)*xn.*yn + p(2)*(r2 + 2*xn.^2);
    yd = yn .* radial + p(1)*(r2 + 2*yn.^2) + 2*p(2)*xn.*yn;
    u = K(1,1)*xd + K(1,3);
    v = K(2,2)*yd + K(2,3);
    valid = valid & u >= 1 & u <= W & v >= 1 & v <= H;
end

function mask = zBufferMaskTemp(u, v, z, W, H, tol)
    if isempty(u)
        mask = false(0,1);
        return
    end
    uu = min(max(round(u), 1), W);
    vv = min(max(round(v), 1), H);
    binIdx = sub2ind([H, W], vv, uu);
    z = double(z);
    minZ = accumarray(binIdx, z, [W*H, 1], @min, Inf);
    mask = z <= minZ(binIdx) + tol;
end

function arr = readNpyFloat32Temp(npyPath)
    fid = fopen(npyPath, 'r');
    if fid < 0
        error('Cannot open %s', npyPath);
    end
    cleanupObj = onCleanup(@() fclose(fid));
    fread(fid, 6, 'uint8=>char');
    fread(fid, 2, 'uint8');
    headerLen = fread(fid, 1, 'uint16');
    headerStr = fread(fid, headerLen, 'uint8=>char')';

    shapeTok = regexp(headerStr, "'shape':\s*\(([^)]*)\)", 'tokens', 'once');
    dims = str2double(strsplit(strtrim(shapeTok{1}), ','));
    dims(isnan(dims)) = [];
    nRows = dims(1); nCols = dims(2);

    if ~contains(headerStr, '<f4')
        error('Unhandled .npy format (expected little-endian float32 ''<f4''): %s', headerStr);
    end
    data = fread(fid, nRows * nCols, 'single=>single');
    arr = reshape(data, [nCols, nRows])';
end
