%% vloam_depth_vo.m
% V-LOAM-style (Zhang & Singh, ICRA 2015) depth-anchored visual odometry
% on Session 7: a real frame-to-frame motion estimator using LiDAR depth
% when available (paper eqs. 2-3) and epipolar-only constraints when not
% (paper eq. 4), NOT a scale correction bolted onto monovslam_test.m's
% output. Produces a trajectory with real metric scale (as opposed to
% monovslam_test.m, which cannot).
%
% This is a faithful-but-practical implementation of paper Sec. V, agreed
% with the user (see the "Visual odometry V-LOAM vera" option): the paper's
% depthmap is a persistent 2D KD-tree in spherical coords with a 3-nearest-
% point planar-patch interpolation; here each TRACKED feature instead
% carries its own single 3D point (in the current frame's camera
% coordinates), assigned/refreshed by the same nearest-projected-LiDAR-
% point-within-a-pixel-radius method already validated in
% lidar_depth_association_test.m, whenever the matched LiDAR scan changes
% (~5Hz -> refresh roughly every 6 frames at 30fps, refreshed every 3rd
% frame here, well inside the paper's own "forget old points" tolerance).
% The paper solves eqs. (2)-(4) jointly via Levenberg-Marquardt with robust
% (residual-threshold) reweighting; same here via lsqnonlin, 2-pass IRLS
% (MAD-based hard reject instead of the paper's linear taper -- documented
% simplification, not a different algorithm).
%
% Pipeline:
%   1) lidar_zed_video_depth_sync.py (same folder) syncs EVERY sampled
%      video frame (stride 3, ~10fps) to its nearest raw /cloud_registered
%      LiDAR scan and dumps the points -- reuses (does not reimplement)
%      sync_manifest.py's nearest_index/load_lidar_poses and
%      lidar_zed_depth_sync.py's PointCloud2 reader / quaternion inversion.
%      Read that script's docstring for the full reuse map and the LiDAR-
%      point-source decision (made with the user, not guessed).
%   2) THIS script tracks ORB keypoints frame-to-frame with KLT
%      (vision.PointTracker), assigns/refreshes LiDAR depth per point,
%      solves each step's 6-DOF motion with lsqnonlin, and chains poses.
%
% Coordinate convention (matches the paper, verified on a synthetic case
% before trusting it on real data -- see conversation/plan): for a step
% k-1 -> k, R_step/T_step satisfy X_cam^k = R_step * X_cam^{k-1} + T_step
% (a world-fixed point's coordinates, expressed in the moving camera
% frame). World pose (camera-to-world) then updates as:
%   R_world_k = R_world_{k-1} * R_step'
%   t_world_k = t_world_{k-1} - R_world_k * T_step
%
% IMPORTANT CAVEATS (same as lidar_depth_association_test.m, still apply):
%   - --lidar-zed-offset UNVERIFIED (assumed 0, shared host clock).
%   - /Odometry (FAST-LIO body/IMU frame) used as the LiDAR/laser frame,
%     assumed negligible offset for this Livox integrated unit.
%   - Steps with zero depth-known correspondences cannot observe
%     translation SCALE that step (epipolar-only, eq. 4 is scale-invariant
%     in T) -- lsqnonlin's warm start just carries the previous scale
%     forward (inertia), not corrected. Logged per-frame, not hidden --
%     see n_depth_known in the TUM header / console summary.
%
% Non modifica alcun file esistente del progetto o della sessione dati.

clear; clc; close all;

%% --- Config ---
scriptDir = fileparts(mfilename('fullpath'));
outDir    = fullfile(scriptDir, 'depth_assoc_out');
pyExe     = 'C:\venvs\sensorfusion\Scripts\python.exe';
pyHelper  = fullfile(scriptDir, 'lidar_zed_video_depth_sync.py');

zedSession = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_155047';
videoFile  = fullfile(zedSession, 'session_right.mp4');

calibJson   = fullfile(outDir, 'rig_calibration_resolved.json');
manifestCsv = fullfile(outDir, 'video_manifest.csv');
pointsCsv   = fullfile(outDir, 'video_lidar_points.csv');

pngOut = fullfile(outDir, 'vloam_trajectory_topview.png');
tumOut = fullfile(outDir, 'vloam_poses_tum.txt');

pxAssocThreshold      = 5.0;   % LiDAR<->keypoint association radius (px);
                                % 3px was too strict in the earlier sweep
                                % (rig_calibration's own lidar_to_zed
                                % reprojection error is ~1.7-2.2px), 5px
                                % gave ~2x the coverage without opening the
                                % door to wrong-surface associations.
minTrackedPoints      = 300;   % reseed KLT tracker below this count
reseedMinSeparationPx = 8;     % skip new ORB detections too close to live tracks
maxSampledFrames      = Inf;   % cap for a quick test run; Inf = full session

%% --- Stage 1: run the Python sync/extraction helper if its outputs are missing ---
needSync = ~(isfile(calibJson) && isfile(manifestCsv) && isfile(pointsCsv));
if needSync
    if ~isfile(pyExe)
        error(['vloam_depth_vo:missingPython', newline, ...
            'venv Python non trovato: %s'], pyExe);
    end
    cmd = sprintf('"%s" "%s"', pyExe, pyHelper);
    fprintf('Output di sync mancanti in %s -- eseguo:\n  %s\n', outDir, cmd);
    status = system(cmd);
    if status ~= 0 || ~(isfile(calibJson) && isfile(manifestCsv) && isfile(pointsCsv))
        error(['vloam_depth_vo:syncFailed', newline, ...
            'lidar_zed_video_depth_sync.py e'' terminato con status %d o non ha scritto ', ...
            'tutti gli output attesi in %s.'], status, outDir);
    end
end

%% --- Load calibration ---
calib = jsondecode(fileread(calibJson));
fprintf('Calibrazione: %s\n', calib.source_rig_calibration);
K3 = calib.zed_K_1920x1080; d = calib.zed_dist;
imageSize = [calib.zed_image_size.height, calib.zed_image_size.width];
intrinsics = cameraIntrinsics([K3.fx K3.fy], [K3.cx K3.cy], imageSize, ...
    RadialDistortion=[d.k1 d.k2 d.k3], TangentialDistortion=[d.p1 d.p2]);
Kmat = [K3.fx 0 K3.cx; 0 K3.fy K3.cy; 0 0 1];
T_lidar_to_zed = calib.T_lidar_to_zed;
identityTform = rigidtform3d(eye(3), [0 0 0]);

% Undistortion lookup, precomputed ONCE on a coarse grid: undistortPoints
% (cameraIntrinsics) runs an iterative per-point solve internally, which
% profiled as the dominant cost (~1s/call, called twice per frame on every
% tracked point) when called fresh every frame -- >15s/frame overall,
% impractical for ~3500 frames. Distortion here is mild (k1=-0.159) and
% spatially smooth, so a griddedInterpolant built from one dense-enough
% grid evaluation is effectively exact at the pixel-threshold scale (5px)
% already used elsewhere, and is O(1) per query afterward instead of an
% iterative solve.
gridStep = 20; % px
[gu, gv] = meshgrid(1:gridStep:imageSize(2), 1:gridStep:imageSize(1));
gridUndist = undistortPoints([gu(:) gv(:)], intrinsics);
undistFu = scatteredInterpolant(gu(:), gv(:), gridUndist(:,1), 'linear', 'linear');
undistFv = scatteredInterpolant(gu(:), gv(:), gridUndist(:,2), 'linear', 'linear');
clear gu gv gridUndist

%% --- Load sync manifest + LiDAR points (grouped for O(1) per-scan lookup) ---
manifest = readtable(manifestCsv);
fprintf('Manifest: %d frame(s) campionati (stride gia'' applicato lato Python)\n', height(manifest));

fprintf('Carico punti LiDAR (%s, puo'' richiedere un minuto) ...\n', pointsCsv);
raw = readmatrix(pointsCsv);
[sOrd, sIdx] = sort(raw(:, 1));
sXYZ = raw(sIdx, 2:4);
edges = find([true; diff(sOrd) ~= 0; true]);
uOrds = sOrd(edges(1:end-1));
starts = edges(1:end-1); ends = edges(2:end) - 1;
ordMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
for gi = 1:numel(uOrds)
    ordMap(uOrds(gi)) = [starts(gi) ends(gi)];
end
fprintf('Caricati %d punti LiDAR su %d scan unici.\n', size(sXYZ, 1), ordMap.Count);
clear raw sOrd sIdx edges starts ends uOrds

%% --- Video ---
vr = VideoReader(videoFile);
fprintf('Video: %dx%d @ %.2ffps\n', vr.Width, vr.Height, vr.FrameRate);

%% --- State ---
tracker = vision.PointTracker('MaxBidirectionalError', 2, 'BlockSize', [31 31]);
trackerInit = false;
prevXY = zeros(0, 2);
depthPt = zeros(0, 3);
hasDepth = false(0, 1);
lastOrdinal = NaN;

Rw = eye(3); tw = zeros(3, 1);
poses = struct('frame_idx', {}, 't', {}, 'R', {}, 'T_world', {}, ...
    'n_depth_known', {}, 'n_depth_unknown', {});

optsLM = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'off', 'MaxIterations', 60);
x0 = zeros(6, 1);

frameCounter = -1;
manifestRowIdx = 0;
tStart = tic;

while hasFrame(vr) && manifestRowIdx < min(height(manifest), maxSampledFrames)
    frameCounter = frameCounter + 1;
    I = readFrame(vr);

    if manifestRowIdx + 1 > height(manifest) || manifest.frame_idx(manifestRowIdx + 1) ~= frameCounter
        continue;  % this video frame wasn't sampled by the Python stride
    end
    manifestRowIdx = manifestRowIdx + 1;
    row = manifest(manifestRowIdx, :);

    Igray = rgb2gray(I);

    if ~trackerInit
        pts0 = detectORBFeatures(Igray);
        curXY = double(pts0.Location);
        initialize(tracker, curXY, Igray);
        trackerInit = true;
        depthPt = nan(size(curXY, 1), 3);
        hasDepth = false(size(curXY, 1), 1);
        poses(end+1) = struct('frame_idx', frameCounter, 't', row.video_epoch, ...
            'R', Rw, 'T_world', tw', 'n_depth_known', 0, 'n_depth_unknown', 0); %#ok<SAGROW>
    else
        [curXY, validTrk] = tracker(Igray);
        curXY = double(curXY(validTrk, :));
        prevXY = prevXY(validTrk, :);
        depthPt = depthPt(validTrk, :);
        hasDepth = hasDepth(validTrk);

        bearingPrev = toBearing(prevXY, undistFu, undistFv, Kmat);
        bearingCur = toBearing(curXY, undistFu, undistFv, Kmat);

        idxKnown = find(hasDepth);
        idxUnknown = find(~hasDepth);

        [R_step, T_step, x0, nUsedKnown, nUsedUnknown] = solveVloamMotion(x0, ...
            depthPt(idxKnown, :), bearingCur(idxKnown, :), ...
            bearingPrev(idxUnknown, :), bearingCur(idxUnknown, :), optsLM);

        Rw = Rw * R_step';
        tw = tw - Rw * T_step;

        if ~isempty(idxKnown)
            depthPt(idxKnown, :) = (R_step * depthPt(idxKnown, :)')' + T_step';
        end

        poses(end+1) = struct('frame_idx', frameCounter, 't', row.video_epoch, ...
            'R', Rw, 'T_world', tw', 'n_depth_known', nUsedKnown, ...
            'n_depth_unknown', nUsedUnknown); %#ok<SAGROW>
    end

    %% --- reseed if too few tracked points ---
    nActive = size(curXY, 1);
    if nActive < minTrackedPoints
        newPts = detectORBFeatures(Igray);
        newXY = double(newPts.Location);
        if nActive > 0 && ~isempty(newXY)
            d2 = (newXY(:,1) - curXY(:,1)').^2 + (newXY(:,2) - curXY(:,2)').^2;
            farEnough = all(d2 > reseedMinSeparationPx^2, 2);
            newXY = newXY(farEnough, :);
        end
        curXY = [curXY; newXY]; %#ok<AGROW>
        depthPt = [depthPt; nan(size(newXY, 1), 3)]; %#ok<AGROW>
        hasDepth = [hasDepth; false(size(newXY, 1), 1)]; %#ok<AGROW>
    end
    setPoints(tracker, curXY);

    %% --- depth refresh when the matched LiDAR scan changes ---
    if row.cloud_ordinal ~= lastOrdinal
        lastOrdinal = row.cloud_ordinal;
        if isKey(ordMap, lastOrdinal)
            rng = ordMap(lastOrdinal);
            pLidar = sXYZ(rng(1):rng(2), :);
            pCamH = [pLidar, ones(size(pLidar, 1), 1)] * T_lidar_to_zed';
            pCam = pCamH(:, 1:3);
            pCam = pCam(pCam(:, 3) > 0, :);
            [uv, validProj] = world2img(pCam, identityTform, intrinsics, ApplyDistortion=true);
            uv = uv(validProj, :); pCam = pCam(validProj, :);
            inb = uv(:,1) >= 1 & uv(:,1) <= imageSize(2) & uv(:,2) >= 1 & uv(:,2) <= imageSize(1);
            uv = uv(inb, :); pCam = pCam(inb, :);

            if ~isempty(uv) && ~isempty(curXY)
                dx = curXY(:,1) - uv(:,1)'; dy = curXY(:,2) - uv(:,2)';
                [minD2, minIdx] = min(dx.^2 + dy.^2, [], 2);
                assoc = minD2 <= pxAssocThreshold^2;
                depthPt(assoc, :) = pCam(minIdx(assoc), :);
                hasDepth(assoc) = true;
            end
        end
    end

    prevXY = curXY;

    if mod(manifestRowIdx, 200) == 0
        fprintf('  frame %d (%d/%d campionati), tracked=%d, elapsed=%.1fs\n', ...
            frameCounter, manifestRowIdx, height(manifest), size(curXY, 1), toc(tStart));
    end
end
fprintf('Loop completato: %d frame campionati processati in %.1fs\n', numel(poses), toc(tStart));

%% --- Assemble trajectory ---
nP = numel(poses);
assert(nP > 1, 'Nessuna posa stimata oltre la prima -- controlla i log sopra.');
X = zeros(nP,1); Y = zeros(nP,1); Z = zeros(nP,1); Q = zeros(nP,4);
tstamps = zeros(nP,1); nKnownArr = zeros(nP,1); nUnknownArr = zeros(nP,1);
for i = 1:nP
    T = poses(i).T_world;
    X(i) = T(1); Y(i) = T(2); Z(i) = T(3);
    qwxyz = rotm2quat(poses(i).R);
    Q(i,:) = [qwxyz(2) qwxyz(3) qwxyz(4) qwxyz(1)];
    tstamps(i) = poses(i).t;
    nKnownArr(i) = poses(i).n_depth_known;
    nUnknownArr(i) = poses(i).n_depth_unknown;
end

nScaleAnchored = sum(nKnownArr > 0);
fprintf('\n=== Riepilogo scala ===\n');
fprintf('  Frame con >=1 corrispondenza a profondita'' nota (scala osservabile quel passo): %d/%d (%.1f%%)\n', ...
    nScaleAnchored, nP - 1, 100 * nScaleAnchored / max(nP - 1, 1));
fprintf('  Frame SENZA profondita'' nota (scala portata avanti per inerzia, eq. 4 sola): %d/%d\n', ...
    (nP - 1) - nScaleAnchored, nP - 1);

%% --- Plot 2D top-view ---
fig = figure('Visible', 'off', 'Color', 'w');
plot(X, Y, '-', 'LineWidth', 1.2); hold on;
scatter(X, Y, 10, nKnownArr, 'filled');
colormap(gca, 'parula'); cb = colorbar; cb.Label.String = 'n. corrispondenze a profondita'' nota';
plot(X(1), Y(1), 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(X(end), Y(end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
axis equal; grid on;
xlabel('X (m, scala metrica da LiDAR)'); ylabel('Y (m, scala metrica da LiDAR)');
title({'Traiettoria V-LOAM (depth-anchored) - vista dall''alto', ...
    sprintf('%d/%d passi scala-ancorati (LiDAR)', nScaleAnchored, nP-1)});
exportgraphics(fig, pngOut, 'Resolution', 200);
close(fig);
fprintf('Plot salvato: %s\n', pngOut);

%% --- TUM output ---
fid = fopen(tumOut, 'w');
fprintf(fid, '# timestamp x y z qx qy qz qw\n');
fprintf(fid, '# vloam_depth_vo.m -- V-LOAM depth-anchored VO, scala metrica da LiDAR (non arbitraria come monovslam_test.m)\n');
fprintf(fid, '# %d/%d passi con >=1 corrispondenza a profondita'' nota; il resto porta avanti la scala per inerzia (eq. 4 sola, epipolare)\n', ...
    nScaleAnchored, nP - 1);
for i = 1:nP
    fprintf(fid, '%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n', ...
        tstamps(i), X(i), Y(i), Z(i), Q(i,1), Q(i,2), Q(i,3), Q(i,4));
end
fclose(fid);
fprintf('TUM salvato: %s\n', tumOut);

fprintf('\nFATTO.\n');

%% ======================================================================= %%
function bearing = toBearing(xy, undistFu, undistFv, Kmat)
% Undistorted-pixel -> unit bearing vector (K^-1, then L2-normalize),
% matching the paper's S X-bar = S X / ||S X|| convention. Undistortion
% via the precomputed grid interpolant (see caller), not a fresh
% undistortPoints solve -- see the comment where undistFu/undistFv are built.
if isempty(xy)
    bearing = zeros(0, 3);
    return;
end
xu = undistFu(xy(:,1), xy(:,2));
yv = undistFv(xy(:,1), xy(:,2));
homog = [xu, yv, ones(size(xy, 1), 1)];
b = (Kmat \ homog')';
bearing = b ./ vecnorm(b, 2, 2);
end

function [R, T, x0Next, nKnown, nUnknown] = solveVloamMotion(x0, Xprev_known, ...
    bearingCur_known, bearingPrev_unknown, bearingCur_unknown, opts)
% 2-pass IRLS (MAD-based hard reject) solve of paper eqs. (2)-(4) via
% lsqnonlin, warm-started from the previous step's solution.
nKnown = size(Xprev_known, 1);
nUnknown = size(bearingPrev_unknown, 1);
if nKnown + nUnknown < 8
    % Not enough constraints this step: carry the previous motion estimate
    % forward unchanged (documented inertia fallback, see header caveats).
    R = rodriguesR(x0(1:3));
    T = x0(4:6);
    x0Next = x0;
    return;
end
w = ones(nKnown + nUnknown, 1);
x = x0;
for pass = 1:2
    wKnown = w(1:nKnown);
    wUnknown = w(nKnown+1:end);
    resFun = @(xx) vloamResidual(xx, Xprev_known, bearingCur_known, ...
        bearingPrev_unknown, bearingCur_unknown, wKnown, wUnknown);
    x = lsqnonlin(resFun, x, [], [], opts);
    if pass == 1
        F = resFun(x);
        if nKnown > 0
            rKnown = sqrt(F(1:nKnown).^2 + F(nKnown+1:2*nKnown).^2);
        else
            rKnown = zeros(0, 1);
        end
        rUnknown = abs(F(2*nKnown+1:end));
        allR = [rKnown; rUnknown];
        medR = median(allR);
        madR = median(abs(allR - medR)) * 1.4826 + eps;
        thresh = medR + 3 * madR;
        w = double([rKnown; rUnknown] <= thresh);
    end
end
R = rodriguesR(x(1:3));
T = x(4:6);
x0Next = x;
end

function R = rodriguesR(phi)
% Axis-angle -> rotation matrix via the closed-form Rodrigues formula,
% with an explicit small-angle Taylor branch -- NOT MATLAB's
% rotvec2mat3d/rotationVectorToMatrix, which was found (empirically, see
% conversation) to snap to exactly eye(3) for ||phi|| below ~1e-5 rad.
% That snap makes the residual's dependence on phi numerically
% discontinuous right at phi=0: lsqnonlin's finite-difference Jacobian
% (default step ~1.5e-8) then measures an all-zero column for the
% rotation unknowns and NEVER moves them -- confirmed as the root cause
% of a real bug (every solved step came out with exactly zero rotation
% across a 3467-frame run, an unmistakable tell since real optimizer
% convergence essentially never lands on an exact bit-for-bit zero).
% This formula stays smooth and correctly first-order-sensitive to phi at
% any scale, including the sub-degree per-step rotations expected here
% (stride-3 video, ~0.1s between processed frames).
phi = phi(:);
theta = norm(phi);
if theta < 1e-8
    a = 1 - theta^2 / 6;        % sin(theta)/theta, Taylor limit
    b = 0.5 - theta^2 / 24;     % (1-cos(theta))/theta^2, Taylor limit
else
    a = sin(theta) / theta;
    b = (1 - cos(theta)) / theta^2;
end
K = [0 -phi(3) phi(2); phi(3) 0 -phi(1); -phi(2) phi(1) 0];
R = eye(3) + a * K + b * (K * K);
end

function F = vloamResidual(x, Xprev_known, bearingCur_known, ...
    bearingPrev_unknown, bearingCur_unknown, wKnown, wUnknown)
% Paper eqs. (2)-(3) (depth-known, 2 residuals/point) and eq. (4)
% (depth-unknown, 1 residual/point), stacked. x = [phi(3); T(3)].
phi = x(1:3); T = x(4:6);
R = rodriguesR(phi);
Tx = T(1); Ty = T(2); Tz = T(3);

if ~isempty(Xprev_known)
    RX = Xprev_known * R';   % RX(i,:) = (R * Xprev_known(i,:)')'
    xk = bearingCur_known(:,1); yk = bearingCur_known(:,2); zk = bearingCur_known(:,3);
    r2 = (zk .* RX(:,1) - xk .* RX(:,3) + zk * Tx - xk * Tz) .* wKnown;
    r3 = (zk .* RX(:,2) - yk .* RX(:,3) + zk * Ty - yk * Tz) .* wKnown;
else
    r2 = zeros(0, 1); r3 = zeros(0, 1);
end

if ~isempty(bearingPrev_unknown)
    Rp = bearingPrev_unknown * R';   % Rp(i,:) = (R * bearingPrev_unknown(i,:)')'
    sx = -Tz .* Rp(:,2) + Ty .* Rp(:,3);
    sy =  Tz .* Rp(:,1) - Tx .* Rp(:,3);
    sz = -Ty .* Rp(:,1) + Tx .* Rp(:,2);
    xu = bearingCur_unknown(:,1); yu = bearingCur_unknown(:,2); zu = bearingCur_unknown(:,3);
    r4 = (xu .* sx + yu .* sy + zu .* sz) .* wUnknown;
else
    r4 = zeros(0, 1);
end

F = [r2; r3; r4];
end
