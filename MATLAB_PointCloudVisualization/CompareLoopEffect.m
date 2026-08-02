%% Con o senza loop closure, dopo il riallineamento di gravita'?
% Il checkpoint contiene gia' le pose riallineate (sezione 6b).
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
load(fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat'), ...
    'kfPoses', 'nKF', 'kfClouds', 'loopConstraints', 'nLoops', 'kfVoxel', 'mapVoxel');

infoOdom = 100; infoLoop = 500; sigma0 = kfVoxel;

function m = buildMap(poses, clouds, nKF, voxel)
    acc = cell(nKF,1);
    for k = 1:nKF
        acc{k} = pctransform(clouds{k}, poses(k)).Location;
    end
    m = pcdownsample(pointCloud(vertcat(acc{:})), 'gridAverage', voxel);
end

% --- A: solo riallineamento, nessuna ottimizzazione
mapA = buildMap(kfPoses, kfClouds, nKF, mapVoxel);
zA = vertcat(kfPoses.Translation); zA = max(zA(:,3)) - min(zA(:,3));

% --- B: riallineamento + pose graph con loop
pg = poseGraph3D;
for k = 2:nKF
    addRelativePose(pg, tform2measurement(kfPoses(k-1).A \ kfPoses(k).A), ...
        buildInfoVector(infoOdom), k-1, k);
end
for c = 1:nLoops
    i = loopConstraints{c}{1}; j = loopConstraints{c}{2};
    tf = loopConstraints{c}{3}; rmse = loopConstraints{c}{4};
    addRelativePose(pg, tform2measurement(tf.A), ...
        buildInfoVector(infoLoop * (sigma0/max(rmse, sigma0/2))^2), i, j);
end
nodes = nodeEstimates(optimizePoseGraph(pg, 'builtin-trust-region'));
% poseGraph3D ancora il nodo 1 all'origine: senza riportare il risultato nel
% frame di kfPoses(1) si confronterebbero due frame globali ruotati fra loro.
T0 = kfPoses(1).A;
posesB = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF
    Ak = eye(4);
    Ak(1:3,1:3) = quat2rotm(nodes(k,4:7));
    Ak(1:3,4)   = nodes(k,1:3)';
    Ak = T0 * Ak;
    posesB(k) = rigidtform3d(Ak(1:3,1:3), Ak(1:3,4)');
end
nodes = vertcat(posesB.Translation);
mapB = buildMap(posesB, kfClouds, nKF, mapVoxel);
zB = max(nodes(:,3)) - min(nodes(:,3));
fprintf('Spostamento medio dei nodi (A -> B): %.3f m\n\n', ...
    mean(vecnorm(nodes - vertcat(kfPoses.Translation), 2, 2)));

fprintf('%-38s %12s %14s\n', '', 'deriva Z pose', 'span Z mappa');
fprintf('%-38s %10.2f m %12.2f m\n', 'A) solo riallineamento gravita', zA, diff(mapA.ZLimits));
fprintf('%-38s %10.2f m %12.2f m\n', 'B) riallineamento + loop closure', zB, diff(mapB.ZLimits));

function v = buildInfoVector(w)
    M = diag([w w w w w w]); v = zeros(1,21); n = 0;
    for i = 1:6
        for j = i:6, n = n+1; v(n) = M(i,j); end
    end
end
function meas = tform2measurement(A)
    meas = [A(1:3,4)', rotm2quat(A(1:3,1:3))];
end
