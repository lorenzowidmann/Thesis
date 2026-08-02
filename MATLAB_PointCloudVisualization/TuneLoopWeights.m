%% Tuning rapido dei pesi del pose graph, senza rifare bag/ICP
% Carica il checkpoint salvato da Piano1_CorridoioL.m (kfPoses,
% loopConstraints gia' verificati) e prova diversi infoLoop per vedere
% quale corregge davvero la deriva in Z, senza rileggere il bag ne'
% ricalcolare Scan Context/ICP.

clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');

load(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops', 'nKF', 'kfVoxel');

infoOdom   = 100;
sigma0Loop = kfVoxel;

candidates = [10 100 500 2000 5000 20000 100000];

fprintf('%10s %14s %14s %10s\n', 'infoLoop', 'Z prima (m)', 'Z dopo (m)', 'riduzione %');

zBefore = [];

for infoLoop = candidates
    pg = poseGraph3D;
    infoVecOdom = buildInfoVector(infoOdom);

    for k = 2:nKF
        Trel = kfPoses(k-1).A \ kfPoses(k).A;
        addRelativePose(pg, tform2measurement(Trel), infoVecOdom, k-1, k);
    end

    for c = 1:nLoops
        i     = loopConstraints{c}{1};
        j     = loopConstraints{c}{2};
        tform = loopConstraints{c}{3};
        rmse  = loopConstraints{c}{4};

        infoLoopEff = infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
        infoVecLoopC = buildInfoVector(infoLoopEff);
        addRelativePose(pg, tform2measurement(tform.A), infoVecLoopC, i, j);
    end

    pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');
    nodesOpt = pgOpt.nodes;

    zRaw = vertcat(kfPoses.Translation);
    zRaw = zRaw(:,3);
    zOpt = nodesOpt(:,3);

    spanBefore = max(zRaw) - min(zRaw);
    spanAfter  = max(zOpt) - min(zOpt);
    reduction  = 100 * (spanBefore - spanAfter) / spanBefore;

    fprintf('%10.0f %14.2f %14.2f %9.1f%%\n', infoLoop, spanBefore, spanAfter, reduction);
end

%% Funzioni di supporto (copiate da Piano1_CorridoioL.m)
function v = buildInfoVector(w)
    M = diag([w w w w w w]);
    v = zeros(1, 21);
    n = 0;
    for i = 1:6
        for j = i:6
            n = n + 1;
            v(n) = M(i, j);
        end
    end
end

function meas = tform2measurement(A)
    R = A(1:3, 1:3);
    t = A(1:3, 4)';
    q = rotm2quat(R);
    meas = [t q];
end
