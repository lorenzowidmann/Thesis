%% Diagnostica: il vincolo di loop dice davvero qualcosa di diverso
%% dall'odometria in Z, o l'ICP e' finito nel minimo vicino al punto di
%% partenza (degenerazione verticale in corridoio)?
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops');

fprintf('%6s %6s %12s %12s %12s %10s\n', ...
    'i', 'j', 'Zodom rel', 'Zicp rel', 'diff (m)', 'rmse');

for c = 1:nLoops
    i     = loopConstraints{c}{1};
    j     = loopConstraints{c}{2};
    tform = loopConstraints{c}{3};
    rmse  = loopConstraints{c}{4};

    Arel = kfPoses(i).A \ kfPoses(j).A;
    zOdom = Arel(3,4);
    zIcp  = tform.A(3,4);

    fprintf('%6d %6d %12.3f %12.3f %12.3f %10.3f\n', ...
        i, j, zOdom, zIcp, zIcp - zOdom, rmse);
end
