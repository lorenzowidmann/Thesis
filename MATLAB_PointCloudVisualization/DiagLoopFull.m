%% Diagnostica completa: confronto 6-DOF tra misura ICP e stima odometria
% Non basta guardare la sola Z: un errore di rotazione nell'ICP puo'
% restare sotto la soglia Z e comunque distorcere pesantemente il resto
% del pose graph una volta propagato lungo la catena.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops');

fprintf('%6s %6s %10s %10s %10s %10s\n', ...
    'i', 'j', 'transErr', 'rotErr(deg)', 'zIcp', 'rmse');

for c = 1:nLoops
    i     = loopConstraints{c}{1};
    j     = loopConstraints{c}{2};
    tform = loopConstraints{c}{3};
    rmse  = loopConstraints{c}{4};

    Arel = kfPoses(i).A \ kfPoses(j).A;
    Terr = Arel \ tform.A;   % scarto tra misura ICP e stima odometria

    transErr = norm(Terr(1:3, 4));
    rotErr = rad2deg(acos(max(-1, min(1, (trace(Terr(1:3,1:3)) - 1) / 2))));

    fprintf('%6d %6d %10.3f %10.2f %10.3f %10.3f\n', ...
        i, j, transErr, rotErr, tform.A(3,4), rmse);
end
