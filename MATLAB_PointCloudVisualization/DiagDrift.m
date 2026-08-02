%% Dove sta davvero la deriva in Z, e i loop la coprono?
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops', 'nKF', 'kfClouds');

xyz = vertcat(kfPoses.Translation);

fprintf('=== Profilo Z della traiettoria (pose) ===\n');
fprintf('Z pose: min %.2f  max %.2f  escursione %.2f m\n', ...
    min(xyz(:,3)), max(xyz(:,3)), max(xyz(:,3)) - min(xyz(:,3)));

fprintf('\nZ dei keyframe ogni 10:\n');
for k = 1:10:nKF
    fprintf('  kf %3d : Z = %7.3f\n', k, xyz(k,3));
end

% Copertura dei loop
fprintf('\n=== Copertura dei loop ===\n');
covered = false(nKF,1);
for c = 1:nLoops
    i = loopConstraints{c}{1};
    j = loopConstraints{c}{2};
    covered(i:j) = true;
    fprintf('  loop %3d -> %3d\n', i, j);
end
fprintf('Keyframe dentro almeno un arco di loop: %d su %d\n', nnz(covered), nKF);
firstUncov = find(~covered, 1);
fprintf('Primo keyframe NON coperto: %d\n', firstUncov);

% Escursione Z delle SINGOLE nuvole: se una singola scansione (che dura
% una frazione di secondo) copre gia' 14 m in Z, allora l'escursione della
% mappa non e' deriva della traiettoria ma estensione reale/rumore dei punti.
fprintf('\n=== Escursione Z delle singole nuvole keyframe (frame body) ===\n');
zs = zeros(nKF,1);
for k = 1:nKF
    zl = kfClouds{k}.ZLimits;
    zs(k) = zl(2) - zl(1);
end
fprintf('Escursione Z per nuvola: mediana %.2f m, max %.2f m (kf %d)\n', ...
    median(zs), max(zs), find(zs == max(zs), 1));
