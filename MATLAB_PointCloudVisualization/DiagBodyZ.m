%% Dove sono davvero i punti in frame body? (per tarare la ricerca pavimento)
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfClouds', 'nKF');

for k = [1 40 80 120]
    loc = kfClouds{k}.Location;
    z = loc(:,3);
    fprintf('kf %3d: n=%6d  z: min %6.2f  p01 %6.2f  p05 %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f\n', ...
        k, numel(z), min(z), prctile(z,1), prctile(z,5), median(z), prctile(z,95), max(z));

    % istogramma grezzo della fascia bassa
    edges = -4:0.25:1;
    h = histcounts(z, edges);
    [~, im] = max(h);
    fprintf('        bin piu popoloso in [-4,1]: [%.2f, %.2f) con %d punti\n', ...
        edges(im), edges(im+1), h(im));
end
