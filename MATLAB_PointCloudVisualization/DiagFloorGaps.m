%% Dove sono i keyframe senza pavimento, e perche' il fit fallisce?
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
load(fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat'), ...
    'kfPoses', 'nKF', 'kfClouds');

reason = strings(nKF,1);
nCand  = zeros(nKF,1);
nInl   = zeros(nKF,1);
nPts   = zeros(nKF,1);

for k = 1:nKF
    loc = kfClouds{k}.Location;
    nPts(k) = size(loc,1);
    zmin = min(loc(:,3));
    cand = loc(loc(:,3) < zmin + 0.30, :);
    nCand(k) = size(cand,1);

    if nCand(k) < 50
        reason(k) = "pochi punti bassi";
        continue
    end
    try
        [~, inl] = pcfitplane(pointCloud(cand), 0.06, [0 0 1], 30);
        nInl(k) = numel(inl);
        if nInl(k) < 40
            reason(k) = "piano trovato ma pochi inlier";
        else
            reason(k) = "OK";
        end
    catch
        reason(k) = "nessun piano entro 30 deg";
    end
end

bad = reason ~= "OK";
fprintf('Keyframe senza pavimento: %d su %d\n\n', nnz(bad), nKF);

fprintf('Indici falliti: ');
fprintf('%d ', find(bad));
fprintf('\n\n');

fprintf('Motivi:\n');
u = unique(reason(bad));
for i = 1:numel(u)
    fprintf('  %-32s %d\n', u(i), nnz(reason == u(i)));
end

fprintf('\nDettaglio dei falliti:\n');
fprintf('%5s %8s %8s %8s  %s\n', 'kf', 'nPts', 'nCand', 'nInl', 'motivo');
idx = find(bad);
for i = 1:min(numel(idx), 30)
    k = idx(i);
    fprintf('%5d %8d %8d %8d  %s\n', k, nPts(k), nCand(k), nInl(k), reason(k));
end

% I buchi cadono vicino alla rottura di assetto (kf 66-73)?
fprintf('\nFalliti nella finestra critica kf 60-80: %d\n', nnz(bad(60:80)));
