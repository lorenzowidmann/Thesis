%% La conca in Z e' reale o e' deriva? Lo dice il piano del pavimento.
%
% Se FAST-LIO e' allineato a gravita' (lo e', l'IMU la osserva), la normale
% del pavimento espressa nel frame mappa deve restare verticale lungo tutto
% il percorso. Se si inclina progressivamente, quella e' deriva in
% roll/pitch: la mappa "ruota" e la traiettoria sembra scendere.
%
% Il pavimento e' un riferimento indipendente dall'odometria, quindi puo'
% arbitrare dove l'odometria da sola non puo'.
%
% NB: la fascia di ricerca e' tarata sui dati (i punti piu' bassi di ogni
% nuvola), non su un'ipotesi a priori sull'altezza del sensore.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'nKF', 'kfClouds');

tilt   = nan(nKF,1);
floorZ = nan(nKF,1);
hSens  = nan(nKF,1);
zPose  = nan(nKF,1);

for k = 1:nKF
    pc  = kfClouds{k};
    loc = pc.Location;
    zPose(k) = kfPoses(k).Translation(3);

    zmin = min(loc(:,3));
    cand = loc(loc(:,3) < zmin + 0.30, :);   % fascia bassa, tarata sui dati
    if size(cand,1) < 50
        continue
    end

    try
        pcCand = pointCloud(cand);
        [model, inl] = pcfitplane(pcCand, 0.06, [0 0 1], 30);
        if numel(inl) < 40
            continue
        end
    catch
        continue
    end

    n = model.Normal(:);
    if n(3) < 0, n = -n; end

    nMap = kfPoses(k).R * n;         % normale nel frame mappa
    if nMap(3) < 0, nMap = -nMap; end

    tilt(k)  = rad2deg(acos(max(-1,min(1,nMap(3)))));
    hSens(k) = abs(model.Parameters(4)) / norm(n);
    floorZ(k)= zPose(k) - hSens(k);
end

ok = ~isnan(tilt);
fprintf('Keyframe con pavimento trovato: %d su %d\n\n', nnz(ok), nKF);

fprintf('%5s %9s %10s %11s %9s\n', 'kf', 'Zpose', 'tilt(deg)', 'floorZmap', 'hSens');
for k = 1:8:nKF
    if ok(k)
        fprintf('%5d %9.3f %10.2f %11.3f %9.3f\n', ...
            k, zPose(k), tilt(k), floorZ(k), hSens(k));
    end
end

fprintf('\n--- Sintesi ---\n');
fprintf('Inclinazione normale pavimento vs verticale mappa: mediana %.2f deg, max %.2f deg\n', ...
    median(tilt(ok)), max(tilt(ok)));
fprintf('Altezza sensore da pavimento: mediana %.3f m, std %.3f m\n', ...
    median(hSens(ok)), std(hSens(ok)));
fprintf('Quota pavimento in frame mappa: min %.2f  max %.2f  escursione %.2f m\n', ...
    min(floorZ(ok)), max(floorZ(ok)), max(floorZ(ok)) - min(floorZ(ok)));

% Correlazione tra quota pavimento e quota traiettoria: se il pavimento
% "segue" la traiettoria, sensore e pavimento scendono insieme.
c = corrcoef(zPose(ok), floorZ(ok));
fprintf('Correlazione Zpose vs quota pavimento: %.3f\n', c(1,2));
