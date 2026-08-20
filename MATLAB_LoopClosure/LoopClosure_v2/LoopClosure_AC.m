close all 
clc
clear

%% Ripresa da checkpoint con i 2 loop A-C iniettati
load('C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\loop_closure_checkpoint.mat');
% NESSUNA modifica a loopConstraints qui — deve restare con le 13 voci originali
fprintf('Loop caricati dal checkpoint: %d\n', numel(loopConstraints));
% poi direttamente la sezione "Pose graph" in poi, come prima

bagPath = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\rosbag2_2026_07_30-17_50_45_0.db3";

% Stessi parametri della Sezione 1 di LoopClosure_v2.m
useLoopClosure    = true;
useGravityAlign   = true;
useGravityFactor  = true;
infoOdom          = 100;
infoLoop          = 500;
sigma0Loop        = kfVoxel;
infoGravRP        = 50;
infoGravFree      = 1e-6;
floorTol          = 0.06;   % m, tolleranza planarita' (come Sezione 6 di v2)
floorMaxTilt      = 30;     % gradi, tilt max normale pavimento
floorBand         = 0.30;   % m, spessore fascia bassa di ricerca pavimento
floorFitMin       = 0.5;    % frazione minima di keyframe con fit pavimento valido
infoGravZ         = 5;   % peso su Z quando il vincolo di quota e' attivo (sweep: 5 / 10 / 20 / 50, vedi diagnostica Z pre/post e forma XY)
mapVoxel          = 0.05;
useMapROI         = true;
mapROI            = [-Inf 42, -Inf Inf, -1 4];

%% Sezione 8.5: fit del pavimento per keyframe (misura Z indipendente dalla deriva del pose graph)
% floorDist(k) e' una misura LOCALE (sensore -> pavimento nella nuvola di
% k): non porta dentro la deriva Z accumulata dallo SLAM. kfPoses(k).Z
% invece e' GIA' affetto da quella deriva (e' quello che vogliamo
% correggere). Percio' il target del vincolo si costruisce da floorDist,
% NON da kfPoses: usare poseZ per il gate/target sarebbe circolare (quando
% la deriva Z e' grande, cioe' proprio quando il vincolo serve, lo std
% derivato da poseZ e' grande e il gate si disattiverebbe da solo).
floorDist = nan(nKF, 1);   % distanza con segno origine-sensore -> piano pavimento
for k = 1:nKF
    loc = kfClouds{k}.Location;
    if size(loc,1) < 80, continue; end
    zref = prctile(loc(:,3), 5);
    cand = loc(loc(:,3) > zref - 0.12 & loc(:,3) < zref + floorBand, :);
    if size(cand,1) < 50, continue; end
    try
        [model, inl] = pcfitplane(pointCloud(cand), floorTol, [0 0 1], floorMaxTilt);
        if numel(inl) < 40, continue; end
    catch
        continue
    end
    floorDist(k) = -model.Parameters(4) / norm(model.Parameters(1:3));
end
validF = ~isnan(floorDist);

% Diagnostica informativa (NON usata come gate): quanto e' incoerente,
% PRIMA dell'ottimizzazione, la quota assunta col pavimento misurato.
pNew = vertcat(kfPoses.Translation);
floorWorldZPre = pNew(:,3) - floorDist;
fprintf('Quota pavimento (pre-ottim.): mediana %.3f m, std %.3f m (su %d/%d keyframe)\n', ...
    median(floorWorldZPre(validF)), std(floorWorldZPre(validF)), nnz(validF), nKF);

floorFitRate = nnz(validF) / nKF;
useGravityZ = useGravityAlign && useGravityFactor && floorFitRate >= floorFitMin && validF(1);
if useGravityZ
    fprintf('Pavimento rilevato in %.0f%% dei keyframe: vincolo Z attivato (peso %g)\n', ...
        100*floorFitRate, infoGravZ);
else
    fprintf('Pavimento rilevato in %.0f%% dei keyframe (< %.0f%% richiesto) o nodo 1 senza fit: nessun vincolo Z\n', ...
        100*floorFitRate, 100*floorFitMin);
end

%% Sezione 9: pose graph
pg = poseGraph3D;
infoVecOdom = buildInfoVector(infoOdom);

for k = 2:nKF
    Trel = kfPoses(k-1).A \ kfPoses(k).A;
    addRelativePose(pg, tform2measurement(Trel), infoVecOdom, k-1, k);
end

if ~useLoopClosure
    nLoops = 0;
end

fprintf('Pesi dei vincoli di loop (infoOdom = %d per confronto):\n', infoOdom);
for c = 1:nLoops
    i     = loopConstraints{c}{1};
    j     = loopConstraints{c}{2};
    tform = loopConstraints{c}{3};
    rmse  = loopConstraints{c}{4};
    infoLoopEff = infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
    fprintf('  %d -> %d : rmse %.3f m -> info %.0f\n', i, j, rmse, infoLoopEff);
    infoVecLoopC = buildInfoVector(infoLoopEff);
    addRelativePose(pg, tform2measurement(tform.A), infoVecLoopC, i, j);
end

if useGravityAlign && useGravityFactor
    T0inv = kfPoses(1).A \ eye(4);
    nZCorr = 0;
    for k = 2:nKF
        Ak = T0inv * kfPoses(k).A;
        wZ = infoGravFree;
        if useGravityZ && validF(k)
            Ak(3,4) = floorDist(k) - floorDist(1);   % target Z dal pavimento, non da poseZ (niente deriva)
            wZ = infoGravZ;
            nZCorr = nZCorr + 1;
        end
        addRelativePose(pg, tform2measurement(Ak), ...
            buildInfoVectorAniso(infoGravFree, infoGravRP, wZ), 1, k);
    end
    fprintf('Vincoli di gravita aggiunti: %d (Z da pavimento su %d/%d)\n', nKF-1, nZCorr, nKF-1);
end

fprintf('\nPose graph: %d nodi, %d vincoli (%d loop)\n', pg.NumNodes, pg.NumEdges, nLoops);

%% Sezione 10: ottimizzazione
fprintf('Ottimizzazione...\n');
pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');

%% Sezione 11: ricostruzione mappa
nodesOpt = nodeEstimates(pgOpt);
T0 = kfPoses(1).A;
posesOpt = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF
    n  = nodesOpt(k, :);
    Ak = eye(4);
    Ak(1:3,1:3) = quat2rotm(n(4:7));
    Ak(1:3,4)   = n(1:3)';
    Ak = T0 * Ak;
    posesOpt(k) = rigidtform3d(Ak(1:3,1:3), Ak(1:3,4)');
end
nodesOpt = [vertcat(posesOpt.Translation), ...
            cell2mat(arrayfun(@(p) rotm2quat(p.R), posesOpt, 'UniformOutput', false))];

% Verifica vera del vincolo Z: lo std deve scendere rispetto al pre-ottim.
% (Sezione 8.5), non e' un gate ma il controllo che il pavimento misurato
% e la traiettoria corretta ora concordino.
floorWorldZPost = nodesOpt(:,3) - floorDist;
fprintf('Quota pavimento (post-ottim.): mediana %.3f m, std %.3f m (su %d/%d keyframe)\n', ...
    median(floorWorldZPost(validF)), std(floorWorldZPost(validF)), nnz(validF), nKF);

allXYZ = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, posesOpt(k));
    allXYZ{k} = pcT.Location;
end
pcOpt = pointCloud(vertcat(allXYZ{:}));
pcOpt = pcdownsample(pcOpt, 'gridAverage', mapVoxel);

allXYZraw = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, kfPoses(k));
    allXYZraw{k} = pcT.Location;
end
pcRaw = pointCloud(vertcat(allXYZraw{:}));
pcRaw = pcdownsample(pcRaw, 'gridAverage', mapVoxel);

fprintf('\n--- Estensione della mappa ---\n');
fprintf('  X: %7.2f  %7.2f\n', pcOpt.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcOpt.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcOpt.ZLimits);

if useMapROI
    pcOpt = select(pcOpt, findPointsInROI(pcOpt, mapROI));
    pcRaw = select(pcRaw, findPointsInROI(pcRaw, mapROI));
end

figure('Name', 'Diagnostica Z - baseline 13 loop');
subplot(2,1,1);
plot(nodesOpt(:,1), nodesOpt(:,2), 'g-', 'LineWidth', 1.2);
xlabel('X (m)'); ylabel('Y (m)'); title('Vista dall''alto');
axis equal; grid on;

subplot(2,1,2);
plot(nodesOpt(:,1), nodesOpt(:,3), 'g-', 'LineWidth', 1.2);
xlabel('X (m)'); ylabel('Z (m)'); title('Profilo laterale');
grid on;

fprintf('Z pose: min %.3f, max %.3f, escursione %.3f m\n', ...
    min(nodesOpt(:,3)), max(nodesOpt(:,3)), max(nodesOpt(:,3))-min(nodesOpt(:,3)));

%% Traiettoria: confronto prima/dopo
figure('Name', 'Traiettoria dopo iniezione A-C');
trajRaw = vertcat(kfPoses.Translation);
plot3(trajRaw(:,1), trajRaw(:,2), trajRaw(:,3), 'r-', 'LineWidth', 1.5);
hold on;
plot3(nodesOpt(:,1), nodesOpt(:,2), nodesOpt(:,3), 'g-', 'LineWidth', 1.5);
legend('Prima', 'Dopo (con A-C)', 'Location', 'best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Traiettoria keyframe');
axis equal; grid on;

%% Salvataggio
outFile = fullfile(fileparts(bagPath), 'loop_closed_map_AC.pcd');
pcwrite(pcOpt, outFile, 'Encoding', 'binary');
fprintf('\nMappa salvata in:\n  %s\n', outFile);

%% Funzioni di supporto (tutte in fondo, dopo tutte le istruzioni)
function [tf, rmse] = checkPair2(i, j, kfPoses, kfClouds)
    Arel = kfPoses(i).A \ kfPoses(j).A;
    ArelFlat = Arel; ArelFlat(3,4) = 0;
    best = inf; tf = [];
    for g = {Arel, ArelFlat}
        tfC = pcregistericp(kfClouds{j}, kfClouds{i}, ...
            'InitialTransform', rigidtform3d(g{1}), 'InlierDistance', 6.0, 'MaxIterations', 50);
        [t, ~, r] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
            'InitialTransform', tfC, 'InlierDistance', 1.0);
        if r < best, best = r; tf = t; end
    end
    rmse = best;
end

function v = buildInfoVectorAniso(wFree, wRP, wZ)
    if nargin < 3, wZ = wFree; end
    M = diag([wFree wFree wZ wRP wRP wFree]);
    v = zeros(1, 21); n = 0;
    for i = 1:6, for j = i:6, n = n+1; v(n) = M(i,j); end, end
end

function v = buildInfoVector(w)
    M = diag([w w w w w w]);
    v = zeros(1, 21); n = 0;
    for i = 1:6, for j = i:6, n = n+1; v(n) = M(i,j); end, end
end

function meas = tform2measurement(A)
    R = A(1:3, 1:3); t = A(1:3, 4)'; q = rotm2quat(R);
    meas = [t q];
end