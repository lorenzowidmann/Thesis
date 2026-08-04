%% Visualizzazione point cloud accumulata da bag ROS2
% Legge /cloud_registered da una bag ROS2 (.db3), accumula piu frame in
% un'unica nuvola e la visualizza.
%
% /cloud_registered e' l'output gia' registrato di FAST-LIO: i punti sono
% gia' espressi nel frame mappa, quindi i frame successivi si possono
% concatenare direttamente senza applicare alcuna trasformazione.

clear
close all
clc

%% 1. Parametri
bagPath    = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20';
topicName  = '/cloud_registered';

frameStep  = 1;      % 1 = tutti i frame. Alzare (es. 5) se la memoria non basta.
maxFrames  = Inf;    % limite di frame da leggere, Inf = nessun limite
voxelSize  = 0.1;   % m, dimensione voxel per il downsampling. 0 = disattivato
markerSize = 20;     % dimensione dei punti a schermo. Il default di pcshow e' piccolo

%% 2. Apertura bag
bag = ros2bagreader(bagPath);
disp('Topic disponibili nella bag:');
disp(bag.AvailableTopics);

sel = select(bag, 'Topic', topicName);
nTotal = sel.NumMessages;
fprintf('\nTopic %s: %d messaggi\n', topicName, nTotal);

if nTotal == 0
    error(['Nessun messaggio su %s.\n' ...
        'Controllare il nome del topic nella lista qui sopra.'], topicName);
end

%% 3. Selezione dei frame da leggere
idx = 1:frameStep:nTotal;
if numel(idx) > maxFrames
    idx = idx(1:maxFrames);
end
fprintf('Lettura di %d frame (step %d)\n', numel(idx), frameStep);

msgs = readMessages(sel, idx);

%% 4. Accumulo dei frame in un'unica nuvola
% Preallocazione in cell array e vertcat finale: molto piu' veloce che
% concatenare dentro il ciclo, dove la matrice verrebbe riallocata a ogni giro.
allXYZ = cell(numel(msgs), 1);
for i = 1:numel(msgs)
    allXYZ{i} = rosReadXYZ(msgs{i});
end
xyz = vertcat(allXYZ{:});

nRaw = size(xyz, 1);

% Rimozione dei ritorni non validi (NaN/Inf), presenti quando il raggio
% non trova superficie entro il range del sensore
xyz = xyz(all(isfinite(xyz), 2), :);
fprintf('Punti letti: %d, validi: %d (scartati %d)\n', ...
    nRaw, size(xyz,1), nRaw - size(xyz,1));

pc = pointCloud(xyz);

%% 5. Filtraggio outlier
% Due filtri complementari, si possono usare insieme o separatamente.

% --- 5a. Crop geometrico (ROI) ---
% Rimuove tutto cio' che cade fuori da un box. E' il filtro giusto per
% cluster spuri COERENTI (es. una linea di punti staccata dal corridoio),
% che il denoise statistico non toglie perche' i loro punti sono vicini
% tra loro e quindi non risultano "isolati".
% Mettere useROI = false per disattivarlo e vedere la nuvola intera.
useROI = true;
roi = [11.5 21, ...    % X min max
       -1 1.5, ...    % Y min max
       -Inf 4.0];       % Z min max, taglia sopra i 4 m

if useROI
    inIdx  = findPointsInROI(pc, roi);
    nBefore = pc.Count;
    pc = select(pc, inIdx);
    fprintf('ROI crop: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

% --- 5b. Denoise statistico ---
% Rimuove i punti la cui distanza media dai k vicini si discosta di piu' di
% 'threshold' deviazioni standard dalla media globale. Efficace sullo
% sparpagliamento diffuso e sui ritorni spuri singoli.
% Alzare threshold = filtro piu' permissivo, abbassarlo = piu' aggressivo.
useDenoise   = true;
denoiseK     = 20;    % numero di vicini considerati
denoiseThres = 1.0;   % soglia in deviazioni standard

if useDenoise
    nBefore = pc.Count;
    pc = pcdenoise(pc, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    fprintf('Denoise: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

%% 6. Downsampling opzionale
% I frame consecutivi di FAST-LIO si sovrappongono molto, quindi gran parte
% dei punti sono quasi duplicati. Il voxel grid riduce il peso senza perdere
% copertura reale.
if voxelSize > 0
    pcView = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('Dopo downsampling %.0f cm: %d punti (%.1f%% dell''originale)\n', ...
        voxelSize*100, pcView.Count, 100*pcView.Count/pc.Count);
else
    pcView = pc;
end

%% 7. Estensione della nuvola
fprintf('\nEstensione [min max] in metri:\n');
fprintf('  X: %7.2f  %7.2f\n', pcView.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcView.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcView.ZLimits);

%% 8. Visualizzazione
figure('Name', sprintf('%s - %d frame - %d punti', ...
    topicName, numel(idx), pcView.Count), 'Color', 'k');

pcshow(pcView, 'MarkerSize', markerSize);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d frame accumulati, %d punti', numel(idx), pcView.Count), ...
    'Color', 'w');
axis equal;
grid on;
colormap(gca, turbo);   % colore in funzione della quota Z