%% Visualizzazione point cloud corretta da fast_lio_sam_sc_qn (result.pcd)
% result.pcd e' gia' la mappa accumulata e corretta dal pose graph (loop
% closure ScanContext + Quatro/Nano-GICP), gia' nel frame mappa: nessuna
% trasformazione o accumulo di frame necessaria, si carica e basta.

clear
close all
clc

%% 1. Parametri
pcdPath    = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\loop_closed_map_ros1.pcd";

voxelSize  = 0.05;   % m, dimensione voxel per il downsampling. 0 = disattivato
markerSize = 10;     % dimensione dei punti a schermo. Il default di pcshow e' piccolo

%% 2. Lettura pcd
pc = pcread(pcdPath);
fprintf('Punti letti: %d\n', pc.Count);

%% 3. Filtraggio outlier
% Due filtri complementari, si possono usare insieme o separatamente.

% --- 3a. Crop geometrico (ROI) ---
% Rimuove tutto cio' che cade fuori da un box. E' il filtro giusto per
% cluster spuri COERENTI (es. una linea di punti staccata dal corridoio),
% che il denoise statistico non toglie perche' i loro punti sono vicini
% tra loro e quindi non risultano "isolati".
% Mettere useROI = false per disattivarlo e vedere la nuvola intera.
useROI = true;
roi = [12 Inf, ...    % X min max
       -0.95 0.5, ...    % Y min max
       -Inf Inf];       % Z min max, taglia sopra i 4 m

if useROI
    inIdx  = findPointsInROI(pc, roi);
    nBefore = pc.Count;
    pc = select(pc, inIdx);
    fprintf('ROI crop: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

% --- 3b. Denoise statistico ---
% Rimuove i punti la cui distanza media dai k vicini si discosta di piu' di
% 'threshold' deviazioni standard dalla media globale. Efficace sullo
% sparpagliamento diffuso e sui ritorni spuri singoli.
% Alzare threshold = filtro piu' permissivo, abbassarlo = piu' aggressivo.
useDenoise   = false;
denoiseK     = 40;    % numero di vicini considerati
denoiseThres = 2;   % soglia in deviazioni standard

if useDenoise
    nBefore = pc.Count;
    pc = pcdenoise(pc, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    fprintf('Denoise: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

%% 4. Downsampling opzionale
% result.pcd e' gia' voxelizzato lato C++ (save_voxel_resolution in
% config.yaml), ma un downsample aggiuntivo qui aiuta comunque a
% velocizzare pcshow se si alza la densita' salvata in futuro.
if voxelSize > 0
    pcView = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('Dopo downsampling %.0f cm: %d punti (%.1f%% dell''originale)\n', ...
        voxelSize*100, pcView.Count, 100*pcView.Count/pc.Count);
else
    pcView = pc;
end

%% 5. Estensione della nuvola
fprintf('\nEstensione [min max] in metri:\n');
fprintf('  X: %7.2f  %7.2f\n', pcView.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcView.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcView.ZLimits);

%% 6. Visualizzazione
figure('Name', sprintf('result.pcd - %d punti', pcView.Count), 'Color', 'k');

pcshow(pcView, 'MarkerSize', markerSize);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('SC-QN corrected map — %d punti', pcView.Count), 'Color', 'w');
axis equal;
grid on;
colormap(gca, turbo);   % colore in funzione della quota Z
