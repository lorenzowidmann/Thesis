clear
clc
close all

%% Parametri
% Il .ply (thermal_voxels.ply) ha SOLO un colore RGB fisso cotto dentro al
% file da voxel_consensus.py (rosso->blu sul percentile 5-95, senza verde,
% nessun valore numerico ne' colorbar): non e' un dato plottabile, e' gia'
% un'immagine, ed e' per questo che sembrava "senza temperatura". Il .csv
% accanto (thermal_voxels.csv) ha invece la temperatura vera per voxel
% (t_mean_c), e da qui si puo' fare un plot con colormap regolabile e
% colorbar in gradi reali.
csvPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map_15cm\thermal_voxels.csv';

% ROI: [Xmin Xmax, Ymin Ymax, Zmin Zmax], in metri. Stringere i numeri per
% isolare un tratto. Estensione reale della mappa: X [0.7 51.5]  Y [-14.5 4.7]  Z [-0.7 5.5]
roi = [10 51.5, ...
       -1.8 2, ...
       -0.7 5.5];

markerSize = 30;
colorBy    = 'temperature';   % 'temperature' (t_mean_c) oppure 'spread' (t_std_c, ripetibilita')

%% Lettura e crop
T = readtable(csvPath);
fprintf('Letti %d voxel\n', height(T));

inROI = T.x >= roi(1) & T.x <= roi(2) & ...
        T.y >= roi(3) & T.y <= roi(4) & ...
        T.z >= roi(5) & T.z <= roi(6);
Troi = T(inROI, :);
fprintf('ROI crop: %d -> %d voxel (rimossi %d, %.1f%%)\n', ...
    height(T), height(Troi), height(T) - height(Troi), ...
    100 * (height(T) - height(Troi)) / height(T));

switch colorBy
    case 'temperature'
        vals = Troi.t_mean_c;
        cmap = hot;
        cbLabel = 'Temperatura corretta media per voxel (°C)';
    case 'spread'
        vals = Troi.t_std_c;
        cmap = parula;
        cbLabel = 'Deviazione standard entro voxel (°C) - ripetibilita'' multi-vista';
    otherwise
        error('colorBy deve essere ''temperature'' o ''spread''');
end

%% Visualizzazione
figure('Name', sprintf('Voxel termici - ROI - %d voxel', height(Troi)), 'Color', 'k');
pcshow([Troi.x Troi.y Troi.z], vals, 'MarkerSize', markerSize);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d voxel nella ROI', height(Troi)), 'Color', 'w', 'Interpreter', 'none');
axis equal;
grid on;
colormap(gca, cmap);
cb = colorbar;
cb.Color = 'w';
cb.Label.String = cbLabel;
cb.Label.Color = 'w';

fprintf('%s: min=%.1f  media=%.1f  max=%.1f\n', colorBy, min(vals), mean(vals), max(vals));

%% Per rigenerare a un'altra dimensione di voxel
% Riusa la temperatura corretta gia' calcolata (correct_session.py, step 4),
% quindi e' un comando veloce (rilegge la bag una volta, niente CLIP/SAM):
%
%   C:\venvs\sensorfusion\Scripts\python.exe voxel_consensus.py --stage thermal ^
%       --session-dir C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate ^
%       --bag C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20 ^
%       --voxel 0.15 ^
%       --corrected-name corrected_temperature_consensus.npy ^
%       --out-dir C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map_15cm
%
% Sotto ~15 cm i voxel scendono sotto l'errore di calibrazione composto
% (~9 cm RMSE fra le due estrinseche LiDAR<->camera, prima della deriva
% SLAM): un voxel piu' fine di quello non descrive piu' struttura reale.
