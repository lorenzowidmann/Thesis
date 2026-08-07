clear
clc
close all

%% Parametri
plyPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map\thermal_voxels.ply';

% ROI: [Xmin Xmax, Ymin Ymax, Zmin Zmax], in metri. Di default e' l'intera
% estensione della nuvola (misurata su questo file), quindi lo script
% funziona subito; stringere i numeri per isolare un tratto.
% Estensione reale: X [0.7 51.5]  Y [-14.5 4.7]  Z [-0.7 5.5]
roi = [0.7 51.5, ...
       -14.5 4.7, ...
       -0.7 5.5];

markerSize = 20;

%% Lettura e crop
ptCloud = pcread(plyPath);
fprintf('Letti %d punti\n', ptCloud.Count);

inIdx = findPointsInROI(ptCloud, roi);
ptCloudROI = select(ptCloud, inIdx);
fprintf('ROI crop: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
    ptCloud.Count, ptCloudROI.Count, ...
    ptCloud.Count - ptCloudROI.Count, ...
    100 * (ptCloud.Count - ptCloudROI.Count) / ptCloud.Count);

%% Visualizzazione
figure('Name', sprintf('Voxel termici - ROI - %d punti', ptCloudROI.Count), 'Color', 'k');
pcshow(ptCloudROI, 'MarkerSize', markerSize);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d voxel nella ROI', ptCloudROI.Count), 'Color', 'w', 'Interpreter', 'none');
axis equal;
grid on;
