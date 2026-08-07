clear
clc
close all

%% Parametri
% voxel_map            -> 20 cm (baseline, generato con voxel_consensus.py
%                          --stage thermal --voxel 0.20)
% voxel_map_15cm        -> 15 cm, stessa temperatura corretta, solo
%                          riaggregata su voxel piu' piccoli (vedi il
%                          comando in fondo a questo file per rigenerare
%                          ad altre dimensioni)
plyPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map_15cm\thermal_voxels.ply';

% ROI: [Xmin Xmax, Ymin Ymax, Zmin Zmax], in metri. Di default e' l'intera
% estensione della nuvola (misurata sul file a 20 cm; a 15 cm l'estensione
% e' la stessa, cambia solo quanti voxel ci sono dentro), quindi lo script
% funziona subito; stringere i numeri per isolare un tratto.
roi = [0.7 51.5, ...
       -1.8 2, ...
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
