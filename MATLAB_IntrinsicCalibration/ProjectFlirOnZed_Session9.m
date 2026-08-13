%% Project FLIR onto ZED — sessione 9 (Piano1_CorridoioLungo), overlay di verifica
%
% Per un pose scelto lungo la sessione 9 (triplet sincronizzato FLIR/ZED/LiDAR),
% proietta la nuvola LiDAR di quell'istante due volte - una in FLIR, una in ZED -
% usando le stesse intrinseche/estrinseche gia' calibrate, poi disegna sull'
% immagine ZED un punto colorato con il valore FLIR campionato nel punto
% corrispondente. Stessa logica a due proiezioni descritta nella nota
% flusso_fusione_sensoriale_zed_flir_voxel (Obsidian), qui usata per un controllo
% visivo di sovrapposizione, non per la fusione radiometrica completa.
%
% Fonti dei parametri (Obsidian ThesisMD/ClaudeNotes):
%   - risultato_sync_fine_sessione9_fullrate.md               -> triplet FLIR/ZED/LiDAR
%   - risultato_calibrazione_estrinseca_lidar_flir.md          -> Tr_laser_to_cam (6 pose, min3d)
%   - risultato_calibrazione_estrinseca_lidar_zed.md           -> Tr_laser_to_cam (8 pose, min3d)
%   - risultato_calibrazione_intrinseca_flir_vue_pro_r.md      -> K FLIR, modello senza skew
%   - risultato_calibrazione_intrinseca_zed_2i_1080p.md        -> K ZED right eye 1080p, senza skew
%
% Convenzione rotazione FLIR (IMPORTANTE, vedi risultato_calibrazione_estrinseca_lidar_flir.md
% punto 1): la termocamera e' montata capovolta sul rover; per l'estrinseca le
% immagini FLIR sono state ruotate di 180 gradi PRIMA della detection, e la K
% originale (fit su immagini NON ruotate) e' stata applicata cosi' com'e', senza
% ricentrare cx/cy sulla nuova griglia. Questo script replica esattamente la
% stessa convenzione (stessa cartella *_rot180*, stessa K non ricentrata), perche'
% e' quella con cui Tr_laser_to_cam e' stata effettivamente stimata: ricentrare
% cx/cy qui introdurrebbe un'inconsistenza rispetto all'estrinseca, non una
% correzione. Verificato empiricamente sotto (>99% dei punti LiDAR proiettano
% dentro i bordi FLIR con questa convenzione).
%
% Frame di /cloud_registered: pubblicato da FAST-LIO2 gia' nel frame mondo
% ("camera_init"), non nel frame body/LiDAR. Per proiettare serve riportarlo nel
% frame body con la posa /Odometry dello stesso istante (gia' salvata nel
% triplet del sync manifest): p_body = R_wb' * (p_world - t_wb).
% Caveat: si assume frame body == frame LiDAR (nessuna extrinsic IMU-LiDAR
% separata nota per questo rig); errore atteso piccolo ma non quantificato.

clear
clc
close all

%% 0. Parametri configurabili

sessionRoot   = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM';
zedSessionDir = fullfile(sessionRoot, 'ZED', '20260730_161223', 'fullrate');
flirRot180Dir = fullfile(sessionRoot, 'Flir', 'session9_only_rot180');
bagPath       = fullfile(sessionRoot, 'Lidar', 'rosbag2_2026_07_30-18_12_20');
syncManifestPath = fullfile(zedSessionDir, 'sync_manifest.json');

tripletIdx = 9;          % indice 0-based nella lista "triplets" del sync manifest ("pose 9")
lidarSearchWindow_s = 1.0; % finestra di ricerca attorno al timestamp target, per /cloud_registered
maxRangeForPlot_m = inf;   % eventuale taglio sulla distanza dei punti (inf = nessun taglio)

% Filtro di occlusione (z-buffer) per camera: FLIR e ZED non sono co-locate
% (baseline reale ~13cm tra loro sul rig). Su uno spigolo vicino le due
% camere vedono "dietro l'angolo" in modo diverso: senza questo filtro un
% punto della parete nascosta dietro lo spigolo (invisibile a FLIR ma
% geometricamente dentro il suo frustum) viene comunque campionato,
% prendendo per errore il colore dello spigolo in primo piano (visto ad
% es. sessione 9 pose 71/106). Per ogni pixel intero di ciascuna camera si
% tiene solo il punto piu' vicino (± tol).
zBufferTol_m = 0.08;    % margine oltre il piu' vicino, stesso ordine dell'RMSE di calibrazione

outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 1. Sincronizzazione: leggi il triplet scelto dal sync manifest

fprintf('Carico sync manifest: %s\n', syncManifestPath);
manifest = jsondecode(fileread(syncManifestPath));
triplets = manifest.triplets;
nTriplets = numel(triplets);
if tripletIdx < 0 || tripletIdx >= nTriplets
    error('tripletIdx=%d fuori range [0, %d]', tripletIdx, nTriplets-1);
end
tr = triplets(tripletIdx + 1);   % MATLAB 1-indexed, manifest 0-indexed

fprintf('Triplet %d: match_status = %s\n', tripletIdx, tr.match_status);
fprintf('  FLIR : %s\n', tr.flir.file);
fprintf('  ZED  : %s\n', tr.zed.file);
fprintf('  LiDAR: t=%.6f  pos=[%.4f %.4f %.4f]\n', tr.lidar.timestamp_lidar, tr.lidar.position);

%% 2. Estrinseche LiDAR -> camera (risultati adottati, gia' calibrati)

% LiDAR -> FLIR, 6 pose pulite (01,02,03,05,07,08), min3d
% risultato_calibrazione_estrinseca_lidar_flir.md
R_lidar2flir = [ 0.048992   -0.998798   -0.00174722;
                 0.0621242   0.00479317 -0.998057;
                 0.996865    0.0487882   0.0622843 ];
t_lidar2flir = [-0.107859; -0.0426556; -0.0135286];
% RMSE min3D adottato: 5.8 cm (vedi nota) -> ordine di grandezza dell'errore
% di sovrapposizione atteso su una superficie qualsiasi, non solo sui 4 punti target.

% LiDAR -> ZED (occhio destro), 8 pose (01,02,03,05,07,08,09,10), min3d
% risultato_calibrazione_estrinseca_lidar_zed.md
R_lidar2zed = [ 0.00758424 -0.999954   -0.00584239;
               -0.0387419   0.00554434 -0.999234;
                0.99922     0.00780478 -0.0386981 ];
t_lidar2zed = [-0.0992363; 0.0888694; -0.000231952];
% RMSE min3D adottato: 6.8 cm

%% 3. Intrinseche (modello senza skew, quello raccomandato per la tesi in entrambe le note)

% FLIR Vue Pro R 336x256 - risultato_calibrazione_intrinseca_flir_vue_pro_r.md
Kf = [570.4796        0  149.1501;
             0  545.4275  117.0047;
             0         0         1];
kFlir = [-0.4241, -0.1241];     % k1, k2
pFlir = [-0.0053,  0.0025];     % p1, p2
flirW = 336; flirH = 256;

% ZED 2i occhio destro, 1080p - risultato_calibrazione_intrinseca_zed_2i_1080p.md
Kz = [1412.3362         0  1012.8503;
             0  1414.4716   569.7181;
             0         0          1];
kZed = [-1.558253e-01, 9.026829e-03];   % k1, k2
pZed = [ 6.208599e-04, 5.667587e-04];   % p1, p2
zedW = 1920; zedH = 1080;

%% 4. Estrai la scansione LiDAR del triplet dal bag ROS2 (.db3)

fprintf('\nApro bag: %s\n', bagPath);
bag = ros2bagreader(bagPath);

targetT = tr.lidar.timestamp_lidar;
sel = select(bag, 'Time', [targetT - lidarSearchWindow_s, targetT + lidarSearchWindow_s], ...
    'Topic', '/cloud_registered');
if sel.NumMessages == 0
    error('Nessun messaggio /cloud_registered trovato entro +-%.2fs da t=%.6f', ...
        lidarSearchWindow_s, targetT);
end
msgs = readMessages(sel);

bestIdx = 1; bestDt = inf;
for i = 1:numel(msgs)
    h = msgs{i}.header.stamp;
    t = double(h.sec) + double(h.nanosec) * 1e-9;
    dt = abs(t - targetT);
    if dt < bestDt
        bestDt = dt; bestIdx = i;
    end
end
fprintf('Scan LiDAR piu vicina: dt = %.4f s (su %d candidate nella finestra)\n', bestDt, numel(msgs));

ptsWorld = rosReadXYZ(msgs{bestIdx});   % /cloud_registered e' gia' nel frame mondo (camera_init)
fprintf('Punti nella scansione: %d\n', size(ptsWorld,1));

%% 5. Riporta i punti dal frame mondo al frame body/LiDAR con la posa /Odometry del triplet

t_wb = tr.lidar.position(:);                 % world_T_body, traslazione
q_xyzw = tr.lidar.orientation(:)';            % [x y z w], come pubblicato da geometry_msgs/Quaternion
q_wxyz = [q_xyzw(4), q_xyzw(1), q_xyzw(2), q_xyzw(3)];  % quat2rotm vuole [w x y z]
R_wb = quat2rotm(q_wxyz);

ptsBody = (R_wb' * (ptsWorld' - t_wb))';

if isfinite(maxRangeForPlot_m)
    keepRange = vecnorm(ptsBody, 2, 2) <= maxRangeForPlot_m;
    ptsBody = ptsBody(keepRange, :);
end

%% 6. Proietta gli stessi punti in FLIR e in ZED (stessa funzione, K/estrinseche diverse)

ptsFlir = (R_lidar2flir * ptsBody' + t_lidar2flir)';
ptsZed  = (R_lidar2zed  * ptsBody' + t_lidar2zed)';

[uFlir, vFlir, validFlir] = projectPinhole(ptsFlir, Kf, kFlir, pFlir, flirW, flirH);
[uZed,  vZed,  validZed ] = projectPinhole(ptsZed,  Kz, kZed,  pZed,  zedW,  zedH);

validBoth = validFlir & validZed;
fprintf('\nValidi in FLIR: %d / %d (%.1f%%)\n', sum(validFlir), numel(validFlir), 100*sum(validFlir)/numel(validFlir));
fprintf('Validi in ZED:  %d / %d (%.1f%%)\n', sum(validZed), numel(validZed), 100*sum(validZed)/numel(validZed));

% z-buffer per camera (occlusione): scarta i punti non piu' vicini nel loro
% pixel, calcolato solo sul sottoinsieme gia' valido in entrambe le camere
okFlir = false(size(validBoth)); okZed = false(size(validBoth));
okFlir(validBoth) = zBufferMask(uFlir(validBoth), vFlir(validBoth), ptsFlir(validBoth,3), ...
    flirW, flirH, zBufferTol_m);
okZed(validBoth)  = zBufferMask(uZed(validBoth),  vZed(validBoth),  ptsZed(validBoth,3), ...
    zedW,  zedH,  zBufferTol_m);
validBoth = validBoth & okFlir & okZed;
fprintf('Validi in entrambi dopo z-buffer occlusione (usati per l''overlay): %d\n', sum(validBoth));

%% 7. Immagine FLIR colorizzata dal dato radiometrico grezzo (.npy, 336x256, gia' ruotato 180)

[~, flirBase, ~] = fileparts(tr.flir.file);
flirBase = erase(flirBase, '_R');   % "20250906_233153_R" -> "20250906_233153"
flirNpyPath = fullfile(flirRot180Dir, [flirBase '.npy']);
fprintf('\nCarico dato radiometrico FLIR: %s\n', flirNpyPath);
flirRaw = readNpyFloat32(flirNpyPath);   % [H x W], stesso layout di rosReadXYZ/imread: righe=v, colonne=u

flirGray = mat2gray(flirRaw);
cmap = hot(256);
flirIdx = min(max(round(flirGray * 255) + 1, 1), 256);
flirRgb = ind2rgb(flirIdx, cmap);   % [H x W x 3], stessa griglia pixel della K FLIR sopra

%% 8. Immagine ZED e overlay

zedPath = fullfile(zedSessionDir, 'frames', tr.zed.file);
fprintf('Carico immagine ZED: %s\n', zedPath);
zedImg = imread(zedPath);

uF = round(uFlir(validBoth)); vF = round(vFlir(validBoth));
uZ = uZed(validBoth);         vZ = vZed(validBoth);

linIdx = sub2ind([flirH, flirW], vF, uF);
rCh = flirRgb(:,:,1); gCh = flirRgb(:,:,2); bCh = flirRgb(:,:,3);
sampledColors = [rCh(linIdx), gCh(linIdx), bCh(linIdx)];

fig = figure('Name', sprintf('FLIR su ZED - sessione 9, pose %d', tripletIdx));
imshow(zedImg); hold on;
scatter(uZ, vZ, 12, sampledColors, 'filled', 'MarkerFaceAlpha', 0.75);
title(sprintf('Sessione 9, pose %d — FLIR %s su ZED %s', tripletIdx, tr.flir.file, tr.zed.file), ...
    'Interpreter', 'none');

outPng = fullfile(outDir, sprintf('flir_on_zed_session9_pose%02d.png', tripletIdx));
exportgraphics(fig, outPng, 'Resolution', 200);
fprintf('\nSalvato: %s\n', outPng);

%% --- Funzioni locali ---

function mask = zBufferMask(u, v, z, W, H, tol)
% Per ogni pixel intero (round(u),round(v)), tiene solo i punti entro "tol"
% dalla profondita' minima osservata in quel pixel; scarta gli altri
% (occlusi da qualcosa di piu' vicino lungo lo stesso raggio della camera).
    if isempty(u)
        mask = false(0,1);
        return
    end
    uu = min(max(round(u), 1), W);
    vv = min(max(round(v), 1), H);
    binIdx = sub2ind([H, W], vv, uu);
    z = double(z);
    minZ = accumarray(binIdx, z, [W*H, 1], @min, Inf);
    mask = z <= minZ(binIdx) + tol;
end

function [u, v, valid] = projectPinhole(P, K, k, p, W, H)
% Proiezione pinhole + distorsione radiale/tangenziale (modello Brown-Conrady,
% stesso modello di estimateCameraParameters con skew=0), P in frame camera [Nx3].
    z = P(:,3);
    valid = z > 0.05;   % davanti alla camera
    xn = P(:,1) ./ z;
    yn = P(:,2) ./ z;
    r2 = xn.^2 + yn.^2;
    radial = 1 + k(1)*r2 + k(2)*r2.^2;
    xd = xn .* radial + 2*p(1)*xn.*yn + p(2)*(r2 + 2*xn.^2);
    yd = yn .* radial + p(1)*(r2 + 2*yn.^2) + 2*p(2)*xn.*yn;
    u = K(1,1)*xd + K(1,3);
    v = K(2,2)*yd + K(2,3);
    valid = valid & u >= 1 & u <= W & v >= 1 & v <= H;
end

function arr = readNpyFloat32(npyPath)
% Reader minimale per .npy float32 2D, formato NPY v1.0 (little-endian '<f4'),
% cosi' non serve una dipendenza esterna solo per caricare i frame FLIR.
    fid = fopen(npyPath, 'r');
    if fid < 0
        error('Impossibile aprire %s', npyPath);
    end
    cleanupObj = onCleanup(@() fclose(fid));
    magic = fread(fid, 6, 'uint8=>char')'; %#ok<NASGU> % "\x93NUMPY"
    fread(fid, 2, 'uint8');                % versione major/minor
    headerLen = fread(fid, 1, 'uint16');
    headerStr = fread(fid, headerLen, 'uint8=>char')';

    shapeTok = regexp(headerStr, "'shape':\s*\(([^)]*)\)", 'tokens', 'once');
    dims = str2double(strsplit(strtrim(shapeTok{1}), ','));
    dims(isnan(dims)) = [];
    if numel(dims) ~= 2
        error('Attesa shape 2D in %s, trovata [%s]', npyPath, num2str(dims));
    end
    nRows = dims(1); nCols = dims(2);

    if ~contains(headerStr, '<f4')
        error('Formato .npy non gestito (atteso float32 little-endian ''<f4''): %s', headerStr);
    end
    data = fread(fid, nRows * nCols, 'single=>single');
    arr = reshape(data, [nCols, nRows])';   % npy e' row-major: reshape column-major poi trasponi
end
