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
voxelSize  = 0.01;   % m, downsampling della nuvola GEOMETRICA (Figura 1, sez. 8).
                     % Qui non c'e' vincolo di registrazione incrociata LiDAR<->camera
                     % (sono le sole coordinate XYZ da SLAM): puo' restare fine. Le
                     % viste colorate per temperatura (sez. 9-10) usano un voxel
                     % diverso e piu' grosso, vedi voxelSizeTemp piu' sotto.
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
       -1.0 1.5, ...    % Y min max
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

%% 9. Colorazione per temperatura FLIR (TUTTA la sessione, multi-pose)
% /cloud_registered e' gia' nel frame mondo e questo frame non cambia nel
% tempo, quindi qualunque punto della mappa accumulata si puo' riproiettare
% "dove si trovava la FLIR" ad un preciso pose lungo la sessione (frustum
% check statico). Un solo pose copre solo un tratto stretto di corridoio
% (FOV FLIR ~32x26 gradi): per colorare TUTTO il corridoio si ripete la
% proiezione per OGNI triplet del sync manifest e si accumula, per ogni
% punto, la media delle temperature osservate da tutti i pose che lo
% vedevano (i tratti visti da piu' pose consecutivi vengono mediati, non
% sovrascritti). Stessa pipeline di estrinseche/intrinseche/z-buffer di
% ProjectFlirOnZed_Session9.m / FlirZedViewer_Session9.m (vedi quegli
% script per le fonti Obsidian dei parametri).
%
% Usa lo stesso pc (post ROI+denoise, PRIMA del downsample) della vista
% principale, ma per il BINNING finale usa un voxel proprio (voxelSizeTemp),
% non voxelSize: qui il colore di ogni punto viene da una riproiezione
% LiDAR->camera (T_lidar_to_flir), che ha un errore composto di ~9 cm RMSE
% (5.8 cm lidar->flir + 6.8 cm lidar->zed, vedi rig_calibration.yaml, PRIMA
% della deriva SLAM) su QUALE pixel un dato punto sta davvero campionando.
% Un voxel piu' piccolo di quell'errore non recupera dettaglio reale: mostra
% il rumore di corrispondenza incrociata come se fosse struttura. 15 cm
% (~1.6x il RMSE) resta un margine ragionevole sopra quella soglia; sotto i
% ~10-12 cm il margine sparisce, sotto i 9 cm si e' sotto la soglia stessa.
%
% useCorrectedTemp = true legge la temperatura CORRETTA prodotta da
% RadiometricCalibration/correct_session.py (emissivita' + atmosfera, con i
% materiali di consenso multi-vista di voxel_consensus.py --stage vote)
% invece della temperatura apparente grezza del sensore. Stesso formato
% .npy (float32, stessa shape), quindi il resto della pipeline di
% riproiezione non cambia: cambia solo QUALE file viene letto per pose.
% correct_session.py puo' scrivere NaN per un segmento senza candidato
% fisicamente plausibile (vedi la sua retry sulla plausibilita'): questi
% punti vengono esclusi dalla media invece di propagare NaN nell'accumulo,
% cosa che altrimenti azzererebbe permanentemente quel punto anche per
% tutte le pose successive che lo osservano correttamente.

useCorrectedTemp = true;    % false = temperatura apparente grezza (comportamento originale)
correctedName    = 'corrected_temperature_consensus.npy';
voxelSizeTemp    = 0.15;    % m, binning delle viste a temperatura (sez. 9-10). Non
                             % scendere sotto ~0.10-0.12 senza anche stringere
                             % zBufferTol_m: sono due fonti di errore comparabili
                             % che si sommano, non indipendenti.

sessionRootSlam  = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM';
sessionDirZed    = fullfile(sessionRootSlam, 'ZED', '20260730_161223', 'fullrate');
flirRot180Dir    = fullfile(sessionRootSlam, 'Flir', 'session9_only_rot180');
emissivityMapDir = fullfile(sessionDirZed, 'emissivity_map');
syncManifestPath = fullfile(sessionDirZed, 'sync_manifest.json');
zBufferTol_m     = 0.08;   % vedi ProjectFlirOnZed_Session9.m
rangeMax_m       = 20;     % prefiltro grossolano per velocita': punti oltre questa distanza dalla posa corrente non vengono nemmeno proiettati

if useCorrectedTemp
    fprintf('Sorgente temperatura: CORRETTA (%s)\n', correctedName);
else
    fprintf('Sorgente temperatura: apparente grezza (sensore, non corretta)\n');
end

% Estrinseca LiDAR -> FLIR, 6 pose pulite, min3d (risultato_calibrazione_estrinseca_lidar_flir.md)
R_lidar2flir = [ 0.048992   -0.998798   -0.00174722;
                 0.0621242   0.00479317 -0.998057;
                 0.996865    0.0487882   0.0622843 ];
t_lidar2flir = [-0.107859; -0.0426556; -0.0135286];

% Intrinseca FLIR, modello senza skew (risultato_calibrazione_intrinseca_flir_vue_pro_r.md)
Kf = [570.4796        0  149.1501;
             0  545.4275  117.0047;
             0         0         1];
kFlir = [-0.4241, -0.1241];
pFlir = [-0.0053,  0.0025];
flirW = 336; flirH = 256;

manifest = jsondecode(fileread(syncManifestPath));
allTriplets = manifest.triplets;
nT = numel(allTriplets);
fprintf('\n--- Colorazione per temperatura FLIR, TUTTA la sessione (%d pose) ---\n', nT);

xyzFilt = pc.Location;   % stessi punti (post ROI+denoise) che alimentano pcView
nPts = size(xyzFilt, 1);
sumTemp = zeros(nPts, 1);
cntTemp = zeros(nPts, 1, 'uint16');

ticId = tic;
for i = 1:nT
    tr = allTriplets(i);

    t_wb = tr.lidar.position(:);
    q_xyzw = tr.lidar.orientation(:)';
    q_wxyz = [q_xyzw(4), q_xyzw(1), q_xyzw(2), q_xyzw(3)];
    R_wb = quat2rotm(q_wxyz);

    % prefiltro grossolano: solo i punti abbastanza vicini alla posa corrente
    d2 = sum((xyzFilt - t_wb').^2, 2);
    nearIdx = find(d2 <= rangeMax_m^2);
    if isempty(nearIdx)
        continue
    end

    ptsBody = (R_wb' * (xyzFilt(nearIdx,:)' - t_wb))';
    ptsFlir = (R_lidar2flir * ptsBody' + t_lidar2flir)';

    [uFlir, vFlir, validFlir] = projectPinholeTemp(ptsFlir, Kf, kFlir, pFlir, flirW, flirH);
    if ~any(validFlir)
        continue
    end
    okFlir = false(size(validFlir));
    okFlir(validFlir) = zBufferMaskTemp(uFlir(validFlir), vFlir(validFlir), ptsFlir(validFlir,3), ...
        flirW, flirH, zBufferTol_m);
    validFlir = validFlir & okFlir;
    if ~any(validFlir)
        continue
    end

    [~, flirStemFull, ~] = fileparts(tr.flir.file);   % es. 20250906_233144_R
    if useCorrectedTemp
        npyPath = fullfile(emissivityMapDir, flirStemFull, correctedName);
        if ~isfile(npyPath)
            continue   % frame senza correzione (classify_session/correct_session non eseguiti su questo frame)
        end
    else
        flirBase = erase(flirStemFull, '_R');         % il .npy grezzo non ha il suffisso _R
        npyPath = fullfile(flirRot180Dir, [flirBase '.npy']);
    end
    flirRaw = readNpyFloat32Temp(npyPath);

    uF = round(uFlir(validFlir)); vF = round(vFlir(validFlir));
    linIdx = sub2ind([flirH, flirW], vF, uF);
    vals = double(flirRaw(linIdx));

    % correct_session.py puo' scrivere NaN dove nessun materiale candidato
    % dava una temperatura fisicamente plausibile: si esclude quel punto da
    % QUESTA osservazione soltanto, senza corrompere il suo accumulo per le
    % pose successive (la temperatura apparente grezza non e' mai NaN, ma il
    % controllo costa nulla e rende il codice corretto in entrambi i casi).
    globalIdx = nearIdx(validFlir);
    okVal = isfinite(vals);
    globalIdx = globalIdx(okVal);
    vals = vals(okVal);

    sumTemp(globalIdx) = sumTemp(globalIdx) + vals;
    cntTemp(globalIdx) = cntTemp(globalIdx) + 1;

    if mod(i, 20) == 0 || i == nT
        fprintf('  pose %d/%d, punti coperti finora: %d, %.0fs trascorsi\n', ...
            i, nT, sum(cntTemp > 0), toc(ticId));
    end
end

hasObs = cntTemp > 0;
temperature = nan(nPts, 1);
temperature(hasObs) = sumTemp(hasObs) ./ double(cntTemp(hasObs));

fprintf('Punti con temperatura valida (>=1 pose): %d / %d (%.1f%%)\n', ...
    sum(hasObs), nPts, 100*sum(hasObs)/nPts);
fprintf('Osservazioni per punto coperto: media=%.1f  max=%d\n', ...
    mean(cntTemp(hasObs)), max(cntTemp));

if useCorrectedTemp
    tempLabel = 'Temperatura CORRETTA (°C, emissivita'' di consenso + atmosfera)';
else
    tempLabel = 'Temperatura (°C, dato radiometrico FLIR grezzo)';
end

pcTemp = pointCloud(xyzFilt, 'Intensity', temperature);
pcViewTemp = pcdownsample(pcTemp, 'gridAverage', voxelSizeTemp);

hasTemp = isfinite(pcViewTemp.Intensity);
fprintf('Voxel con temperatura media valida: %d / %d\n', sum(hasTemp), pcViewTemp.Count);

if any(hasTemp)
    pcViewTempValid = select(pcViewTemp, find(hasTemp));

    figure('Name', sprintf('Temperatura FLIR su voxel - %d pose', nT), 'Color', 'k');
    pcshow(pcViewTempValid.Location, pcViewTempValid.Intensity, 'MarkerSize', markerSize);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Temperatura media per voxel (%.0f cm) - %d pose fusi', ...
        voxelSizeTemp*100, nT), 'Color', 'w', 'Interpreter', 'none');
    axis equal;
    grid on;
    colormap(gca, hot);
    cb = colorbar;
    cb.Color = 'w';
    cb.Label.String = tempLabel;
    cb.Label.Color = 'w';

    tVals = pcViewTempValid.Intensity;
    fprintf('Temperatura media per voxel: min=%.1f  media=%.1f  max=%.1f  [gradi, unita'' del dato .npy]\n', ...
        min(tVals), mean(tVals), max(tVals));
else
    warning('Nessun voxel con temperatura valida su tutta la sessione.');
end

%% 10. Voxel come cubi trasparenti (non solo un punto al centro)
% Disegna ogni voxel occupato come un cubo vero, colorato con la temperatura
% media dei punti che ci cadono dentro, con trasparenza.
%
% Stessa soglia di sezione 9: il colore per voxel viene da punti la cui
% corrispondenza LiDAR->camera ha ~9 cm RMSE di incertezza, quindi il cubo
% non dovrebbe essere piu' piccolo di quello. Di default e' quindi uguale a
% voxelSizeTemp, non un valore indipendente scelto solo per la resa.
%
% A 1 cm i voxel occupati sarebbero ~570k, comunque non renderizzabili con
% la trasparenza attiva (MATLAB si pianta) -- ma non e' quello il motivo
% principale per cui questa vista non scende sotto i 15 cm.
%
% Ottimizzazione: le facce condivise tra due voxel adiacenti entrambi
% occupati vengono scartate (face culling). Serve sia per le prestazioni
% sia per la resa: con la trasparenza attiva, le facce interne nascoste si
% sommerebbero visivamente rendendo tutto opaco e confuso.

voxelSizeCubes = voxelSizeTemp;   % m, lato del cubo. Cambiarlo qui lo scollega dal
                                   % ragionamento sulla soglia di rumore qui sopra,
                                   % non solo dal valore di default.
cubeAlpha      = 0.5;   % 0 = invisibile, 1 = opaco
cubeEdges      = false;  % true = disegna gli spigoli (leggibile solo con pochi voxel)

fprintf('\n--- Voxel come cubi trasparenti (%.0f cm) ---\n', voxelSizeCubes*100);

validT = isfinite(temperature);
ivAll = floor(xyzFilt(validT,:) / voxelSizeCubes);      % coordinate intere di voxel
tAll  = temperature(validT);

[ivU, ~, ic] = unique(ivAll, 'rows');
meanT = accumarray(ic, tAll, [], @mean);                % temperatura media per voxel
nVox = size(ivU, 1);
fprintf('Voxel occupati: %d\n', nVox);

% --- geometria: 8 vertici e 6 facce per voxel, vettorizzato ---
% Corner locale del cubo unitario, poi scalato e traslato sul voxel
cornerOffsets = [0 0 0; 1 0 0; 1 1 0; 0 1 0; ...
                 0 0 1; 1 0 1; 1 1 1; 0 1 1];
% Facce come quad sugli 8 vertici locali, una riga per direzione
faceDefs = [1 2 3 4;   % -Z
            5 6 7 8;   % +Z
            1 2 6 5;   % -Y
            4 3 7 8;   % +Y
            1 4 8 5;   % -X
            2 3 7 6];  % +X
% Direzione del vicino che nasconde ciascuna faccia, stesso ordine
faceNeighborDir = [0 0 -1; 0 0 1; 0 -1 0; 0 1 0; -1 0 0; 1 0 0];

% Vertici: 8 per voxel (con duplicati tra voxel adiacenti, accettabile)
vertsAll = zeros(nVox * 8, 3);
for c = 1:8
    vertsAll(c:8:end, :) = (ivU + cornerOffsets(c,:)) * voxelSizeCubes;
end

% Facce visibili: scarta quelle verso un vicino occupato
baseIdx = (0:nVox-1)' * 8;
facesVis = cell(6, 1);
colorVis = cell(6, 1);
for f = 1:6
    hidden = ismember(ivU + faceNeighborDir(f,:), ivU, 'rows');
    keep = ~hidden;
    facesVis{f} = baseIdx(keep) + faceDefs(f,:);
    colorVis{f} = meanT(keep);
end
faces = vertcat(facesVis{:});
faceColors = vertcat(colorVis{:});

fprintf('Facce totali: %d, visibili dopo culling: %d (%.0f%% scartate)\n', ...
    nVox*6, size(faces,1), 100*(1 - size(faces,1)/(nVox*6)));

figure('Name', sprintf('Voxel cubi trasparenti %.0f cm - %d pose', ...
    voxelSizeCubes*100, nT), 'Color', 'k');
if cubeEdges
    edgeArg = {'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.1};
else
    edgeArg = {'EdgeColor', 'none'};
end
patch('Vertices', vertsAll, 'Faces', faces, ...
    'FaceVertexCData', faceColors, 'FaceColor', 'flat', ...
    'FaceAlpha', cubeAlpha, edgeArg{:});

ax = gca;
ax.Color = 'k';
ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Voxel %.0f cm come cubi trasparenti (alpha %.2f) - %d pose fusi', ...
    voxelSizeCubes*100, cubeAlpha, nT), 'Color', 'w');
axis equal; grid on; view(3);
colormap(ax, hot);
cbc = colorbar;
cbc.Color = 'w';
cbc.Label.String = tempLabel;
cbc.Label.Color = 'w';
camlight headlight; lighting none;   % niente shading: il colore e' il dato, non la luce

fprintf('Temperatura per voxel (%.0f cm): min=%.1f  media=%.1f  max=%.1f\n', ...
    voxelSizeCubes*100, min(meanT), mean(meanT), max(meanT));

%% --- Funzioni locali ---

function [u, v, valid] = projectPinholeTemp(P, K, k, p, W, H)
    z = P(:,3);
    valid = z > 0.05;
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

function mask = zBufferMaskTemp(u, v, z, W, H, tol)
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

function arr = readNpyFloat32Temp(npyPath)
    fid = fopen(npyPath, 'r');
    if fid < 0
        error('Impossibile aprire %s', npyPath);
    end
    cleanupObj = onCleanup(@() fclose(fid));
    fread(fid, 6, 'uint8=>char');
    fread(fid, 2, 'uint8');
    headerLen = fread(fid, 1, 'uint16');
    headerStr = fread(fid, headerLen, 'uint8=>char')';

    shapeTok = regexp(headerStr, "'shape':\s*\(([^)]*)\)", 'tokens', 'once');
    dims = str2double(strsplit(strtrim(shapeTok{1}), ','));
    dims(isnan(dims)) = [];
    nRows = dims(1); nCols = dims(2);

    if ~contains(headerStr, '<f4')
        error('Formato .npy non gestito (atteso float32 little-endian ''<f4''): %s', headerStr);
    end
    data = fread(fid, nRows * nCols, 'single=>single');
    arr = reshape(data, [nCols, nRows])';
end