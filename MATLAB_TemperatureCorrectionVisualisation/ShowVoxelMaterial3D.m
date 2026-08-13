%% Visualizzazione 3D del materiale di consenso per voxel (20 cm)
% Stessa idea di ..\MATLAB_PointCloudVisualization\Piano1_CorridoioLungo.m,
% sez. 9-10 (voxel come cubi trasparenti, geometria fine / colore stabile),
% ma al posto della temperatura FLIR colora ogni cubo per il MATERIALE di
% consenso. Il materiale e' gia' per-voxel (20 cm) nel CSV scritto da
% EmissivityCalculation\voxel_consensus.py --stage thermal (colonna
% 'material'), quindi non serve rifare ne' la classificazione ne' il voto:
% serve solo, come in Piano1, DISACCOPPIARE la geometria mostrata dalla
% risoluzione del dato.
%
% Perche' la bag e' necessaria (enableFineVoxels = true)
% ---------------------------------------------------------
% Suddividere ogni cubo da 20 cm in NxNxN cubetti tutti dello stesso colore
% non basta: le facce condivise fra due cubetti adiacenti dello STESSO
% colore vengono scartate dal culling (stessa ottimizzazione di Piano1
% sez. 10), quindi il contorno esterno risultante e' identico al cubo
% grande di partenza. Qui si rilegge /cloud_registered dalla bag (stessa
% sorgente di voxel_consensus.py), ogni punto grezzo prende il materiale
% del voxel da 20 cm in cui cade, e la geometria mostrata segue la densita'
% REALE dei punti in quel voxel, non un riempimento uniforme.
%
% Buchi = vetro senza ritorno (fillGlassHoles)
% ---------------------------------------------
% Anche con la geometria fine restano buchi netti sulle pareti nord/sud,
% allineati con le finestre: il vetro trasmette il raggio invece di
% rifletterlo, quindi quel voxel non riceve mai un punto valido. Un
% opaco (intonaco, cemento, metallo) da' sempre un ritorno -- un buco
% dentro il rettangolo occupato di una parete e' quindi un forte indizio di
% vetro, anche senza aver mai ricevuto un campione. fillGlassHoles = true
% (default) riempie questi buchi con un voxel 'glass' inferito: per ogni
% parete in glassFillPlaneIds, si costruisce la griglia COMPLETA (20 cm)
% del suo rettangolo occupato e si marca vetro ogni cella non gia' presente
% fra i voxel dati reali. Colorati in un teal piu' chiaro del vetro reale
% (misurato) per restare distinguibili, con conteggio separato in legenda.
% NOTA: e' un'euristica per esclusione, non una misura -- un pilastro o un
% radiatore che occlude la parete lascia la stessa firma (nessun punto li')
% e verrebbe marcato vetro allo stesso modo. Il topic salva solo i colpi,
% non i raggi mancati: non c'e' modo di distinguere i due casi senza quella
% informazione. fillGlassHoles = false disattiva questo riempimento.
%
% enableFineVoxels = false salta la bag e disegna direttamente i cubi da
% 20 cm del CSV (comportamento piu' veloce, nessuna dipendenza dalla bag).
%
% Filtro stanza (--room-bbox): esclude i voxel fuori dall'ingombro x/y del
% floor plane (planes.json, id 0) -- rimuove il rumore LiDAR passato
% attraverso il vetro delle finestre (il raggio prosegue oltre e ritorna da
% fuori edificio). Confermato su questa sessione: i 16 voxel entro 0.15 m
% dal muro di testa (id3, x=34.45) avevano y=2.7-3.5 m, ~1-2 m oltre la
% parete sud (stanza larga 2.46 m) -- non la porta, rumore.
%
% Comandi da tastiera:
%   m   mostra / nasconde la legenda dei materiali
%   g   mostra / nasconde gli spigoli dei cubi
%   i   mostra / nasconde i voxel vetro inferiti (buchi)
%
% Uso:
%   ShowVoxelMaterial3D                          % percorsi di default, sotto
%   ShowVoxelMaterial3D(csvPathIn)
%   ShowVoxelMaterial3D(csvPathIn, planesPathIn)

function ShowVoxelMaterial3D(csvPathIn, planesPathIn)

close all
clc

%% 1. Parametri
sessionDir = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate';
csvPath    = fullfile(sessionDir, 'voxel_map', 'thermal_voxels.csv');   % sessione 9, voxel 20 cm
planesPath = 'C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis\OpenStudioModel\planes.json';

if nargin >= 1 && ~isempty(csvPathIn)
    csvPath = csvPathIn;
end
if nargin >= 2 && ~isempty(planesPathIn)
    planesPath = planesPathIn;
end

voxelSize   = 0.20;    % m, deve combaciare con --voxel di voxel_consensus.py
cubeAlpha   = 1;       % 0 = invisibile, 1 = opaco
cubeEdges   = false;   % true = disegna gli spigoli (leggibile solo con pochi voxel)
useRoomBBox = true;    % esclude i voxel fuori dall'ingombro x/y/z della stanza (vedi sopra)
bboxPlaneId = 0;       % id del floor plane in planesPath usato per l'ingombro x/y e per z minima
ceilingPlaneId = 2;    % id del ceiling plane usato per z massima
bboxTol     = 0.15;    % m, stesso margine di voxel_solar_ns.py --plane-threshold

% --- Crop ROI (Piano1_CorridoioLungo.m, sez. 5a) ---
useROI = true;
roi = [-Inf Inf, ...    % X min max
       0 Inf, ...    % Y min max
       -3 6];       % Z min max

% --- Voxel fini per la resa (Piano1_CorridoioLungo.m, sez. 9-10) ---
enableFineVoxels = true;
voxelSizeCubes   = 0.05;   % m, granularita' del binning dei punti grezzi per la resa

% --- Buchi = vetro inferito (vedi commento in testa) ---
fillGlassHoles   = true;
glassFillPlaneIds = [1 4];   % nord, sud -- pareti dove il pattern vetro/muro e' noto
glassFillTol      = 0.15;    % m, distanza dal piano per l'appartenenza alla parete
glassInferredColor = 0.5 * [60 210 200]/255 + 0.5 * [1 1 1];   % teal reale, sbiadito
% I voxel inferiti non hanno punti LiDAR reali da cui prendere una densita':
% suddividerli in cubetti tutti dello stesso colore darebbe di nuovo un
% blocco liscio (le facce interne condivise vengono scartate dal culling,
% vedi commento in testa -- stesso limite gia' incontrato sulla geometria
% reale). L'unico modo per farli sembrare "fini" senza dati e' rimpicciolirli
% (shrink < 1) cosi' resta un vuoto visibile fra un cubetto e l'altro: una
% griglia di cubetti separati invece di una lastra piena. Con shrink < 1 il
% culling fra celle adiacenti viene disattivato (i cubetti non si toccano
% davvero, ogni faccia e' potenzialmente visibile).
glassInferredCubeSize = 0.05;   % m, lato del cubetto inferito mostrato
glassInferredShrink   = 0.6;    % 0-1, frazione della cella occupata (1 = pieno, senza vuoti)

% --- Lettura bag (solo se enableFineVoxels = true) ---
bagPath   = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20';
topicName = '/cloud_registered';
frameStep = 1;
maxFrames = Inf;
useDenoise   = false;
denoiseK     = 20;
denoiseThres = 1.0;

if ~isfile(csvPath)
    error(['CSV voxel non trovato: %s\n' ...
           'Eseguire prima EmissivityCalculation\\voxel_consensus.py --stage thermal'], csvPath);
end

%% 2. Lettura del CSV (materiale di consenso per voxel dati, 20 cm)
T = readtable(csvPath, 'TextType', 'string');
needCols = {'x', 'y', 'z', 'material'};
missing = needCols(~ismember(needCols, T.Properties.VariableNames));
if ~isempty(missing)
    error('%s manca delle colonne: %s', csvPath, strjoin(missing, ', '));
end
xyzCsv = [T.x, T.y, T.z];
materialCsv = string(T.material);
materialCsv(materialCsv == "") = "sconosciuto";
nRaw = numel(materialCsv);
fprintf('%d voxel dati letti da %s\n', nRaw, csvPath);

%% 3. Crop ROI sui voxel dati (solo se enableFineVoxels = false; altrimenti sui punti grezzi, sez. 9)
if useROI && ~enableFineVoxels
    nBefore = numel(materialCsv);
    inRoi = xyzCsv(:,1) >= roi(1) & xyzCsv(:,1) <= roi(2) & ...
            xyzCsv(:,2) >= roi(3) & xyzCsv(:,2) <= roi(4) & ...
            xyzCsv(:,3) >= roi(5) & xyzCsv(:,3) <= roi(6);
    xyzCsv = xyzCsv(inRoi, :);
    materialCsv = materialCsv(inRoi);
    fprintf('ROI crop: %d -> %d voxel dati (rimossi %d, %.1f%%)\n', ...
        nBefore, numel(materialCsv), nBefore - numel(materialCsv), ...
        100*(nBefore - numel(materialCsv))/nBefore);
    if isempty(materialCsv)
        error('Nessun voxel rimasto dopo il crop ROI.');
    end
end

%% 4. Filtro ingombro stanza sui voxel dati (rumore LiDAR attraverso il vetro)
planes = [];
if isfile(planesPath)
    planes = jsondecode(fileread(planesPath));
end

nBeforeBBox = numel(materialCsv);
if useRoomBBox
    if isempty(planes)
        warning('planes.json non trovato (%s): filtro stanza disattivato', planesPath);
    else
        floorPlane = findPlane(planes, bboxPlaneId);
        ceilingPlane = findPlane(planes, ceilingPlaneId);
        if isempty(floorPlane)
            warning('Nessun piano con id %d in %s: filtro stanza disattivato', bboxPlaneId, planesPath);
        else
            corners = floorPlane.corners_3d;   % 4x3
            rx = [min(corners(:,1)) max(corners(:,1))] + [-bboxTol bboxTol];
            ry = [min(corners(:,2)) max(corners(:,2))] + [-bboxTol bboxTol];
            % z: dal pavimento (corners del floor plane) al soffitto (se
            % trovato, altrimenti solo il pavimento fa da limite inferiore).
            % Senza questo, gli outlier z del rumore attraverso vetro (stesso
            % fenomeno gia' visto su y per id3 e sulle pareti nord/sud, qui
            % fino a z=5.5 m contro un soffitto reale a 1.8 m) restano nel
            % CSV, finiscono nella geometria fine (un punto grezzo reale puo'
            % cadere nello stesso voxel spurio) e fanno esplodere gli assi
            % del grafico (axis equal include qualunque punto plottato,
            % anche uno solo, lontanissimo).
            if isempty(ceilingPlane)
                warning('Nessun piano soffitto con id %d: limito z solo dal basso (pavimento)', ceilingPlaneId);
                rz = [min(corners(:,3)) Inf];
            else
                cCeil = ceilingPlane.corners_3d;
                rz = [min(corners(:,3)) max(cCeil(:,3))] + [-bboxTol bboxTol];
            end
            keep = xyzCsv(:,1) >= rx(1) & xyzCsv(:,1) <= rx(2) & ...
                   xyzCsv(:,2) >= ry(1) & xyzCsv(:,2) <= ry(2) & ...
                   xyzCsv(:,3) >= rz(1) & xyzCsv(:,3) <= rz(2);
            fprintf(['--room-bbox: ingombro floor plane %d x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f] ' ...
                     '(tolleranza %.2f m) -- scartati %d/%d voxel dati fuori stanza\n'], ...
                    bboxPlaneId, rx(1), rx(2), ry(1), ry(2), rz(1), rz(2), bboxTol, ...
                    nBeforeBBox - sum(keep), nBeforeBBox);
            xyzCsv = xyzCsv(keep, :);
            materialCsv = materialCsv(keep);
        end
    end
end
nVoxCsv = size(xyzCsv, 1);
if nVoxCsv == 0
    error('Nessun voxel dati rimasto dopo i filtri.');
end
ivCsv = round(xyzCsv / voxelSize - 0.5);   % chiave intera, stesso binning di voxel_consensus.py

%% 5. Voxel vetro inferiti sui buchi (vedi commento in testa)
ivInferred = zeros(0, 3);
nInferred = 0;
if fillGlassHoles
    if isempty(planes)
        warning('planes.json non trovato: riempimento buchi vetro disattivato');
    else
        for pid = glassFillPlaneIds
            wallPlane = findPlane(planes, pid);
            if isempty(wallPlane)
                warning('Nessun piano con id %d: salto il riempimento per quella parete', pid);
                continue
            end
            n = wallPlane.normal(:)';
            d = wallPlane.d;
            distToPlane = abs(xyzCsv * n' + d) / norm(n);
            wallMask = distToPlane < glassFillTol;
            if ~any(wallMask)
                continue
            end
            [constAxis, ax1, ax2] = planeAxes(n);
            ivWall = ivCsv(wallMask, :);
            % Una colonna (x,z) e' un buco solo se NON c'e' nulla su NESSUNO
            % degli strati lungo l'asse di spessore -- il rumore di
            % registrazione sparge i dati reali su piu' di uno strato (nord
            % ha 822 voxel a y=-0.7 e anche 30 a y=-0.9, confermato su questa
            % sessione), e un dato reale su UN SOLO strato basta a dire che
            % la parete e' li'. Confrontare per-strato (versione precedente)
            % marcava vetro la stessa colonna sullo strato minoritario anche
            % quando quello dominante aveva gia' un dato reale -- risultato:
            % vetro sovrapposto al materiale reale, effetto "a righe".
            occupiedXZ = unique(ivWall(:, [ax1 ax2]), 'rows');
            % Range della griglia dai corners_3d REALI del piano, non dal
            % min/max dei voxel dati: bastano pochi outlier (stesso rumore
            % attraverso vetro di id3, qui su x/z invece che y -- confermato
            % su questa sessione: 3-6 voxel a z=-0.5 o z=2.1, ben oltre
            % pavimento/soffitto veri a z=-0.27/1.80) per allungare min:max
            % ben oltre l'ingombro vero della parete.
            corners = wallPlane.corners_3d;
            a1 = floor(min(corners(:,ax1))/voxelSize - 0.5) : ceil(max(corners(:,ax1))/voxelSize - 0.5);
            a2 = floor(min(corners(:,ax2))/voxelSize - 0.5) : ceil(max(corners(:,ax2))/voxelSize - 0.5);
            [g1, g2] = ndgrid(a1, a2);
            candXZ = [g1(:) g2(:)];
            holesXZ = setdiff(candXZ, occupiedXZ, 'rows');

            % Il vetro inferito viene generato direttamente in coordinate
            % FINI (glassInferredCubeSize), non a 20 cm: una singola lamina
            % spessa un voxel fine, posata sul piano della parete. Riempire
            % l'intero voxel dati da 20 cm (versione precedente) produce una
            % lastra spessa 20 cm che sporge di 15 cm oltre la superficie
            % reale del muro, perche' i punti LiDAR veri stanno solo sul
            % primo strato fine (la superficie), non su tutta la profondita'.
            % La quota di profondita' viene dal piano stesso (n*p + d = 0,
            % assi allineati => p = -d/n(constAxis)), cioe' esattamente dove
            % fit_planes.py ha misurato la parete.
            planePos = -d / n(constAxis);
            depthFine = floor(planePos / glassInferredCubeSize);
            ratioFine = round(voxelSize / glassInferredCubeSize);
            % Ogni colonna-buco da 20 cm copre ratioFine x ratioFine celle
            % fini nel piano della parete (ma una sola in profondita').
            [f1, f2] = ndgrid(0:ratioFine-1, 0:ratioFine-1);
            sub = [f1(:) f2(:)];
            nH = size(holesXZ, 1);
            nS = size(sub, 1);
            holes = zeros(nH * nS, 3);
            for si = 1:nS
                rows = (si-1)*nH + (1:nH);
                holes(rows, ax1) = holesXZ(:, 1) * ratioFine + sub(si, 1);
                holes(rows, ax2) = holesXZ(:, 2) * ratioFine + sub(si, 2);
                holes(rows, constAxis) = depthFine;
            end
            fprintf(['Piano %d: parete %d x %d celle (20 cm), %d voxel dati reali, ' ...
                     '%d colonne (x,z) mai occupate -> %d celle vetro inferite ' ...
                     '(%.0f cm, lamina singola sul piano y/x=%.3f m)\n'], ...
                pid, numel(a1), numel(a2), sum(wallMask), nH, size(holes,1), ...
                glassInferredCubeSize*100, planePos);
            ivInferred = [ivInferred; holes]; %#ok<AGROW>
        end
        ivInferred = unique(ivInferred, 'rows');
        fprintf('Totale voxel vetro inferiti (prima del ROI): %d\n', size(ivInferred, 1));

        % Il ROI si applica anche ai voxel inferiti: sono generati sull'intero
        % rettangolo della parete (sez. 5 sopra), a monte del crop ROI dei
        % dati reali (sez. 3/7b) -- senza questo, restringere la vista con
        % useROI lascerebbe comunque vetro inferito fuori dal crop.
        if useROI
            centerInf = (ivInferred + 0.5) * glassInferredCubeSize;   % ivInferred e' gia' in celle fini
            inRoiInf = centerInf(:,1) >= roi(1) & centerInf(:,1) <= roi(2) & ...
                       centerInf(:,2) >= roi(3) & centerInf(:,2) <= roi(4) & ...
                       centerInf(:,3) >= roi(5) & centerInf(:,3) <= roi(6);
            nBeforeRoiInf = size(ivInferred, 1);
            ivInferred = ivInferred(inRoiInf, :);
            fprintf('ROI crop sul vetro inferito: %d -> %d\n', nBeforeRoiInf, size(ivInferred,1));
        end
        nInferred = size(ivInferred, 1);
        fprintf('Totale voxel vetro inferiti: %d\n', nInferred);
    end
end

%% 6. Colori per materiale (stessi di ShowMaterialConsensus.m, per restare confrontabili)
colorMap = materialColors();
matNames = unique(materialCsv);
fprintf('Materiali presenti nei voxel dati (%d):\n', numel(matNames));
counts = zeros(numel(matNames), 1);
for i = 1:numel(matNames)
    counts(i) = sum(materialCsv == matNames(i));
    fprintf('  %-16s %5d voxel\n', matNames(i), counts(i));
end
rgbTable = zeros(numel(matNames), 3);
for k = 1:numel(matNames)
    rgbTable(k, :) = materialColor(colorMap, matNames(k));
end
matIdxCsv = zeros(nVoxCsv, 1);
for k = 1:numel(matNames)
    matIdxCsv(materialCsv == matNames(k)) = k;
end

%% 7. Geometria del materiale reale: fine (punti veri) oppure cubi pieni dal CSV
if enableFineVoxels
    %% 7a. Lettura bag ROS2 (Piano1_CorridoioLungo.m, sez. 1-4)
    fprintf('\n--- Lettura bag per la geometria fine (enableFineVoxels = true) ---\n');
    bag = ros2bagreader(bagPath);
    sel = select(bag, 'Topic', topicName);
    nTotal = sel.NumMessages;
    fprintf('Topic %s: %d messaggi\n', topicName, nTotal);
    if nTotal == 0
        error(['Nessun messaggio su %s.\n' ...
            'Controllare il nome del topic (proprieta'' AvailableTopics della bag).'], topicName);
    end

    idxMsg = 1:frameStep:nTotal;
    if numel(idxMsg) > maxFrames
        idxMsg = idxMsg(1:maxFrames);
    end
    fprintf('Lettura di %d frame (step %d)\n', numel(idxMsg), frameStep);
    msgs = readMessages(sel, idxMsg);

    allXYZ = cell(numel(msgs), 1);
    for i = 1:numel(msgs)
        allXYZ{i} = rosReadXYZ(msgs{i});
    end
    xyzRawPts = vertcat(allXYZ{:});
    nRawPts = size(xyzRawPts, 1);
    xyzRawPts = xyzRawPts(all(isfinite(xyzRawPts), 2), :);
    fprintf('Punti letti: %d, validi: %d (scartati %d NaN/Inf)\n', ...
        nRawPts, size(xyzRawPts,1), nRawPts - size(xyzRawPts,1));

    %% 7b. Crop ROI e denoise sui punti grezzi
    if useROI
        nBefore = size(xyzRawPts, 1);
        inRoi = xyzRawPts(:,1) >= roi(1) & xyzRawPts(:,1) <= roi(2) & ...
                xyzRawPts(:,2) >= roi(3) & xyzRawPts(:,2) <= roi(4) & ...
                xyzRawPts(:,3) >= roi(5) & xyzRawPts(:,3) <= roi(6);
        xyzRawPts = xyzRawPts(inRoi, :);
        fprintf('ROI crop: %d -> %d punti grezzi (rimossi %d, %.1f%%)\n', ...
            nBefore, size(xyzRawPts,1), nBefore - size(xyzRawPts,1), ...
            100*(nBefore - size(xyzRawPts,1))/max(1,nBefore));
    end
    if useDenoise
        nBefore = size(xyzRawPts, 1);
        pcRaw = pcdenoise(pointCloud(xyzRawPts), 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
        xyzRawPts = pcRaw.Location;
        fprintf('Denoise: %d -> %d punti (rimossi %d, %.1f%%)\n', ...
            nBefore, size(xyzRawPts,1), nBefore - size(xyzRawPts,1), ...
            100*(nBefore - size(xyzRawPts,1))/max(1,nBefore));
    end

    %% 7c. Assegnazione del materiale ai punti grezzi (binning vettoriale via lookup table)
    ixMin = min(ivCsv(:,1)); ixMax = max(ivCsv(:,1));
    iyMin = min(ivCsv(:,2)); iyMax = max(ivCsv(:,2));
    izMin = min(ivCsv(:,3)); izMax = max(ivCsv(:,3));
    nx = ixMax - ixMin + 1; ny = iyMax - iyMin + 1; nz = izMax - izMin + 1;

    lut = zeros(nx, ny, nz, 'int32');   % 0 = nessun voxel dati valido li'
    lutLinIdx = sub2ind([nx ny nz], ivCsv(:,1)-ixMin+1, ivCsv(:,2)-iyMin+1, ivCsv(:,3)-izMin+1);
    lut(lutLinIdx) = matIdxCsv;

    ivRawPts = floor(xyzRawPts / voxelSize);
    ixR = ivRawPts(:,1) - ixMin + 1;
    iyR = ivRawPts(:,2) - iyMin + 1;
    izR = ivRawPts(:,3) - izMin + 1;
    inRange = ixR >= 1 & ixR <= nx & iyR >= 1 & iyR <= ny & izR >= 1 & izR <= nz;

    matIdxRaw = zeros(size(xyzRawPts,1), 1, 'int32');
    linIdx = sub2ind([nx ny nz], ixR(inRange), iyR(inRange), izR(inRange));
    matIdxRaw(inRange) = lut(linIdx);

    validPt = matIdxRaw > 0;
    fprintf('Punti grezzi con voxel dati valido: %d / %d (%.1f%%)\n', ...
        sum(validPt), numel(validPt), 100*sum(validPt)/numel(validPt));
    if ~any(validPt)
        error('Nessun punto grezzo cade in un voxel dati valido -- controllare bagPath/voxelSize/planesPath.');
    end
    xyzValid = xyzRawPts(validPt, :);
    rgbValid = rgbTable(matIdxRaw(validPt), :);

    %% 7d. Voxel fini per la resa: un cubo per ogni cella occupata da punti veri
    ratio = voxelSize / voxelSizeCubes;
    if abs(ratio - round(ratio)) > 1e-6
        warning(['voxelSize (%.3f) non e'' un multiplo esatto di voxelSizeCubes (%.3f).'], ...
            voxelSize, voxelSizeCubes);
    end
    ivFine = floor(xyzValid / voxelSizeCubes);
    [ivGeom, ia, ~] = unique(ivFine, 'rows');
    rgbGeom = rgbValid(ia, :);
    sizeGeom = voxelSizeCubes;
    fprintf('Voxel fini: %d punti validi -> %d celle occupate (%.0f cm)\n', ...
        size(xyzValid,1), size(ivGeom,1), voxelSizeCubes*100);
else
    ivGeom = ivCsv;
    rgbGeom = rgbTable(matIdxCsv, :);
    sizeGeom = voxelSize;
end

%% 8. Geometria dei cubi (real data + vetro inferito come patch separati)
[vertsReal, facesReal, colorsReal] = buildCubePatch(ivGeom, sizeGeom, rgbGeom);
fprintf('Cubi materiale reale: %d, facce visibili dopo culling: %d\n', ...
    size(ivGeom,1), size(facesReal,1));

vertsInf = zeros(0,3); facesInf = zeros(0,4); colorsInf = zeros(0,3);
if nInferred > 0
    % ivInferred e' gia' in celle fini (sez. 5): una lamina spessa un solo
    % voxel sul piano della parete, non un blocco da 20 cm -- qui resta solo
    % da costruirne la geometria.
    rgbInf = repmat(glassInferredColor, nInferred, 1);
    [vertsInf, facesInf, colorsInf] = buildCubePatch(ivInferred, glassInferredCubeSize, ...
        rgbInf, glassInferredShrink);
    fprintf('Cubi vetro inferito: %d celle (%.0f cm, shrink %.1f), facce: %d\n', ...
        nInferred, glassInferredCubeSize*100, glassInferredShrink, size(facesInf,1));
end

%% 9. Figura
fig = figure('Name', sprintf('Materiale voxel - %d voxel dati (%d cubi, %d vetro inferito)', ...
             nVoxCsv, size(ivGeom,1), nInferred), 'Color', 'k');
if cubeEdges
    edgeArg = {'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.1};
else
    edgeArg = {'EdgeColor', 'none'};
end
hold on
S.patchH = patch('Vertices', vertsReal, 'Faces', facesReal, ...
    'FaceVertexCData', colorsReal, 'FaceColor', 'flat', ...
    'FaceAlpha', cubeAlpha, edgeArg{:});
S.patchInfH = patch('Vertices', vertsInf, 'Faces', facesInf, ...
    'FaceVertexCData', colorsInf, 'FaceColor', 'flat', ...
    'FaceAlpha', cubeAlpha, 'Visible', ternary(nInferred > 0, 'on', 'off'), edgeArg{:});
hold off

ax = gca;
ax.Color = 'k';
ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
if enableFineVoxels
    geomTag = sprintf('cubi %.0f cm da punti LiDAR reali, materiale ereditato dal voxel dati %.0f cm', ...
        sizeGeom*100, voxelSize*100);
else
    geomTag = sprintf('cubi %.0f cm = voxel dati', voxelSize*100);
end
title(sprintf('Materiale di consenso per voxel -- %d voxel dati, %d cubi, %d vetro inferito (%s)%s', ...
    nVoxCsv, size(ivGeom,1), nInferred, geomTag, ternary(useRoomBBox, ', filtro stanza attivo', '')), ...
    'Color', 'w', 'Interpreter', 'none');
axis equal; grid on; view(3);
% axis equal, su un corridoio cosi' sproporzionato (~24 m in X contro ~2 m
% in Y/Z) in vista 3D, spesso allarga i LIMITI degli assi corti (non i
% dati) per riempire un riquadro piu' cubico -- risultato: tacche fino a
% +-20 m pur non essendoci alcun dato oltre ~2 m. Fissare i limiti sui
% bound reali dei vertici disegnati sovrascrive quel comportamento.
allVerts = [vertsReal; vertsInf];
if ~isempty(allVerts)
    pad = 2;     % m, margine visivo sopra/sotto i dati, su ogni asse
    xlim([min(allVerts(:,1))-pad max(allVerts(:,1))+pad]);
    ylim([min(allVerts(:,2))-pad max(allVerts(:,2))+pad]);
    zlim([min(allVerts(:,3))-pad max(allVerts(:,3))+pad]);
end
camlight headlight; lighting none;   % niente shading: il colore e' il dato, non la luce

S.ax = ax;
S.colorMap = colorMap;
S.matNames = matNames;
S.counts = counts;
S.nInferred = nInferred;
S.glassInferredColor = glassInferredColor;
S.legendOn = true;
S.edgesOn = cubeEdges;
S.inferredOn = nInferred > 0;
guidata(fig, S);
set(fig, 'WindowKeyPressFcn', @(src, ev) onKey(src, ev));

drawLegend(fig);

end % ShowVoxelMaterial3D


%% ------------------------------------------------------------------ %%
function [vertsAll, faces, faceColors] = buildCubePatch(iv, sizeVox, rgb, shrink)
%BUILDCUBEPATCH Geometria di un set di cubi allineati a griglia.
% shrink (default 1) = frazione della cella occupata da ogni cubo, centrata
% nella cella: 1 = pieno (i cubi si toccano, le facce condivise fra due
% voxel entrambi occupati vengono scartate -- stesso idioma di
% Piano1_CorridoioLungo.m sez. 10). shrink < 1 lascia un vuoto fra un cubo
% e l'altro: in quel caso i cubi non si toccano davvero, quindi il culling
% viene disattivato e si disegnano tutte le facce di ogni cubo.
if nargin < 4
    shrink = 1;
end
n = size(iv, 1);
if n == 0
    vertsAll = zeros(0,3); faces = zeros(0,4); faceColors = zeros(0,3);
    return
end
cornerOffsets = [0 0 0; 1 0 0; 1 1 0; 0 1 0; ...
                 0 0 1; 1 0 1; 1 1 1; 0 1 1];
if shrink < 1
    cornerOffsets = 0.5 + (cornerOffsets - 0.5) * shrink;
end
faceDefs = [1 2 3 4;   % -Z
            5 6 7 8;   % +Z
            1 2 6 5;   % -Y
            4 3 7 8;   % +Y
            1 4 8 5;   % -X
            2 3 7 6];  % +X
faceNeighborDir = [0 0 -1; 0 0 1; 0 -1 0; 0 1 0; -1 0 0; 1 0 0];

vertsAll = zeros(n * 8, 3);
for c = 1:8
    vertsAll(c:8:end, :) = (iv + cornerOffsets(c,:)) * sizeVox;
end

baseIdx = (0:n-1)' * 8;
facesVis = cell(6, 1);
colorVis = cell(6, 1);
for f = 1:6
    if shrink < 1
        keep = true(n, 1);   % i cubi non si toccano: nessuna faccia nascosta
    else
        hidden = ismember(iv + faceNeighborDir(f,:), iv, 'rows');
        keep = ~hidden;
    end
    facesVis{f} = baseIdx(keep) + faceDefs(f,:);
    colorVis{f} = rgb(keep, :);
end
faces = vertcat(facesVis{:});
faceColors = vertcat(colorVis{:});
end


%% ------------------------------------------------------------------ %%
function p = findPlane(planes, planeId)
%FINDPLANE Struct del piano con id planeId in planes.json, [] se assente.
pl = planes.planes;
if ~iscell(pl), pl = num2cell(pl); end
p = [];
for i = 1:numel(pl)
    if double(pl{i}.id) == planeId
        p = pl{i};
        return
    end
end
end


%% ------------------------------------------------------------------ %%
function [constAxis, ax1, ax2] = planeAxes(normal)
%PLANEAXES Asse "di spessore" (quello piu' allineato con la normale, dove
% la parete e' sottile di un solo voxel) e i due assi "nel piano" (dove va
% costruita la griglia). Funziona per pareti allineate agli assi (questa
% sessione: normal = [0,1,0] per nord/sud, [1,0,0] per la testa) -- una
% parete non allineata richiederebbe basis_u/basis_v invece degli indici
% x,y,z diretti.
[~, constAxis] = max(abs(normal));
inPlane = setdiff(1:3, constAxis);
ax1 = inPlane(1);
ax2 = inPlane(2);
end


%% ------------------------------------------------------------------ %%
function drawLegend(fig)
%DRAWLEGEND Un quadratino colorato per materiale presente (voxel dati) piu'
% una voce separata per il vetro inferito, con i rispettivi conteggi.
S = guidata(fig);
delete(findobj(fig, 'Tag', 'materialLegend'));
if ~S.legendOn
    return
end
hold(S.ax, 'on');
nMat = numel(S.matNames);
extra = S.nInferred > 0;
h = gobjects(nMat + extra, 1);
labels = strings(nMat + extra, 1);
for i = 1:nMat
    col = materialColor(S.colorMap, S.matNames(i));
    h(i) = plot(S.ax, NaN, NaN, 's', 'MarkerFaceColor', col, ...
        'MarkerEdgeColor', 'none', 'MarkerSize', 12);
    labels(i) = sprintf('%s (%d)', S.matNames(i), S.counts(i));
end
if extra
    h(end) = plot(S.ax, NaN, NaN, 's', 'MarkerFaceColor', S.glassInferredColor, ...
        'MarkerEdgeColor', 'none', 'MarkerSize', 12);
    labels(end) = sprintf('glass inferito, buco (%d)', S.nInferred);
end
hold(S.ax, 'off');
lg = legend(h, labels, 'TextColor', 'w', 'Color', [0.15 0.15 0.15], ...
    'Location', 'eastoutside', 'Interpreter', 'none');
lg.Tag = 'materialLegend';
end


%% ------------------------------------------------------------------ %%
function onKey(fig, ev)
S = guidata(fig);
switch ev.Key
    case 'm'
        S.legendOn = ~S.legendOn;
        guidata(fig, S);
        drawLegend(fig);
    case 'g'
        S.edgesOn = ~S.edgesOn;
        if S.edgesOn
            set([S.patchH S.patchInfH], 'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.1);
        else
            set([S.patchH S.patchInfH], 'EdgeColor', 'none');
        end
        guidata(fig, S);
    case 'i'
        if S.nInferred > 0
            S.inferredOn = ~S.inferredOn;
            set(S.patchInfH, 'Visible', ternary(S.inferredOn, 'on', 'off'));
            guidata(fig, S);
        end
end
end


%% ------------------------------------------------------------------ %%
function m = materialColors()
%MATERIALCOLORS Stessa palette di ShowMaterialConsensus.m in questa stessa
% cartella, cosi' i due visualizzatori restano confrontabili.
names = {'concrete', 'painted_metal', 'glass', 'paint', 'rubber', 'fabric', ...
         'plastic', 'plaster', 'brick', 'ceramic', 'wood', 'asphalt', ...
         'cardboard', 'steel_oxidized', 'iron_rusted', 'copper_oxidized'};
rgb = [ 70 130 230; 240 120  40;  60 210 200; 200 100 220; 240 220  60; ...
       230  70 120; 120 230  90; 150 200 255; 220  90  60; 170 170 250; ...
       200 160  90; 110 110 130; 210 180 140; 190 190 190; 200 120  80; ...
       120 200 160] / 255;
m = containers.Map(names, num2cell(rgb, 2)');
end


function col = materialColor(map, name)
name = char(name);
if isKey(map, name)
    col = map(name);
else
    col = [0.5 0.5 0.5];        % materiale non previsto: grigio
end
end


function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
