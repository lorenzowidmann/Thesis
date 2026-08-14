%% Segmentazione iterativa di piani (ground + pareti) su point cloud SLAM
% Implementa la pipeline di piani del paper "3D Radiometric Mapping by Means
% of LiDAR SLAM and Thermal Camera Data Fusion" (De Pazzi, Chiodini, Pertile,
% Sensors 2022): rimozione del ground plane per primo (vincolo di normale
% verticale), poi ricerca iterativa MSAC/RANSAC del piano con piu' inlier tra
% i punti rimanenti (pareti ed eventuale soffitto), isolamento di quel piano,
% ripetizione fino a maxPlanes piani o finche' il piano trovato non ha
% abbastanza inlier. I punti che non cadono in nessun piano trovato non
% vengono scartati silenziosamente: restano come outlier residui, salvati in
% una point cloud separata (sez. 5) per verificare cosa lo script sta
% escludendo (porte/infissi incassati potrebbero finire li').
%
% Standalone: per cambiare bag basta cambiare bagPath qui sotto, nessun
% altro cambiamento al codice. loadAccumulatedCloud riusa la stessa logica
% di lettura di Piano1_CorridoioLungo.m (sez. 2-4), riscritta come funzione
% locale riusabile.

clear
close all
clc

%% 1. Parametri
bagPath   = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20';
topicName = '/cloud_registered';

maxPlanes = 10;    % numero massimo di piani da estrarre (pavimento + soffitto + pareti)

% Soglia di distanza inlier RANSAC/MSAC (m). Il rumore geometrico noto di
% questa pipeline e' l'errore composto di calibrazione LiDAR<->camera, ~9 cm
% RMSE (5.8 cm lidar->flir + 6.8 cm lidar->zed, vedi rig_calibration.yaml e
% il commento in Piano1_CorridoioLungo.m sez. 9, PRIMA della deriva SLAM che
% si somma a parte): una soglia molto piu' fine di quell'ordine di grandezza
% scarterebbe come "fuori piano" punti che sono in realta' rumore di misura,
% non struttura reale. Si parte quindi a meta' dell'intervallo 5-8 cm, non
% sotto.
ransacMaxDistance  = 0.06;   % m
ransacMaxNumTrials = 1000;   % iterazioni RANSAC/MSAC per ogni chiamata a pcfitplane

% Frazione minima di punti RIMANENTI (ricalcolata a ogni iterazione, NON sul
% totale iniziale) perche' un piano sia accettato: evita che un cluster
% piccolo di rumore residuo, verso la fine del loop quando i punti rimasti
% sono pochi, venga scambiato per un piano vero.
minInlierFraction  = 0.02;   % 2% dei punti rimanenti a quel giro
minPointsToAttempt = 200;    % soglia assoluta: sotto questa non si tenta nemmeno il fit

% Vincolo di normale per il SOLO ground plane (sez. 3): il pavimento e'
% l'unico piano assunto perpendicolare all'asse verticale Z, quindi va
% isolato per primo con un vincolo di normale. Il loop pareti (sez. 4) non
% ha un vincolo analogo: dopo il pavimento, il paper cerca semplicemente il
% piano con piu' inlier tra i punti rimanenti a ogni iterazione (in genere le
% pareti; un eventuale soffitto, pur con normale verticale come il
% pavimento, e' comunque un piano DIVERSO e viene trovato come "il prossimo
% con piu' inlier", senza bisogno di un vincolo dedicato).
groundReferenceVector    = [0 0 1];
groundMaxAngularDistance = 20;   % gradi, tolleranza sull'inclinazione del pavimento

% Filtro automatico "corridoio principale" (sez. 2b): tiene solo il cluster
% euclideo con piu' punti, scarta cio' che e' geometricamente scollegato dal
% corridoio (una stanza intravista da una porta aperta, drift isolato,
% riflessi multi-path) senza un box ROI manuale da ritarare per ogni bag.
useCorridorFilter = true;   % false = disattiva, si segmenta pcRaw cosi' com'e'

corridorClusterVoxelSize   = 0.10;   % m, risoluzione SOLO per decidere il cluster (i
                                      % punti in output restano quelli originali, sez. 2b)
corridorClusterMinDistance = 1.8 * corridorClusterVoxelSize;   % m, connette voxel
                                      % occupati fino alla diagonale (sqrt(3) =~ 1.73*voxel);
                                      % un vuoto reale maggiore di questo separa i cluster

% Voxelizzazione octree dell'output pulito (sez. 5b): stesso principio di
% PointCloudElaboration/OcTree (voxelize_octree in octree/voxelizer.py) -
% cubo radice centrato sul bounding box, voxel = lato_cubo / 2^depth. Serve
% solo a "ordinare" l'output su una griglia regolare (un punto per voxel
% occupato) invece dei punti grezzi sparsi; non e' un filtro, non cambia
% quali piani/punti sono stati accettati.
useOctreeVoxelize = true;   % false = disattiva, pcClean resta ai punti grezzi
octreeVoxelSize = 0.20;     % m, lato del voxel. L'octree "vero" (voxelize_octree,
                             % octree/voxelizer.py) quantizza a estensione_cubo_radice/2^depth,
                             % non a una taglia diretta: qui si tiene la STESSA indicizzazione a
                             % cubo radice centrato (stesso principio), ma la taglia e' diretta,
                             % non vincolata a una potenza di 2. 20 cm = stesso passo di
                             % EmissivityCalculation/voxel_consensus.py (sez. 5c, join termico),
                             % anche se l'origine delle due griglie resta diversa.

% Join spaziale con la temperatura corretta di EmissivityCalculation (sez. 5c):
% thermal_voxels_u.csv e' output REALE di voxel_consensus.py --stage thermal
% (non sintetico: quello finisce sotto demo_data/), stesso frame fisico
% (--odom-topic default /cloud_registered, come qui sopra) ma griglia
% ancorata al mondo (0,0,0) passo 0.20 m fisso, DIVERSA dalla nostra
% (centrata sul bounding box di cleanXYZ): gli ID voxel non corrispondono
% per costruzione, va fatto un join per posizione reale, non per indice.
useThermalJoin = true;
thisFileDir    = fileparts(mfilename('fullpath'));
thermalCsvPath = fullfile(thisFileDir, '..', 'EmissivityCalculation', 'thermal_voxels_u.csv');

markerSize = 20;   % dimensione dei punti a schermo (pcshow), stile coerente con gli altri script

%% 2. Caricamento bag
pcRaw = loadAccumulatedCloud(bagPath, topicName);
nRawPoints = pcRaw.Count;
fprintf('\nPoint cloud accumulata: %d punti\n', nRawPoints);

%% 2b. Filtro automatico "corridoio principale" (clustering euclideo)
% pcsegdist su 4-5M di punti e' improponibile: si clusterizza sui VOXEL
% occupati (risoluzione corridorClusterVoxelSize), non sui singoli punti, poi
% il risultato si ridistribuisce ("broadcast") a ogni punto originale con lo
% stesso idioma floor+unique+ic gia' usato per il pooling statistico in
% Piano1_CorridoioLungo.m (sez. 9): ogni punto eredita il cluster del suo
% voxel, la densita' vera dei punti resta intatta in output.
if useCorridorFilter
    voxIdx = floor(pcRaw.Location / corridorClusterVoxelSize);
    [voxU, ~, ic] = unique(voxIdx, 'rows');
    voxXYZ = voxU * corridorClusterVoxelSize;

    [voxLabels, numClusters] = pcsegdist(pointCloud(voxXYZ), corridorClusterMinDistance);
    voxCounts = accumarray(voxLabels, 1);
    [mainVoxCount, mainLabel] = max(voxCounts);

    pointLabels = voxLabels(ic);   % broadcast: un'etichetta di cluster per ogni punto originale
    corridorIdx = find(pointLabels == mainLabel);

    fprintf('\n--- Filtro corridoio principale (clustering euclideo, voxel %.0f cm, gap %.0f cm) ---\n', ...
        corridorClusterVoxelSize*100, corridorClusterMinDistance*100);
    fprintf('Cluster trovati: %d (su %d voxel occupati)\n', numClusters, size(voxU, 1));
    fprintf('Cluster principale (corridoio): %d voxel (%.1f%%)\n', ...
        mainVoxCount, 100*mainVoxCount/size(voxU, 1));

    pcOutsideCorridor = select(pcRaw, find(pointLabels ~= mainLabel));
    pcWork = select(pcRaw, corridorIdx);
    fprintf('Punti nel corridoio principale: %d (%.1f%%), scartati fuori corridoio: %d (%.1f%%)\n', ...
        pcWork.Count, 100*pcWork.Count/nRawPoints, ...
        pcOutsideCorridor.Count, 100*pcOutsideCorridor.Count/nRawPoints);
else
    pcOutsideCorridor = select(pcRaw, []);
    pcWork = pcRaw;   % nuvola di lavoro: si svuota progressivamente man mano che i piani vengono isolati
end

planeXYZ          = cell(maxPlanes, 1);
planeIdOf         = cell(maxPlanes, 1);
planeNormals      = nan(maxPlanes, 3);
planeInlierCounts = zeros(maxPlanes, 1);
planeLabels       = strings(maxPlanes, 1);
nPlanesFound      = 0;

%% 3. Step 1 - rimozione ground plane (vincolo di normale verticale)
fprintf('\n--- Step 1: ground plane (vincolo normale verticale, tolleranza %d gradi) ---\n', ...
    groundMaxAngularDistance);

[groundModel, groundInlierIdx, ~, groundRmse] = pcfitplane(pcWork, ...
    ransacMaxDistance, groundReferenceVector, groundMaxAngularDistance, ...
    'MaxNumTrials', ransacMaxNumTrials);

if numel(groundInlierIdx) < minPointsToAttempt
    warning(['Nessun ground plane trovato entro %d gradi dalla verticale ' ...
        '(%d inlier, sotto la soglia minima %d). Il pavimento non viene ' ...
        'rimosso separatamente: si procede direttamente con il loop pareti ' ...
        '(sez. 4), che potrebbe trovarlo comunque come piano generico.'], ...
        groundMaxAngularDistance, numel(groundInlierIdx), minPointsToAttempt);
else
    nPlanesFound = nPlanesFound + 1;
    planeXYZ{nPlanesFound}          = pcWork.Location(groundInlierIdx, :);
    planeIdOf{nPlanesFound}         = repmat(nPlanesFound, numel(groundInlierIdx), 1);
    planeNormals(nPlanesFound, :)   = groundModel.Normal;
    planeInlierCounts(nPlanesFound) = numel(groundInlierIdx);
    planeLabels(nPlanesFound)       = "ground";

    fprintf('Ground plane: %d inlier (%.1f%% dei %d punti), normale = [%.3f %.3f %.3f], RMSE = %.3f m\n', ...
        numel(groundInlierIdx), 100*numel(groundInlierIdx)/pcWork.Count, pcWork.Count, ...
        groundModel.Normal, groundRmse);

    keepMask = true(pcWork.Count, 1);
    keepMask(groundInlierIdx) = false;
    pcWork = select(pcWork, find(keepMask));
end

%% 4. Step 2 - loop iterativo sulle pareti (piano con piu' inlier a ogni giro)
fprintf('\n--- Step 2: loop iterativo pareti/soffitto (fino a %d piani totali) ---\n', maxPlanes);

while nPlanesFound < maxPlanes
    remaining = pcWork.Count;

    if remaining < minPointsToAttempt
        fprintf('Punti rimanenti (%d) sotto la soglia minima (%d): stop loop.\n', ...
            remaining, minPointsToAttempt);
        break
    end

    minInliers = max(minPointsToAttempt, ceil(minInlierFraction * remaining));

    [model, inlierIdx, ~, rmse] = pcfitplane(pcWork, ransacMaxDistance, ...
        'MaxNumTrials', ransacMaxNumTrials);

    if numel(inlierIdx) < minInliers
        fprintf(['Piano candidato scartato: %d inlier < soglia minima %d ' ...
            '(%.0f%% di %d punti rimanenti). Stop loop, il resto e'' outlier residuo.\n'], ...
            numel(inlierIdx), minInliers, 100*minInlierFraction, remaining);
        break
    end

    nPlanesFound = nPlanesFound + 1;
    planeXYZ{nPlanesFound}          = pcWork.Location(inlierIdx, :);
    planeIdOf{nPlanesFound}         = repmat(nPlanesFound, numel(inlierIdx), 1);
    planeNormals(nPlanesFound, :)   = model.Normal;
    planeInlierCounts(nPlanesFound) = numel(inlierIdx);
    planeLabels(nPlanesFound)       = sprintf('piano_%d', nPlanesFound);

    fprintf('Piano %d: %d inlier (%.1f%% di %d punti rimanenti), normale = [%.3f %.3f %.3f], RMSE = %.3f m\n', ...
        nPlanesFound, numel(inlierIdx), 100*numel(inlierIdx)/remaining, remaining, ...
        model.Normal, rmse);

    keepMask = true(pcWork.Count, 1);
    keepMask(inlierIdx) = false;
    pcWork = select(pcWork, find(keepMask));
end

if nPlanesFound == maxPlanes
    fprintf('Raggiunto il numero massimo di piani (%d).\n', maxPlanes);
end

%% 5. Costruzione output: nuvola pulita (per piano) + outlier residui
planeXYZ          = planeXYZ(1:nPlanesFound);
planeIdOf         = planeIdOf(1:nPlanesFound);
planeNormals      = planeNormals(1:nPlanesFound, :);
planeInlierCounts = planeInlierCounts(1:nPlanesFound);
planeLabels       = planeLabels(1:nPlanesFound);

cleanXYZ     = vertcat(planeXYZ{:});
cleanPlaneId = vertcat(planeIdOf{:});   % un plane_id per ogni punto di cleanXYZ, stesso ordine di scoperta dei piani

pcClean    = pointCloud(cleanXYZ);
pcOutliers = pcWork;   % quello che resta dopo tutte le rimozioni = outlier residuo, non buttato

nClean    = pcClean.Count;
nOutliers = pcOutliers.Count;

fprintf('\n--- Riepilogo ---\n');
fprintf('Punti iniziali:              %d\n', nRawPoints);
fprintf('Scartati fuori corridoio:    %d (%.1f%%)\n', ...
    pcOutsideCorridor.Count, 100*pcOutsideCorridor.Count/nRawPoints);
for k = 1:nPlanesFound
    fprintf('  Piano %d (%s): %d punti\n', k, planeLabels(k), planeInlierCounts(k));
end
fprintf('Punti puliti (piani):         %d (%.1f%%)\n', nClean, 100*nClean/nRawPoints);
fprintf('Scartati come outlier RANSAC: %d (%.1f%%)\n', nOutliers, 100*nOutliers/nRawPoints);

%% 5b. Voxelizzazione octree dell'output pulito
% Stessa indicizzazione di voxelize_octree (PointCloudElaboration/OcTree/octree/voxelizer.py):
% cubo radice centrato sul bounding box di cleanXYZ, lato = dimensione massima
% del box (non solo un box, un CUBO: stesso passo su tutti e 3 gli assi),
% indice intero floor((xyz-origine)/octreeVoxelSize). Il rappresentante di
% ogni voxel occupato e' il suo plane_id di MAGGIORANZA tra i punti caduti
% dentro: voto vettorizzato per piano (accumarray colonna per colonna,
% nPlanesFound e' piccolo) invece di un mode() per voxel, che su centinaia
% di migliaia di voxel sarebbe lentissimo. Disegnati come cubi veri in
% Figura 3 (sez. 6), non come punti al centro.
if useOctreeVoxelize
    octLo = min(cleanXYZ, [], 1);
    octHi = max(cleanXYZ, [], 1);
    octreeOrigin = (octLo + octHi)/2 - max(octHi - octLo)/2;   % min-corner del cubo radice
    octreeRootExtent = max(octHi - octLo);

    octVoxIdx = floor((cleanXYZ - octreeOrigin) / octreeVoxelSize);
    [octVoxU, ~, icOct] = unique(octVoxIdx, 'rows');

    pcCleanVox = (octVoxU + 0.5) * octreeVoxelSize + octreeOrigin;   % centro di ogni voxel occupato
    nVoxClean  = size(pcCleanVox, 1);

    planeVoteCounts = zeros(nVoxClean, nPlanesFound);
    for p = 1:nPlanesFound
        planeVoteCounts(:, p) = accumarray(icOct(cleanPlaneId == p), 1, [nVoxClean, 1]);
    end
    [~, voxPlaneId] = max(planeVoteCounts, [], 2);

    fprintf('\n--- Voxelizzazione octree (voxel %.1f cm, cubo radice %.2f m) ---\n', ...
        octreeVoxelSize*100, octreeRootExtent);
    fprintf('Punti puliti: %d -> voxel occupati: %d (riduzione %.1f%%)\n', ...
        nClean, nVoxClean, 100*(1 - nVoxClean/nClean));
end

%% 5c. Join spaziale: temperatura corretta (thermal_voxels_u.csv) sui voxel octree
% Le due griglie non condividono origine (vedi parametri, sez. 1): si
% ricalcola l'indice octree di ogni punto del CSV con la STESSA formula
% (origine e passo) usata sopra per cleanXYZ, poi si fa ismember contro i
% voxel GIA' occupati (octVoxU) - un punto CSV "cade" in un voxel occupato
% solo se la sua posizione reale ci sta davvero dentro, indipendentemente da
% come i due sistemi numerano le celle. Piu' voxel CSV cadono nello stesso
% voxel octree (es. per via del passo diverso, 20 cm vs 20 cm qui ma origini
% sfalsate) vengono mediati.
voxTemperature = [];
hasT = [];
if useThermalJoin && useOctreeVoxelize
    if ~isfile(thermalCsvPath)
        warning('CSV termico non trovato: %s. Join saltato.', thermalCsvPath);
    else
        thermalT = readtable(thermalCsvPath);
        nCsvRaw = height(thermalT);

        if ismember('plausible', thermalT.Properties.VariableNames)
            thermalT = thermalT(thermalT.plausible == 1, :);
        end

        fprintf('\n--- Join temperatura corretta (%s) sui voxel octree ---\n', thermalCsvPath);
        fprintf('Righe CSV: %d, plausibili: %d\n', nCsvRaw, height(thermalT));

        csvXYZ = [thermalT.x, thermalT.y, thermalT.z];
        csvVoxIdx = floor((csvXYZ - octreeOrigin) / octreeVoxelSize);

        [tfCsv, locInOct] = ismember(csvVoxIdx, octVoxU, 'rows');
        fprintf('Punti CSV dentro un voxel octree occupato: %d / %d (%.1f%%)\n', ...
            sum(tfCsv), height(thermalT), 100*sum(tfCsv)/height(thermalT));

        matchedVox = locInOct(tfCsv);
        matchedT   = thermalT.t_mean_c(tfCsv);
        sumT = accumarray(matchedVox, matchedT, [nVoxClean, 1]);
        cntT = accumarray(matchedVox, 1, [nVoxClean, 1]);

        hasT = cntT > 0;
        voxTemperature = nan(nVoxClean, 1);
        voxTemperature(hasT) = sumT(hasT) ./ cntT(hasT);

        fprintf('Voxel octree con temperatura assegnata: %d / %d (%.1f%%)\n', ...
            sum(hasT), nVoxClean, 100*sum(hasT)/nVoxClean);
    end
end

%% 6. Visualizzazione

% --- Figura 1: piani colorati per plane_id ---
figure('Name', sprintf('Segmentazione piani - %d piani trovati - %d punti puliti', ...
    nPlanesFound, nClean), 'Color', 'k');
pcshow(cleanXYZ, cleanPlaneId, 'MarkerSize', markerSize);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d piani (ground + pareti/soffitto), %d punti puliti', nPlanesFound, nClean), ...
    'Color', 'w');
axis equal; grid on;
colormap(gca, lines(nPlanesFound));
clim([0.5, nPlanesFound + 0.5]);
cb = colorbar;
cb.Color = 'w';
cb.Ticks = 1:nPlanesFound;
cb.TickLabels = arrayfun(@(k) sprintf('%d: %s', k, planeLabels(k)), 1:nPlanesFound, 'UniformOutput', false);
cb.Label.String = 'plane_id';
cb.Label.Color = 'w';

% --- Figura 2: grezza vs pulita, stessi assi per confronto diretto ---
xl = pcRaw.XLimits; yl = pcRaw.YLimits; zl = pcRaw.ZLimits;

figure('Name', 'Grezza vs pulita (solo inlier dei piani)', 'Color', 'k', ...
    'Position', [100 100 1400 600]);

subplot(1, 2, 1);
pcshow(pcRaw, 'MarkerSize', markerSize);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Grezza: %d punti', nRawPoints), 'Color', 'w');
axis equal; grid on;
xlim(xl); ylim(yl); zlim(zl);
colormap(gca, turbo);

subplot(1, 2, 2);
pcshow(pcClean, 'MarkerSize', markerSize);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Pulita (solo piani): %d punti (%.1f%%)', nClean, 100*nClean/nRawPoints), 'Color', 'w');
axis equal; grid on;
xlim(xl); ylim(yl); zlim(zl);
colormap(gca, turbo);

% --- Figura 3: pulita voxelizzata octree, cubi veri (non punti al centro) ---
% Stesso idioma geometrico di Piano1_CorridoioLungo.m sez. 10 (8 vertici e 6
% facce per voxel, vettorizzato, con face culling verso i vicini occupati:
% le facce interne tra due voxel adiacenti entrambi pieni non si disegnano,
% servono sia per le prestazioni sia per non sommare visivamente facce
% nascoste sotto la trasparenza). Colore per plane_id di maggioranza (sez.
% 5b), stessa palette/scala di Figura 1.
if useOctreeVoxelize
    cubeAlpha = 1;       % 0 = invisibile, 1 = opaco
    cubeEdges = false;   % true = disegna gli spigoli (leggibile solo con pochi voxel)

    cornerOffsets = [0 0 0; 1 0 0; 1 1 0; 0 1 0; ...
                     0 0 1; 1 0 1; 1 1 1; 0 1 1];
    faceDefs = [1 2 3 4;   % -Z
                5 6 7 8;   % +Z
                1 2 6 5;   % -Y
                4 3 7 8;   % +Y
                1 4 8 5;   % -X
                2 3 7 6];  % +X
    faceNeighborDir = [0 0 -1; 0 0 1; 0 -1 0; 0 1 0; -1 0 0; 1 0 0];

    vertsAll = zeros(nVoxClean * 8, 3);
    for c = 1:8
        vertsAll(c:8:end, :) = (octVoxU + cornerOffsets(c,:)) * octreeVoxelSize + octreeOrigin;
    end

    baseIdx = (0:nVoxClean-1)' * 8;
    facesVis = cell(6, 1);
    colorVis = cell(6, 1);
    for f = 1:6
        hidden = ismember(octVoxU + faceNeighborDir(f,:), octVoxU, 'rows');
        keep = ~hidden;
        facesVis{f} = baseIdx(keep) + faceDefs(f,:);
        colorVis{f} = voxPlaneId(keep);
    end
    facesOct = vertcat(facesVis{:});
    faceColorsOct = vertcat(colorVis{:});

    fprintf('Facce totali: %d, visibili dopo culling: %d (%.0f%% scartate)\n', ...
        nVoxClean*6, size(facesOct,1), 100*(1 - size(facesOct,1)/(nVoxClean*6)));

    figure('Name', sprintf('Voxelizzazione octree - %.0f cm - %d voxel', ...
        octreeVoxelSize*100, nVoxClean), 'Color', 'k');
    if cubeEdges
        edgeArg = {'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.1};
    else
        edgeArg = {'EdgeColor', 'none'};
    end
    patch('Vertices', vertsAll, 'Faces', facesOct, ...
        'FaceVertexCData', faceColorsOct, 'FaceColor', 'flat', ...
        'FaceAlpha', cubeAlpha, edgeArg{:});

    ax3 = gca;
    ax3.Color = 'k';
    ax3.XColor = 'w'; ax3.YColor = 'w'; ax3.ZColor = 'w';
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Cubi octree %.0f cm: %d voxel occupati (da %d punti puliti)', ...
        octreeVoxelSize*100, nVoxClean, nClean), 'Color', 'w');
    axis equal; grid on; view(3);
    colormap(ax3, lines(nPlanesFound));
    clim(ax3, [0.5, nPlanesFound + 0.5]);
    cb3 = colorbar;
    cb3.Color = 'w';
    cb3.Ticks = 1:nPlanesFound;
    cb3.TickLabels = arrayfun(@(k) sprintf('%d: %s', k, planeLabels(k)), 1:nPlanesFound, 'UniformOutput', false);
    cb3.Label.String = 'plane_id';
    cb3.Label.Color = 'w';
    camlight headlight; lighting none;   % niente shading: il colore e' il dato, non la luce
end

% --- Figura 4: voxel octree con temperatura corretta (join sez. 5c), cubi veri ---
% Stessa geometria/culling di Figura 3, ma costruita SOLO sul sottoinsieme di
% voxel che hanno ricevuto una temperatura dal join (hasT) - stesso idioma di
% Piano1_CorridoioLungo.m sez. 10 (li' validT), il culling considera occupato
% solo quel sottoinsieme, non tutti gli nVoxClean voxel.
if useThermalJoin && useOctreeVoxelize && any(hasT)
    voxIdxT = octVoxU(hasT, :);
    tempT   = voxTemperature(hasT);
    nVoxT   = size(voxIdxT, 1);

    vertsT = zeros(nVoxT * 8, 3);
    for c = 1:8
        vertsT(c:8:end, :) = (voxIdxT + cornerOffsets(c,:)) * octreeVoxelSize + octreeOrigin;
    end

    baseIdxT = (0:nVoxT-1)' * 8;
    facesVisT = cell(6, 1);
    colorVisT = cell(6, 1);
    for f = 1:6
        hiddenT = ismember(voxIdxT + faceNeighborDir(f,:), voxIdxT, 'rows');
        keepT = ~hiddenT;
        facesVisT{f} = baseIdxT(keepT) + faceDefs(f,:);
        colorVisT{f} = tempT(keepT);
    end
    facesT = vertcat(facesVisT{:});
    faceColorsT = vertcat(colorVisT{:});

    figure('Name', sprintf('Temperatura corretta sui voxel octree - %.0f cm - %d voxel', ...
        octreeVoxelSize*100, nVoxT), 'Color', 'k');
    patch('Vertices', vertsT, 'Faces', facesT, ...
        'FaceVertexCData', faceColorsT, 'FaceColor', 'flat', ...
        'FaceAlpha', cubeAlpha, edgeArg{:});

    ax4 = gca;
    ax4.Color = 'k';
    ax4.XColor = 'w'; ax4.YColor = 'w'; ax4.ZColor = 'w';
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Temperatura corretta (thermal\\_voxels\\_u.csv) su %d/%d voxel octree %.0f cm', ...
        nVoxT, nVoxClean, octreeVoxelSize*100), 'Color', 'w', 'Interpreter', 'none');
    axis equal; grid on; view(3);
    colormap(ax4, hot);
    cb4 = colorbar;
    cb4.Color = 'w';
    cb4.Label.String = 'Temperatura corretta (°C)';
    cb4.Label.Color = 'w';
    camlight headlight; lighting none;

    fprintf('Temperatura sui voxel octree mostrati: min=%.1f  media=%.1f  max=%.1f\n', ...
        min(tempT), mean(tempT), max(tempT));
elseif useThermalJoin && useOctreeVoxelize
    warning('Nessun voxel octree con temperatura valida dal join: Figura 4 saltata.');
end

%% --- Funzioni locali ---

function pc = loadAccumulatedCloud(bagPath, topicName)
% Stessa logica di lettura di Piano1_CorridoioLungo.m (sez. 2-4): apre la
% bag, legge tutti i messaggi sul topic, concatena i punti e scarta i
% ritorni non validi (NaN/Inf). Riscritta qui come funzione locale
% riusabile: per cambiare bag basta cambiare bagPath in testa allo script,
% questa funzione resta invariata. Nessun ROI/denoise qui: in questo script
% e' la segmentazione RANSAC stessa a fare da filtro (sez. 3-5).
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

    msgs = readMessages(sel, 1:nTotal);

    % Preallocazione in cell array e vertcat finale: molto piu' veloce che
    % concatenare dentro il ciclo, dove la matrice verrebbe riallocata a ogni giro.
    allXYZ = cell(numel(msgs), 1);
    for i = 1:numel(msgs)
        allXYZ{i} = rosReadXYZ(msgs{i});
    end
    xyz = vertcat(allXYZ{:});

    nRaw = size(xyz, 1);
    xyz = xyz(all(isfinite(xyz), 2), :);
    fprintf('Punti letti: %d, validi: %d (scartati %d)\n', ...
        nRaw, size(xyz,1), nRaw - size(xyz,1));

    pc = pointCloud(xyz);
end
