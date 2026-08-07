function ShowTemperatureCorrection(sessionDirIn, flirDirIn)
%% Confronto interattivo temperatura apparente / corretta (sessione 9)
% Mostra il frame FLIR della sessione e, puntando col mouse su un pixel,
% legge la temperatura PRIMA della correzione radiometrica (il .npy grezzo
% della camera, temperatura apparente) e DOPO (corrected_temperature.npy
% prodotto da RadiometricCalibration\correct_session.py), con la differenza.
%
% Stessa idea di RadiometricCalibration\ThermalData.py --show (hover per
% leggere il valore sotto il cursore), ma qui i valori sono due e affiancati,
% e si puo' scorrere tutta la sessione con le frecce.
%
% Oltre alla temperatura, per il pixel puntato vengono mostrati anche i dati
% che hanno determinato la correzione: emissivita' applicata, distanza LiDAR,
% materiale scelto da CLIP e se quel pixel era un campione LiDAR diretto
% oppure riempito per vicinanza (sampled_mask). Serve a capire subito se un
% valore strano e' misurato o interpolato.
%
% Il terzo pannello mostra il frame ZED da cui tutto questo nasce: e' li' che
% SLIC ritaglia i superpixel e CLIP classifica ogni ritaglio, quindi e' la
% sorgente dell'emissivita' poi riproiettata su FLIR. Puntando un pixel FLIR
% si evidenzia sul ZED il superpixel corrispondente (bbox = il crop visto da
% CLIP, croce = centroide).
%
% Comandi da tastiera:
%   freccia dx / sx   frame successivo / precedente
%   pag su / pag giu  avanti / indietro di 10 frame
%   home / fine       primo / ultimo frame
%   c                 blocca (o sblocca) la lettura sul punto cliccato
%   s                 mostra / nasconde i campioni LiDAR diretti
%   b                 mostra / nasconde i bordi dei superpixel (segment_id)
%   l                 scala colore fissa su tutta la sessione / per frame
%
% Uso:
%   ShowTemperatureCorrection                       % percorsi di default, qui sotto
%   ShowTemperatureCorrection(sessionDir)           % altra sessione ZED
%   ShowTemperatureCorrection(sessionDir, flirDir)

close all
clc

%% 1. Parametri
sessionDir = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate';
flirDir    = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Flir\session9_only_rot180';

% Gli argomenti, se passati, hanno la precedenza sui default qui sopra.
if nargin >= 1 && ~isempty(sessionDirIn)
    sessionDir = sessionDirIn;
end
if nargin >= 2 && ~isempty(flirDirIn)
    flirDir = flirDirIn;
end

emisDir     = fullfile(sessionDir, 'emissivity_map');
materialDir = fullfile(sessionDir, 'material_map');
framesDir   = fullfile(sessionDir, 'frames');   % PNG ZED sorgenti

if ~isfolder(emisDir)
    error('Cartella emissivity_map non trovata: %s', emisDir);
end
if ~isfolder(flirDir)
    error('Cartella FLIR non trovata: %s', flirDir);
end

%% 2. Elenco dei frame corretti
% Un frame e' utilizzabile solo se esistono sia il .npy apparente sia la
% correzione: correct_session.py salta i frame senza distance/segment_id.
d = dir(fullfile(emisDir, '*'));
d = d([d.isdir] & ~ismember({d.name}, {'.', '..'}));

frames = struct('stem', {}, 'apparentPath', {}, 'correctedPath', {}, 'dir', {});
for k = 1:numel(d)
    stem = d(k).name;                       % es. 20250906_233144_R
    frameDir = fullfile(emisDir, stem);
    corrPath = fullfile(frameDir, 'corrected_temperature.npy');

    % Il .npy FLIR grezzo non ha il suffisso _R del nome cartella.
    appPath = fullfile(flirDir, [strrep(stem, '_R', '') '.npy']);

    if isfile(corrPath) && isfile(appPath)
        frames(end+1) = struct('stem', stem, ...
                               'apparentPath', appPath, ...
                               'correctedPath', corrPath, ...
                               'dir', frameDir); %#ok<AGROW>
    end
end

if isempty(frames)
    error('Nessun frame con correzione trovato in %s', emisDir);
end
nFrames = numel(frames);
fprintf('%d frame corretti in %s\n', nFrames, emisDir);

%% 3. Scala colore comune a tutta la sessione
% Presa dai correction_report.json, cosi' non serve rileggere tutti i .npy:
% con una scala fissa il confronto fra frame e' onesto (un frame non sembra
% piu' caldo solo perche' riscalato su se stesso).
%
% Non si usano pero' il minimo e il massimo assoluti: in questa sessione un
% solo frame arriva a 62 C su una manciata di pixel, e prendendolo alla
% lettera tutti gli altri frame finirebbero schiacciati nel primo quinto
% della palette (immagini tutte viola scuro). Si scarta quindi la coda con
% dei percentili sui minimi/massimi dei singoli frame: la scala resta comune
% a tutta la sessione, ma coperta davvero dai dati.
fMin = nan(nFrames, 1);
fMax = nan(nFrames, 1);
for k = 1:nFrames
    repPath = fullfile(frames(k).dir, 'correction_report.json');
    if ~isfile(repPath), continue; end
    rep = jsondecode(fileread(repPath));
    if ~isempty(rep.corrected_c.min)
        fMin(k) = rep.corrected_c.min;
        fMax(k) = rep.corrected_c.max;
    end
end
fMin = fMin(~isnan(fMin));
fMax = fMax(~isnan(fMax));
if isempty(fMin)
    sessMin = 20; sessMax = 50;
else
    sessMin = pctOf(fMin, 0.05);
    sessMax = pctOf(fMax, 0.95);
end
fprintf('Scala colore sessione: %.1f .. %.1f C', sessMin, sessMax);
if ~isempty(fMax)
    fprintf('  (estremi reali %.1f .. %.1f C, code escluse)', min(fMin), max(fMax));
end
fprintf('\n');

%% 4. Stato dell'interfaccia
S.frames      = frames;
S.nFrames     = nFrames;
S.idx         = 1;
S.sessClim    = [sessMin sessMax];
S.lockedClim  = true;    % true = scala fissa sessione, false = per frame
S.showSamples = false;   % overlay dei campioni LiDAR diretti
S.showSegs    = true;    % overlay dei bordi dei superpixel (griglia SLIC)
S.pinned      = false;   % lettura bloccata sul punto cliccato
S.pinXY       = [NaN NaN];
S.materialDir = materialDir;
S.framesDir   = framesDir;
S.cache       = struct();

%% 5. Figura
S.fig = figure('Name', 'Correzione radiometrica - apparente vs corretta', ...
               'NumberTitle', 'off', 'Color', 'w', ...
               'Units', 'normalized', 'Position', [0.03 0.15 0.94 0.70]);

S.axApp  = subplot(1, 3, 1, 'Parent', S.fig);
S.axCorr = subplot(1, 3, 2, 'Parent', S.fig);
S.axZed  = subplot(1, 3, 3, 'Parent', S.fig);

S.imApp  = imagesc(S.axApp,  zeros(2));
S.imCorr = imagesc(S.axCorr, zeros(2));
axis(S.axApp,  'image'); axis(S.axCorr, 'image');
colormap(S.fig, inferno_like());
cb1 = colorbar(S.axApp);  cb1.Label.String = 'deg C';
cb2 = colorbar(S.axCorr); cb2.Label.String = 'deg C';

% Terzo pannello: il frame ZED da cui nascono i superpixel e, tramite CLIP,
% l'emissivita'. E' un'immagine RGB vera (image, non imagesc), quindi non
% risente della colormap termica ne' della scala colore.
S.imZed = image(S.axZed, zeros(2, 2, 3, 'uint8'));
axis(S.axZed, 'image');
set(S.axZed, 'XTick', [], 'YTick', []);

hold(S.axApp,  'on');
hold(S.axCorr, 'on');
S.samplesApp  = plot(S.axApp,  NaN, NaN, '.', 'Color', [0 0.6 1], 'MarkerSize', 1);
S.samplesCorr = plot(S.axCorr, NaN, NaN, '.', 'Color', [0 0.6 1], 'MarkerSize', 1);
% Reticolo dei superpixel: una sola linea NaN-separata per asse.
S.segsApp  = plot(S.axApp,  NaN, NaN, '-', 'Color', [1 1 1 0.45], 'LineWidth', 0.5);
S.segsCorr = plot(S.axCorr, NaN, NaN, '-', 'Color', [1 1 1 0.45], 'LineWidth', 0.5);
S.markApp  = plot(S.axApp,  NaN, NaN, '+', 'Color', 'c', 'MarkerSize', 14, 'LineWidth', 1.5);
S.markCorr = plot(S.axCorr, NaN, NaN, '+', 'Color', 'c', 'MarkerSize', 14, 'LineWidth', 1.5);
hold(S.axApp,  'off');
hold(S.axCorr, 'off');

hold(S.axZed, 'on');
S.segsZed = plot(S.axZed, NaN, NaN, '-', 'Color', [1 1 0 0.55], 'LineWidth', 0.5);
% Riquadro del superpixel puntato: e' il crop che CLIP ha effettivamente
% classificato (bbox da segments.json), piu' il suo centroide.
S.boxZed  = plot(S.axZed, NaN, NaN, '-', 'Color', 'c', 'LineWidth', 1.5);
S.markZed = plot(S.axZed, NaN, NaN, '+', 'Color', 'c', 'MarkerSize', 14, 'LineWidth', 1.5);
hold(S.axZed, 'off');

% Riga di lettura sotto le due immagini.
S.readout = annotation(S.fig, 'textbox', [0.02 0.005 0.96 0.085], ...
                       'String', '', 'FontName', 'Consolas', 'FontSize', 10, ...
                       'EdgeColor', [0.8 0.8 0.8], 'BackgroundColor', [0.97 0.97 0.97], ...
                       'VerticalAlignment', 'middle', 'Interpreter', 'none');

guidata(S.fig, S);
set(S.fig, 'WindowKeyPressFcn',        @(src, ev) onKey(src, ev), ...
           'WindowButtonMotionFcn',    @(src, ev) onMove(src), ...
           'WindowButtonDownFcn',      @(src, ev) onClick(src));

loadFrame(S.fig, 1);

end % ShowTemperatureCorrection


%% ------------------------------------------------------------------ %%
function loadFrame(fig, idx)
%LOADFRAME Carica e disegna il frame idx (con cache dei .npy gia' letti).
S = guidata(fig);
S.idx = max(1, min(S.nFrames, idx));
f = S.frames(S.idx);

key = matlab.lang.makeValidName(f.stem);
if ~isfield(S.cache, key)
    D.apparent  = readNPY(f.apparentPath);
    D.corrected = readNPY(f.correctedPath);

    % I file accessori possono mancare: la lettura resta comunque possibile,
    % semplicemente senza emissivita' / distanza / materiale.
    D.emissivity = tryReadNPY(fullfile(f.dir, 'emissivity_used.npy'));
    D.distance   = tryReadNPY(fullfile(f.dir, 'distance.npy'));
    D.segment    = tryReadNPY(fullfile(f.dir, 'segment_id.npy'));
    D.sampled    = tryReadNPY(fullfile(f.dir, 'sampled_mask.npy'));

    % I bordi dei superpixel si ricavano una volta sola: sono fissi per frame.
    [D.segX, D.segY] = segBoundaryLines(D.segment);

    D.material = containers.Map('KeyType', 'double', 'ValueType', 'any');
    segPath = fullfile(S.materialDir, f.stem, 'segments.json');
    if isfile(segPath)
        seg = jsondecode(fileread(segPath));
        for i = 1:numel(seg.segments)
            s = seg.segments(i);
            if iscell(s), s = s{1}; end
            D.material(double(s.id)) = s;
        end
    end

    % Frame ZED sorgente: il nome sta in stats.json (source_zed_frame), che
    % project_to_flir.py scrive accanto alle mappe. I superpixel qui sono
    % quelli originali di SLIC (labels.npy, risoluzione ZED), non la loro
    % riproiezione su FLIR.
    D.zed     = [];
    D.zedName = '';
    statsPath = fullfile(f.dir, 'stats.json');
    if isfile(statsPath)
        st = jsondecode(fileread(statsPath));
        if isfield(st, 'source_zed_frame')
            D.zedName = st.source_zed_frame;
            zedPath = fullfile(S.framesDir, D.zedName);
            if isfile(zedPath)
                D.zed = imread(zedPath);
            end
        end
    end
    D.labels = tryReadNPY(fullfile(S.materialDir, f.stem, 'labels.npy'));
    [D.zedSegX, D.zedSegY] = segBoundaryLines(D.labels);
    % segments.json non salva la bbox: la si ricava dalle label, una volta
    % per frame, per poter evidenziare il crop passato a CLIP.
    D.bbox = segBBoxes(D.labels);

    S.cache.(key) = D;
end
D = S.cache.(key);

set(S.imApp,  'CData', D.apparent);
set(S.imCorr, 'CData', D.corrected);
[h, w] = size(D.apparent);
set(S.axApp,  'XLim', [0.5 w+0.5], 'YLim', [0.5 h+0.5]);
set(S.axCorr, 'XLim', [0.5 w+0.5], 'YLim', [0.5 h+0.5]);

if ~isempty(D.zed)
    set(S.imZed, 'CData', D.zed);
    set(S.axZed, 'XLim', [0.5 size(D.zed, 2)+0.5], 'YLim', [0.5 size(D.zed, 1)+0.5]);
else
    set(S.imZed, 'CData', zeros(2, 2, 3, 'uint8'));
    set(S.axZed, 'XLim', [0.5 2.5], 'YLim', [0.5 2.5]);
end

if S.lockedClim
    clim(S.axApp,  S.sessClim);
    clim(S.axCorr, S.sessClim);
    climTag = sprintf('scala fissa %.1f-%.1f C', S.sessClim(1), S.sessClim(2));
else
    clim(S.axApp,  robustClim(D.apparent));
    clim(S.axCorr, robustClim(D.corrected));
    climTag = 'scala per frame';
end

title(S.axApp, sprintf('PRIMA - apparente (FLIR grezzo)\nmin %.1f  max %.1f  media %.1f C', ...
      min(D.apparent(:)), max(D.apparent(:)), mean(D.apparent(:))), 'FontSize', 9);
title(S.axCorr, sprintf('DOPO - corretta (emissivita + atmosfera)\nmin %.1f  max %.1f  media %.1f C', ...
      min(D.corrected(:), [], 'omitnan'), max(D.corrected(:), [], 'omitnan'), ...
      mean(D.corrected(:), 'omitnan')), 'FontSize', 9);

if isempty(D.zed)
    zedTitle = sprintf('ZED - sorgente emissivita\nframe non trovato in %s', S.framesDir);
else
    nSeg = 0;
    if ~isempty(D.labels), nSeg = numel(unique(D.labels(:))); end
    zedTitle = sprintf('ZED - da qui superpixel + CLIP\n%s   %d superpixel', D.zedName, nSeg);
end
title(S.axZed, zedTitle, 'FontSize', 9, 'Interpreter', 'none');

nNaN = sum(isnan(D.corrected(:)));
sgtitle(S.fig, sprintf('[%d/%d]  %s     %s     NaN %d px     (frecce = scorri, click = blocca punto, b = bordi superpixel)', ...
        S.idx, S.nFrames, f.stem, climTag, nNaN), 'FontSize', 10, 'Interpreter', 'none');

if S.showSamples && ~isempty(D.sampled)
    [ys, xs] = find(D.sampled);
    set(S.samplesApp,  'XData', xs, 'YData', ys);
    set(S.samplesCorr, 'XData', xs, 'YData', ys);
else
    set(S.samplesApp,  'XData', NaN, 'YData', NaN);
    set(S.samplesCorr, 'XData', NaN, 'YData', NaN);
end

if S.showSegs && ~isempty(D.segX)
    set(S.segsApp,  'XData', D.segX, 'YData', D.segY);
    set(S.segsCorr, 'XData', D.segX, 'YData', D.segY);
else
    set(S.segsApp,  'XData', NaN, 'YData', NaN);
    set(S.segsCorr, 'XData', NaN, 'YData', NaN);
end

if S.showSegs && ~isempty(D.zedSegX)
    set(S.segsZed, 'XData', D.zedSegX, 'YData', D.zedSegY);
else
    set(S.segsZed, 'XData', NaN, 'YData', NaN);
end

guidata(fig, S);
if S.pinned
    updateReadout(fig, S.pinXY(1), S.pinXY(2));
else
    updateReadout(fig, NaN, NaN);
end
end


%% ------------------------------------------------------------------ %%
function updateReadout(fig, xi, yi)
%UPDATEREADOUT Riempie la riga di testo con i valori del pixel (xi, yi).
S = guidata(fig);
D = S.cache.(matlab.lang.makeValidName(S.frames(S.idx).stem));
[h, w] = size(D.apparent);

if isnan(xi) || xi < 1 || xi > w || yi < 1 || yi > h
    set(S.readout, 'String', ...
        'Punta il mouse sull''immagine per leggere la temperatura prima / dopo la correzione.');
    set(S.markApp,  'XData', NaN, 'YData', NaN);
    set(S.markCorr, 'XData', NaN, 'YData', NaN);
    set(S.boxZed,   'XData', NaN, 'YData', NaN);
    set(S.markZed,  'XData', NaN, 'YData', NaN);
    return
end

tBefore = double(D.apparent(yi, xi));
tAfter  = double(D.corrected(yi, xi));
delta   = tAfter - tBefore;

if isnan(tAfter)
    line1 = sprintf('x=%3d y=%3d   PRIMA %6.2f C   DOPO   NaN (nessun materiale plausibile)', ...
                    xi, yi, tBefore);
else
    line1 = sprintf('x=%3d y=%3d   PRIMA %6.2f C   DOPO %6.2f C   DELTA %+5.2f C', ...
                    xi, yi, tBefore, tAfter, delta);
end

% Seconda riga: da dove viene quella correzione.
parts = {};
if ~isempty(D.emissivity) && ~isnan(D.emissivity(yi, xi))
    parts{end+1} = sprintf('emissivita %.3f', D.emissivity(yi, xi));
end
if ~isempty(D.distance) && D.distance(yi, xi) > 0
    parts{end+1} = sprintf('distanza LiDAR %.2f m', D.distance(yi, xi));
end
% Il superpixel puntato viene anche evidenziato sul frame ZED: la bbox e' il
% crop che CLIP ha classificato, quindi si vede subito su cosa ha deciso.
bx = NaN; by = NaN; cx = NaN; cy = NaN;
if ~isempty(D.segment)
    sid = double(D.segment(yi, xi));
    if sid >= 0 && isKey(D.material, sid)
        s = D.material(sid);
        parts{end+1} = sprintf('segmento %d = %s (CLIP %.0f%%)', ...
                               sid, s.top_material, 100 * s.confidence);
        if isKey(D.bbox, sid)
            b = D.bbox(sid);          % [x0 y0 x1 y1] in pixel ZED, 1-based
            bx = [b(1)-0.5 b(3)+0.5 b(3)+0.5 b(1)-0.5 b(1)-0.5];
            by = [b(2)-0.5 b(2)-0.5 b(4)+0.5 b(4)+0.5 b(2)-0.5];
        end
        if isfield(s, 'centroid_px')
            cx = double(s.centroid_px(1)) + 1;   % 0-based Python -> 1-based
            cy = double(s.centroid_px(2)) + 1;
        end
    else
        parts{end+1} = sprintf('segmento %d', sid);
    end
end
set(S.boxZed,  'XData', bx, 'YData', by);
set(S.markZed, 'XData', cx, 'YData', cy);
if ~isempty(D.sampled)
    if D.sampled(yi, xi)
        parts{end+1} = 'campione LiDAR diretto';
    else
        parts{end+1} = 'riempito per vicinanza (non misurato)';
    end
end
line2 = strjoin(parts, '   |   ');

if S.pinned
    line1 = ['[BLOCCATO]  ' line1];
end
set(S.readout, 'String', {line1, line2});
set(S.markApp,  'XData', xi, 'YData', yi);
set(S.markCorr, 'XData', xi, 'YData', yi);
end


%% ------------------------------------------------------------------ %%
function [xi, yi, inside] = cursorPixel(fig)
%CURSORPIXEL Pixel intero sotto il cursore, in uno qualsiasi dei due assi.
S = guidata(fig);
xi = NaN; yi = NaN; inside = false;
for ax = [S.axApp, S.axCorr]
    p = get(ax, 'CurrentPoint');
    x = round(p(1, 1));
    y = round(p(1, 2));
    xl = get(ax, 'XLim'); yl = get(ax, 'YLim');
    if x >= xl(1) && x <= xl(2) && y >= yl(1) && y <= yl(2)
        xi = x; yi = y; inside = true;
        return
    end
end
end


function onMove(fig)
S = guidata(fig);
if S.pinned, return; end
[xi, yi] = cursorPixel(fig);
updateReadout(fig, xi, yi);
end


function onClick(fig)
S = guidata(fig);
[xi, yi, inside] = cursorPixel(fig);
if ~inside, return; end
S.pinned = true;
S.pinXY = [xi yi];
guidata(fig, S);
updateReadout(fig, xi, yi);
end


function onKey(fig, ev)
S = guidata(fig);
switch ev.Key
    case 'rightarrow', loadFrame(fig, S.idx + 1);
    case 'leftarrow',  loadFrame(fig, S.idx - 1);
    case 'pagedown',   loadFrame(fig, S.idx + 10);
    case 'pageup',     loadFrame(fig, S.idx - 10);
    case 'home',       loadFrame(fig, 1);
    case 'end',        loadFrame(fig, S.nFrames);
    case 'c'
        S.pinned = ~S.pinned;
        guidata(fig, S);
        if S.pinned
            updateReadout(fig, S.pinXY(1), S.pinXY(2));
        else
            updateReadout(fig, NaN, NaN);
        end
    case 's'
        S.showSamples = ~S.showSamples;
        guidata(fig, S);
        loadFrame(fig, S.idx);
    case 'b'
        S.showSegs = ~S.showSegs;
        guidata(fig, S);
        loadFrame(fig, S.idx);
    case 'l'
        S.lockedClim = ~S.lockedClim;
        guidata(fig, S);
        loadFrame(fig, S.idx);
end
end


%% ------------------------------------------------------------------ %%
function B = segBBoxes(labels)
%SEGBBOXES Bounding box di ogni superpixel: mappa id -> [x0 y0 x1 y1] in
% pixel MATLAB (1-based, estremi inclusi). E' l'equivalente di segment_boxes()
% in emissivity\segmentation.py, ricalcolato qui perche' classify_session.py
% non riporta la bbox in segments.json.
B = containers.Map('KeyType', 'double', 'ValueType', 'any');
if isempty(labels)
    return
end
lab = double(labels);
[h, w] = size(lab);
[ids, ~, idx] = unique(lab(:));
[Y, X] = ndgrid(1:h, 1:w);
x0 = accumarray(idx, X(:), [], @min);
x1 = accumarray(idx, X(:), [], @max);
y0 = accumarray(idx, Y(:), [], @min);
y1 = accumarray(idx, Y(:), [], @max);
for i = 1:numel(ids)
    B(ids(i)) = [x0(i) y0(i) x1(i) y1(i)];
end
end


%% ------------------------------------------------------------------ %%
function [X, Y] = segBoundaryLines(seg)
%SEGBOUNDARYLINES Reticolo dei superpixel: segmenti di linea NaN-separati
% lungo ogni confine fra due segment_id diversi. Le linee cadono a meta' fra
% due pixel (x+0.5 / y+0.5), quindi combaciano con i bordi mostrati da
% imagesc e disegnano i quadrati SLIC senza coprire i pixel.
X = NaN; Y = NaN;
if isempty(seg)
    return
end
seg = double(seg);

% Confini verticali: fra la colonna x e la x+1.
[yv, xv] = find(seg(:, 1:end-1) ~= seg(:, 2:end));
Xv = [xv + 0.5, xv + 0.5, nan(size(xv))]';
Yv = [yv - 0.5, yv + 0.5, nan(size(yv))]';

% Confini orizzontali: fra la riga y e la y+1.
[yh, xh] = find(seg(1:end-1, :) ~= seg(2:end, :));
Xh = [xh - 0.5, xh + 0.5, nan(size(xh))]';
Yh = [yh + 0.5, yh + 0.5, nan(size(yh))]';

X = [Xv(:); Xh(:)];
Y = [Yv(:); Yh(:)];
if isempty(X)
    X = NaN; Y = NaN;
end
end


%% ------------------------------------------------------------------ %%
function c = robustClim(A)
%ROBUSTCLIM Limiti colore sui percentili 1-99, cosi' un singolo pixel caldo
% non schiaccia tutto il resto dell'immagine.
v = A(isfinite(A));
if isempty(v)
    c = [0 1];
    return
end
c = [pctOf(v, 0.01) pctOf(v, 0.99)];
if c(2) <= c(1)
    c = [min(v) max(v) + eps];
end
end


function p = pctOf(v, q)
%PCTOF Percentile q (0-1) di v, calcolato a mano per non dipendere dallo
% Statistics Toolbox (prctile non e' sempre disponibile).
v = sort(v(isfinite(v)));
if isempty(v)
    p = NaN;
    return
end
p = v(min(numel(v), max(1, round(q * numel(v)))));
end


function cmap = inferno_like()
%INFERNO_LIKE Palette scura->gialla stile 'inferno' (matplotlib), per avere
% lo stesso aspetto di ThermalData.py --show senza toolbox aggiuntivi.
anchors = [0.001 0.000 0.014
           0.259 0.039 0.406
           0.576 0.149 0.404
           0.865 0.317 0.226
           0.988 0.645 0.040
           0.988 0.998 0.645];
x = linspace(0, 1, size(anchors, 1));
xi = linspace(0, 1, 256);
cmap = [interp1(x, anchors(:,1), xi)', ...
        interp1(x, anchors(:,2), xi)', ...
        interp1(x, anchors(:,3), xi)'];
end


%% ------------------------------------------------------------------ %%
function A = tryReadNPY(path)
%TRYREADNPY Come readNPY ma restituisce [] se il file non c'e'.
if isfile(path)
    A = readNPY(path);
else
    A = [];
end
end


function A = readNPY(path)
%READNPY Lettore minimale del formato .npy di NumPy (v1.0/2.0).
% Copre i soli casi prodotti da questa pipeline: array 2-D, little-endian,
% ordine C, dtype float32/float64/int32/int64/uint8/bool. Evita di dipendere
% da npy-matlab, che non e' installato.
fid = fopen(path, 'r');
if fid < 0
    error('Impossibile aprire %s', path);
end
cleaner = onCleanup(@() fclose(fid));

magic = fread(fid, 6, '*uint8')';
if ~isequal(magic, uint8([147 78 85 77 80 89]))   % \x93NUMPY
    error('%s non e'' un file .npy', path);
end
major = fread(fid, 1, 'uint8');
fread(fid, 1, 'uint8');                            % minor, non serve
if major == 1
    headerLen = fread(fid, 1, 'uint16=>double');
else
    headerLen = fread(fid, 1, 'uint32=>double');
end
header = fread(fid, headerLen, '*char')';

descr = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
fortran = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
shapeTok = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
if isempty(descr) || isempty(shapeTok)
    error('Header .npy non riconosciuto in %s', path);
end
descr = descr{1};
shape = str2double(strsplit(strtrim(strrep(shapeTok{1}, ',', ' ')), ' '));
shape = shape(~isnan(shape));

switch descr
    case {'<f4', '=f4', 'f4'},                 fmt = 'single=>single';
    case {'<f8', '=f8', 'f8'},                 fmt = 'double=>double';
    case {'<i4', '=i4', 'i4'},                 fmt = 'int32=>int32';
    case {'<i8', '=i8', 'i8'},                 fmt = 'int64=>int64';
    case {'|u1', '<u1', 'u1'},                 fmt = 'uint8=>uint8';
    case {'|b1', '<b1', 'b1'},                 fmt = 'uint8=>uint8';
    otherwise
        error('dtype .npy non supportato (%s) in %s', descr, path);
end

n = prod(shape);
raw = fread(fid, n, fmt, 0, 'ieee-le');
if numel(raw) ~= n
    error('File .npy troncato: %s', path);
end

% NumPy salva in ordine C (righe consecutive), MATLAB legge in ordine
% colonne: si riempie trasposto e poi si traspone.
if strcmp(fortran{1}, 'True')
    A = reshape(raw, shape);
else
    A = reshape(raw, fliplr(shape))';
end
if strcmp(descr(end-1:end), 'b1')
    A = logical(A);
end
if ~islogical(A) && ~isinteger(A)
    A = double(A);
end
end
