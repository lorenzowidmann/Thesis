function ShowMaterialConsensus(sessionDirIn, mapDirIn)
%% Confronto interattivo materiali prima / dopo il consenso multi-vista
% Mostra il frame ZED con i materiali assegnati da classify_session.py (una
% sola vista, a sinistra) e quelli decisi da voxel_consensus.py --stage vote
% votando su tutte le viste della sessione (a destra), e si puo' scorrere
% tutta la sessione con le frecce.
%
% Stessa idea di ShowTemperatureCorrection.m, ma sul lato materiali invece
% che sul lato temperatura: qui la domanda non e' "quanto cambia il valore"
% ma "quali regioni il voto ha cambiato idea su, e con quanto accordo".
%
% Entrambi i pannelli leggono material_map_consensus/<stem>/segments.json:
% il consenso e' in top_material, la scelta della singola vista e' conservata
% in consensus.from_frame. Le regioni riassegnate sono contornate in rosso.
%
% Comandi da tastiera:
%   freccia dx / sx   frame successivo / precedente
%   pag su / pag giu  avanti / indietro di 10 frame
%   home / fine       primo / ultimo frame
%   c                 zoom sul ritaglio FLIR / frame ZED intero
%   e                 colore per materiale / per emissivita'
%   t                 mostra / nasconde le etichette di testo
%   a                 nelle etichette: confidenza CLIP / accordo del voto
%   b                 mostra / nasconde i bordi fra materiali diversi
%
% Uso:
%   ShowMaterialConsensus                        % percorsi di default, qui sotto
%   ShowMaterialConsensus(sessionDir)
%   ShowMaterialConsensus(sessionDir, mapDir)    % es. per confrontare due run

close all
clc

%% 1. Parametri
sessionDir = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate";   % sessione 6
mapDir     = '';   % vuoto = <sessionDir>\material_map_consensus

if nargin >= 1 && ~isempty(sessionDirIn)
    sessionDir = sessionDirIn;
end
if nargin >= 2 && ~isempty(mapDirIn)
    mapDir = mapDirIn;
end
if isempty(mapDir)
    mapDir = fullfile(sessionDir, 'material_map_consensus');
end
framesDir = fullfile(sessionDir, 'frames');

if ~isfolder(mapDir)
    error(['Cartella del consenso non trovata: %s\n' ...
           'Eseguire prima voxel_consensus.py --stage vote'], mapDir);
end
if ~isfolder(framesDir)
    error('Cartella frames non trovata: %s', framesDir);
end

%% 2. Elenco dei frame utilizzabili
% Serve la terna completa: raster delle regioni, record dei materiali e il
% PNG ZED sorgente (il cui nome sta dentro segments.json).
d = dir(fullfile(mapDir, '*'));
d = d([d.isdir] & ~ismember({d.name}, {'.', '..'}));

frames = struct('stem', {}, 'dir', {}, 'zedPath', {});
for k = 1:numel(d)
    frameDir = fullfile(mapDir, d(k).name);
    labPath  = fullfile(frameDir, 'labels.npy');
    segPath  = fullfile(frameDir, 'segments.json');
    if ~isfile(labPath) || ~isfile(segPath)
        continue
    end
    meta = jsondecode(fileread(segPath));
    if ~isfield(meta, 'source_zed_frame')
        continue
    end
    zedPath = fullfile(framesDir, meta.source_zed_frame);
    if ~isfile(zedPath)
        continue
    end
    frames(end+1) = struct('stem', d(k).name, ...
                           'dir', frameDir, ...
                           'zedPath', zedPath); %#ok<AGROW>
end

if isempty(frames)
    error('Nessun frame utilizzabile in %s', mapDir);
end
nFrames = numel(frames);
fprintf('%d frame in %s\n', nFrames, mapDir);

%% 3. Stato dell'interfaccia
S.frames     = frames;
S.nFrames    = nFrames;
S.idx        = 1;
S.cropZoom   = true;    % parte inquadrando il ritaglio FLIR
S.byEps      = false;   % false = colore per materiale, true = per emissivita'
S.showText   = true;
S.showAgree  = false;   % nelle etichette: false = confidenza, true = accordo
S.showEdges  = true;
S.colors     = materialColors();
S.cache      = struct();

%% 4. Figura
S.fig = figure('Name', 'Materiali - singola vista vs consenso multi-vista', ...
               'NumberTitle', 'off', 'Color', 'w', ...
               'Units', 'normalized', 'Position', [0.04 0.12 0.92 0.76]);

S.axBefore = subplot(1, 2, 1, 'Parent', S.fig);
S.axAfter  = subplot(1, 2, 2, 'Parent', S.fig);

S.imBefore = image(S.axBefore, zeros(2, 2, 3, 'uint8'));
S.imAfter  = image(S.axAfter,  zeros(2, 2, 3, 'uint8'));
axis(S.axBefore, 'image'); axis(S.axAfter, 'image');
set([S.axBefore S.axAfter], 'XTick', [], 'YTick', []);

% Le etichette di testo vengono ricreate a ogni frame: si tengono in un
% array per poterle cancellare senza ridisegnare le immagini.
S.txtBefore = gobjects(0);
S.txtAfter  = gobjects(0);

S.readout = annotation(S.fig, 'textbox', [0.02 0.005 0.96 0.075], ...
                       'String', '', 'FontName', 'Consolas', 'FontSize', 10, ...
                       'EdgeColor', [0.8 0.8 0.8], 'BackgroundColor', [0.97 0.97 0.97], ...
                       'VerticalAlignment', 'middle', 'Interpreter', 'none');

guidata(S.fig, S);
set(S.fig, 'WindowKeyPressFcn', @(src, ev) onKey(src, ev));

loadFrame(S.fig, 1);

end % ShowMaterialConsensus


%% ------------------------------------------------------------------ %%
function loadFrame(fig, idx)
%LOADFRAME Carica e disegna il frame idx (con cache dei dati gia' letti).
S = guidata(fig);
S.idx = max(1, min(S.nFrames, idx));
f = S.frames(S.idx);

key = matlab.lang.makeValidName(f.stem);
if ~isfield(S.cache, key)
    D.labels = readNPY(fullfile(f.dir, 'labels.npy'));
    D.rgb    = imread(f.zedPath);
    meta     = jsondecode(fileread(fullfile(f.dir, 'segments.json')));

    % jsondecode restituisce struct array se tutti i segmenti hanno gli
    % stessi campi, cell array altrimenti: si normalizza a cell.
    segs = meta.segments;
    if ~iscell(segs)
        segs = num2cell(segs);
    end

    n = numel(segs);
    D.id       = zeros(n, 1);
    D.after    = cell(n, 1);
    D.before   = cell(n, 1);
    D.conf     = nan(n, 1);
    D.agree    = nan(n, 1);
    D.epsAfter = nan(n, 1);
    D.area     = zeros(n, 1);
    D.cx       = nan(n, 1);
    D.cy       = nan(n, 1);
    D.status   = cell(n, 1);
    for i = 1:n
        s = segs{i};
        D.id(i)       = double(s.id);
        D.after{i}    = s.top_material;
        D.conf(i)     = double(s.confidence);
        D.epsAfter(i) = double(s.emissivity);
        D.area(i)     = double(s.area_px);
        D.cx(i)       = double(s.centroid_px(1)) + 1;   % 0-based -> 1-based
        D.cy(i)       = double(s.centroid_px(2)) + 1;
        D.before{i}   = s.top_material;                 % default se manca il consenso
        D.status{i}   = 'n/a';
        if isfield(s, 'consensus') && isstruct(s.consensus)
            c = s.consensus;
            if isfield(c, 'from_frame') && ~isempty(c.from_frame)
                D.before{i} = c.from_frame;
            end
            if isfield(c, 'agreement') && ~isempty(c.agreement)
                D.agree(i) = double(c.agreement);
            end
            if isfield(c, 'status')
                D.status{i} = c.status;
            end
        end
    end

    D.changed = ~strcmp(D.before, D.after);
    D.crop = [];
    if isfield(meta, 'flir_fov_crop') && isstruct(meta.flir_fov_crop)
        c = meta.flir_fov_crop;
        D.crop = [double(c.x0) double(c.y0) double(c.x1) double(c.y1)];
    end
    D.zedName = meta.source_zed_frame;
    S.cache.(key) = D;
end
D = S.cache.(key);

% I due pannelli condividono geometria e differiscono solo nell'etichetta
% assegnata a ciascuna regione.
visBefore = paintPanel(S, D, D.before, false);
visAfter  = paintPanel(S, D, D.after,  true);

set(S.imBefore, 'CData', visBefore);
set(S.imAfter,  'CData', visAfter);
[h, w, ~] = size(visBefore);
if S.cropZoom && ~isempty(D.crop)
    xl = [D.crop(1)+0.5 D.crop(3)+0.5];
    yl = [D.crop(2)+0.5 D.crop(4)+0.5];
else
    xl = [0.5 w+0.5];
    yl = [0.5 h+0.5];
end
set([S.axBefore S.axAfter], 'XLim', xl, 'YLim', yl);

nChanged = sum(D.changed);
nVotable = sum(~strcmp(D.status, 'no_lidar_sample'));
title(S.axBefore, sprintf('PRIMA - singola vista (classify\\_session)\n%d regioni', numel(D.id)), ...
      'FontSize', 10);
title(S.axAfter, sprintf('DOPO - consenso multi-vista (voxel\\_consensus)\n%d regioni riassegnate (contorno rosso)', nChanged), ...
      'FontSize', 10);

delete(S.txtBefore(isgraphics(S.txtBefore)));
delete(S.txtAfter(isgraphics(S.txtAfter)));
S.txtBefore = gobjects(0);
S.txtAfter  = gobjects(0);
if S.showText
    S.txtBefore = drawLabels(S.axBefore, S, D, D.before, false);
    S.txtAfter  = drawLabels(S.axAfter,  S, D, D.after,  true);
end

modeTag = 'colore per materiale';
if S.byEps
    modeTag = 'colore per emissivita';
end
zoomTag = 'ritaglio FLIR';
if ~S.cropZoom || isempty(D.crop)
    zoomTag = 'frame intero';
end
sgtitle(S.fig, sprintf('[%d/%d]  %s  <-  %s     %s     %s', ...
        S.idx, S.nFrames, f.stem, D.zedName, zoomTag, modeTag), ...
        'FontSize', 10, 'Interpreter', 'none');

% Riepilogo testuale: cosa e' cambiato e con quanto accordo.
changedIdx = find(D.changed);
if isempty(changedIdx)
    line2 = 'nessuna regione riassegnata dal consenso in questo frame';
else
    parts = cell(1, min(4, numel(changedIdx)));
    for i = 1:numel(parts)
        j = changedIdx(i);
        parts{i} = sprintf('#%d %s->%s (acc %.2f)', ...
                           D.id(j), D.before{j}, D.after{j}, D.agree(j));
    end
    line2 = strjoin(parts, '   ');
    if numel(changedIdx) > numel(parts)
        line2 = [line2 sprintf('   (+%d)', numel(changedIdx) - numel(parts))];
    end
end
line1 = sprintf(['%d regioni   %d riassegnate (%.0f%% delle %d votate)   ' ...
                 '%d senza voto utile   |   frecce = scorri, c = zoom, ' ...
                 'e = emissivita, t = testo, a = accordo, b = bordi'], ...
                numel(D.id), nChanged, ...
                100 * nChanged / max(1, nVotable), nVotable, ...
                numel(D.id) - nVotable);
set(S.readout, 'String', {line1, line2});

guidata(fig, S);
end


%% ------------------------------------------------------------------ %%
function vis = paintPanel(S, D, materials, markChanged)
%PAINTPANEL Frame ZED con le regioni colorate per materiale (o emissivita'),
% i bordi fra materiali diversi in bianco e, sul pannello del consenso, il
% contorno rosso delle regioni riassegnate.
base = im2double(D.rgb);
[h, w, ~] = size(base);
fill = zeros(h, w, 3);
matId = zeros(h, w);            % 0 = nessuna regione

for i = 1:numel(D.id)
    m = (D.labels == D.id(i));
    if ~any(m(:))
        continue
    end
    if S.byEps
        col = epsColor(D.epsAfter(i));
    else
        col = materialColor(S.colors, materials{i});
    end
    for c = 1:3
        ch = fill(:, :, c);
        ch(m) = col(c);
        fill(:, :, c) = ch;
    end
    matId(m) = materialIndex(S.colors, materials{i});
end

covered = D.labels >= 0;
vis = base;
for c = 1:3
    ch = vis(:, :, c);
    fc = fill(:, :, c);
    ch(covered) = 0.50 * ch(covered) + 0.50 * fc(covered);
    vis(:, :, c) = ch;
end

if S.showEdges
    e = false(h, w);
    e(:, 1:end-1) = e(:, 1:end-1) | (matId(:, 1:end-1) ~= matId(:, 2:end));
    e(1:end-1, :) = e(1:end-1, :) | (matId(1:end-1, :) ~= matId(2:end, :));
    vis = stampMask(vis, e, [1 1 1]);
end

if markChanged && any(D.changed)
    sel = false(h, w);
    for i = find(D.changed)'
        sel = sel | (D.labels == D.id(i));
    end
    b = false(h, w);
    b(:, 1:end-1) = b(:, 1:end-1) | (sel(:, 1:end-1) ~= sel(:, 2:end));
    b(1:end-1, :) = b(1:end-1, :) | (sel(1:end-1, :) ~= sel(2:end, :));
    b = b | circshift(b, 1, 1) | circshift(b, 1, 2);   % ispessisce di 1 px
    vis = stampMask(vis, b, [1 0.16 0.16]);
end

vis = uint8(255 * min(max(vis, 0), 1));
end


function vis = stampMask(vis, mask, col)
%STAMPMASK Scrive un colore pieno dove mask e' vera.
for c = 1:3
    ch = vis(:, :, c);
    ch(mask) = col(c);
    vis(:, :, c) = ch;
end
end


%% ------------------------------------------------------------------ %%
function handles = drawLabels(ax, S, D, materials, isAfter)
%DRAWLABELS Etichetta le regioni abbastanza grandi da reggere del testo.
% Il centroide arriva da segments.json: su una regione non convessa puo'
% cadere fuori dalla regione stessa, ma per un visualizzatore va bene ed
% evita di dipendere dall'Image Processing Toolbox.
handles = gobjects(0);
minArea = 2500;
for i = 1:numel(D.id)
    if D.area(i) < minArea
        continue
    end
    if S.showAgree && isAfter && ~isnan(D.agree(i))
        txt = sprintf('%s %.2f', materials{i}, D.agree(i));
    else
        txt = sprintf('%s %.2f', materials{i}, D.conf(i));
    end
    col = [1 1 1];
    if isAfter && D.changed(i)
        col = [1 0.55 0.55];
    end
    handles(end+1) = text(ax, D.cx(i), D.cy(i), txt, ...
        'Color', col, 'FontSize', 8, 'FontName', 'Consolas', ...
        'HorizontalAlignment', 'center', 'Interpreter', 'none', ...
        'BackgroundColor', [0 0 0 0.55], 'Margin', 1); %#ok<AGROW>
end
end


%% ------------------------------------------------------------------ %%
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
        S.cropZoom = ~S.cropZoom; guidata(fig, S); loadFrame(fig, S.idx);
    case 'e'
        S.byEps = ~S.byEps;       guidata(fig, S); loadFrame(fig, S.idx);
    case 't'
        S.showText = ~S.showText; guidata(fig, S); loadFrame(fig, S.idx);
    case 'a'
        S.showAgree = ~S.showAgree; guidata(fig, S); loadFrame(fig, S.idx);
    case 'b'
        S.showEdges = ~S.showEdges; guidata(fig, S); loadFrame(fig, S.idx);
end
end


%% ------------------------------------------------------------------ %%
function m = materialColors()
%MATERIALCOLORS Un colore per materiale, gli stessi usati nelle figure Python
% cosi' i due insiemi di immagini restano confrontabili.
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
if isKey(map, name)
    col = map(name);
else
    col = [0.5 0.5 0.5];        % materiale non previsto: grigio
end
end


function idx = materialIndex(map, name)
%MATERIALINDEX Indice stabile del materiale, solo per trovare i bordi fra
% materiali diversi (non serve che sia lo stesso fra frame diversi).
k = keys(map);
idx = find(strcmp(k, name), 1);
if isempty(idx)
    idx = numel(k) + 1;
end
end


function col = epsColor(eps)
%EPSCOLOR Blu -> giallo su 0.89-0.95, l'intervallo in cui cade tutto quello
% che questa pipeline puo' produrre al chiuso.
t = min(max((eps - 0.89) / 0.06, 0), 1);
col = [0.15 + 0.80 * t, 0.25 + 0.65 * t, 0.85 - 0.65 * t];
end


%% ------------------------------------------------------------------ %%
function A = readNPY(path)
%READNPY Lettore minimale del formato .npy di NumPy (v1.0/2.0).
% Stessa implementazione di ShowTemperatureCorrection.m: copre i soli casi
% prodotti da questa pipeline (array 2-D, little-endian, ordine C) ed evita
% di dipendere da npy-matlab, che non e' installato.
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
A = double(A);
end
