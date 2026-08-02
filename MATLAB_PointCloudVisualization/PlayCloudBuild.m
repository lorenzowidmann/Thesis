function PlayCloudBuild(bagPathIn, topicIn)
%% Riproduzione temporale della costruzione della point cloud (bag ROS2)
% Stessa sorgente dati di Matlab_try1.m (/cloud_registered di FAST-LIO, gia'
% nel frame mappa), ma invece di mostrare la nuvola gia' accumulata la
% ricostruisce frame per frame come un video, con controlli play / pausa /
% avanti / indietro e barra di scorrimento.
%
% I punti restano identici a quelli di Matlab_try1.m: qui cambia solo il
% modo di visualizzarli, ogni frame viene filtrato singolarmente cosi' da
% poterlo aggiungere in modo incrementale.
%
% La checkbox "frame passati" decide cosa resta a schermo, senza rileggere i
% dati: spenta si vede solo il frame corrente, come un video; accesa si
% vedono anche i precedenti (tutti, o gli ultimi trailFrames).
%
% Comandi da tastiera:
%   spazio        play / pausa
%   freccia dx    frame successivo
%   freccia sx    frame precedente
%   home / fine   primo / ultimo frame
%   p             frame passati si' / no
%
% Uso:
%   PlayCloudBuild                      % bag e topic di default, qui sotto
%   PlayCloudBuild(bagPath)             % altra bag, stesso topic
%   PlayCloudBuild(bagPath, topic)

close all
clc

%% 1. Parametri
bagPath    = 'C:\Users\loren\Desktop\Dati_vfinal\Calibration\Extr_tryN\Lidar\extr_tryN_frames.mat';
topicName  = '/cloud_registered';

% Gli argomenti, se passati, hanno la precedenza sui default qui sopra.
if nargin >= 1 && ~isempty(bagPathIn)
    bagPath = bagPathIn;
end
if nargin >= 2 && ~isempty(topicIn)
    topicName = topicIn;
end

frameStep  = 10;     % 1 = tutti i frame. Alzare (es. 5) se la memoria non basta.
                     % Vale solo per le bag: i .mat sono gia' decimati in export.
maxFrames  = Inf;    % limite di frame da leggere, Inf = nessun limite
voxelSize  = 0.02;   % m, dimensione voxel per il downsampling. 0 = disattivato
markerSize = 20;     % dimensione dei punti a schermo

fpsStart   = 10;     % velocita' iniziale di riproduzione (frame al secondo)
colorStart = 'quota';        % 'quota' (Z) oppure 'tempo' (indice frame)
maxRenderPoints = 8e5;       % tetto ai punti disegnati, oltre si decima la vista

% Frame passati a schermo o no. E' la checkbox "frame passati", commutabile
% anche a finestra aperta (tasto p):
%   false  solo il frame corrente, come un video vero
%   true   anche i frame gia' passati, la nuvola che cresce
showPastStart = true;
% Quanti frame passati tenere quando la checkbox e' attiva:
%   Inf  tutti, dal primo frame
%   N    solo gli ultimi N, cioe' una scia
trailFrames   = Inf;

% --- Crop geometrico (ROI), come in Matlab_try1.m ---
useROI = true;
roi = [-Inf Inf, ...    % X min max
       -Inf Inf, ...    % Y min max
       -Inf Inf];       % Z min max

% --- Denoise statistico, applicato al singolo frame ---
% Nota: qui gira una volta per frame invece che sulla nuvola totale, quindi
% il preprocessing puo' richiedere qualche decina di secondi su bag lunghe.
useDenoise   = true;
denoiseK     = 20;    % numero di vicini considerati
denoiseThres = 1.0;   % soglia in deviazioni standard

%% 2. Lettura dei frame grezzi
% Due sorgenti possibili, entrambe producono 'raw', un cell array di nuvole
% Nx3 nell'ordine di acquisizione:
%   - directory rosbag2 con un topic sensor_msgs/PointCloud2;
%   - file .mat prodotto da Calibration/export_livox_cloud.py, per le bag
%     Livox grezze (livox_ros_driver2/CustomMsg) che MATLAB non sa leggere
%     senza generare prima il supporto ai messaggi custom con ros2genmsg.
[~, ~, ext] = fileparts(bagPath);

if strcmpi(ext, '.mat')
    D = load(bagPath);
    if ~all(isfield(D, {'xyz', 'counts', 'stamps'}))
        error(['%s non contiene xyz/counts/stamps.\n' ...
            'Rigenerarlo con export_livox_cloud.py.'], bagPath);
    end
    if isfield(D, 'topic'), topicName = D.topic; end

    countsAll = double(D.counts(:));
    nAll      = numel(countsAll);
    edges     = [0; cumsum(countsAll)];   % il frame j occupa edges(j)+1 : edges(j+1)

    % frameStep non si applica qui: il .mat esce gia' decimato da
    % export_livox_cloud.py --step, e decimare una seconda volta lascerebbe
    % pochissimi frame senza che si veda perche'. maxFrames invece vale.
    idx = 1:nAll;
    if numel(idx) > maxFrames
        idx = idx(1:maxFrames);
    end
    fprintf('%s: %d frame, ne uso %d\n', bagPath, nAll, numel(idx));

    raw = cell(numel(idx), 1);
    for i = 1:numel(idx)
        j = idx(i);
        raw{i} = D.xyz(edges(j)+1 : edges(j+1), :);
    end
    tAbs = double(D.stamps(:));
    tRel = tAbs(idx) - tAbs(idx(1));
    clear D
else
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

    % rosReadXYZ legge solo sensor_msgs/PointCloud2: meglio dirlo subito che
    % fallire dentro il ciclo di preprocessing.
    try
        msgType = string(bag.AvailableTopics{topicName, 'MessageType'});
    catch
        msgType = "";
    end
    if strlength(msgType) > 0 && ~contains(msgType, 'PointCloud2')
        error(['Il topic %s e'' di tipo %s, non sensor_msgs/PointCloud2.\n' ...
            'Per le bag Livox grezze passare da Calibration/export_livox_cloud.py:\n' ...
            '  py export_livox_cloud.py --bag "%s" --output frames.mat\n' ...
            '  PlayCloudBuild(''frames.mat'')'], topicName, msgType, bagPath);
    end

    idx = 1:frameStep:nTotal;
    if numel(idx) > maxFrames
        idx = idx(1:maxFrames);
    end
    fprintf('Lettura di %d frame (step %d)\n', numel(idx), frameStep);

    msgs = readMessages(sel, idx);
    raw  = cell(numel(msgs), 1);
    for i = 1:numel(msgs)
        p = rosReadXYZ(msgs{i});
        if ndims(p) == 3    % nuvola organizzata: la si linearizza in Nx3
            p = reshape(p, [], 3);
        end
        raw{i} = p;
    end

    % Istanti dei frame, riportati a zero sul primo. Servono solo per il
    % testo di stato, se la bag non li espone si ripiega sull'indice.
    try
        tAbs = sel.MessageList.Time(idx);
        tRel = tAbs(:) - tAbs(1);
    catch
        tRel = (0:numel(raw)-1)' / max(fpsStart, 1);
    end
    clear msgs
end

nF = numel(raw);

%% 3. Filtraggio frame per frame
% A differenza di Matlab_try1.m i filtri girano sul singolo frame: cosi' la
% nuvola mostrata al passo k e' esattamente la somma dei primi k frame gia'
% puliti, e non serve rifiltrare nulla durante la riproduzione.
pts   = cell(nF, 1);
nRaw  = 0;
fprintf('Preprocessing frame');
for i = 1:nF
    p = raw{i};
    nRaw = nRaw + size(p, 1);

    % Rimozione dei ritorni non validi (NaN/Inf)
    p = p(all(isfinite(p), 2), :);

    if isempty(p)
        pts{i} = zeros(0, 3, 'single');
        continue
    end

    pcF = pointCloud(p);

    if useROI
        pcF = select(pcF, findPointsInROI(pcF, roi));
    end
    if useDenoise && pcF.Count > denoiseK
        pcF = pcdenoise(pcF, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    end
    if voxelSize > 0 && pcF.Count > 0
        pcF = pcdownsample(pcF, 'gridAverage', voxelSize);
    end

    if pcF.Count == 0
        pts{i} = zeros(0, 3, 'single');
    else
        pts{i} = single(pcF.Location);
    end

    if mod(i, 10) == 0 || i == nF
        fprintf(' %d', i);
    end
end
fprintf('\n');

counts = cellfun(@(x) size(x, 1), pts);
cumEnd = cumsum(counts);            % ultimo indice della nuvola al frame k
XYZ    = vertcat(pts{:});
clear pts raw

if isempty(XYZ)
    error('Nessun punto valido dopo il filtraggio: allentare ROI/denoise.');
end

fprintf('Punti letti: %d, dopo filtri: %d\n', nRaw, size(XYZ, 1));
fprintf('Media per frame: %.0f punti, totale finale: %d\n', ...
    mean(counts), cumEnd(end));

% Le due scale di colore disponibili: quota Z oppure istante di acquisizione
cQuota = XYZ(:, 3);
cTempo = repelem(single(1:nF)', counts);

%% 4. Estensione della nuvola completa
% I limiti degli assi vengono fissati sulla nuvola finale, altrimenti la
% vista cambierebbe scala a ogni frame aggiunto.
lo = min(XYZ, [], 1);
hi = max(XYZ, [], 1);
pad = 0.02 * max(hi - lo);

fprintf('\nEstensione [min max] in metri:\n');
fprintf('  X: %7.2f  %7.2f\n', lo(1), hi(1));
fprintf('  Y: %7.2f  %7.2f\n', lo(2), hi(2));
fprintf('  Z: %7.2f  %7.2f\n', lo(3), hi(3));

%% 5. Stato della riproduzione
k        = 1;            % frame corrente
cData    = cQuota;       % scala colore attiva, va inizializzata qui perche'
                         % e' condivisa fra le funzioni interne
playing  = false;
fpsList  = [2 5 10 15 25 50];
fps      = fpsStart;
showNew  = true;         % evidenzia in bianco il frame appena aggiunto
loopPlay = false;
showPast = logical(showPastStart);   % mostra anche i frame gia' passati

%% 6. Finestra e controlli
fig = figure('Name', sprintf('%s - costruzione nel tempo (%d frame)', topicName, nF), ...
    'Color', 'k', 'NumberTitle', 'off', 'Position', [80 80 1200 800]);

ax = axes('Parent', fig, 'Position', [0.07 0.20 0.88 0.74], 'Color', 'k', ...
    'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', ...
    'GridColor', [0.4 0.4 0.4], 'NextPlot', 'add');

hAll = scatter3(ax, nan, nan, nan, markerSize, 0, '.');
hNew = scatter3(ax, nan, nan, nan, markerSize * 2, 'w', '.');

xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
xlim(ax, [lo(1) - pad, hi(1) + pad]);
ylim(ax, [lo(2) - pad, hi(2) + pad]);
zlim(ax, [lo(3) - pad, hi(3) + pad]);
daspect(ax, [1 1 1]);
grid(ax, 'on');
view(ax, 3);
colormap(ax, turbo);

hTitle = title(ax, '', 'Color', 'w');

% --- barra di scorrimento ---
hSlider = uicontrol(fig, 'Style', 'slider', 'Units', 'normalized', ...
    'Position', [0.05 0.075 0.90 0.035], ...
    'Min', 1, 'Max', max(nF, 1 + eps), 'Value', 1, ...
    'SliderStep', sliderStepFor(nF), ...
    'Callback', @(s, ~) setFrame(round(get(s, 'Value'))));
if nF < 2
    set(hSlider, 'Enable', 'off');
end

% --- testo di stato ---
hStatus = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.05 0.115 0.90 0.03], 'HorizontalAlignment', 'left', ...
    'BackgroundColor', 'k', 'ForegroundColor', 'w', 'FontSize', 10);

% --- pulsanti ---
uicontrol(fig, 'Style', 'pushbutton', 'String', '|<', 'Units', 'normalized', ...
    'Position', [0.050 0.015 0.060 0.045], 'TooltipString', 'Primo frame', ...
    'Callback', @(~, ~) setFrame(1));
uicontrol(fig, 'Style', 'pushbutton', 'String', '<<', 'Units', 'normalized', ...
    'Position', [0.115 0.015 0.060 0.045], 'TooltipString', 'Frame precedente', ...
    'Callback', @(~, ~) stepFrame(-1));
hPlay = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Play', 'Units', 'normalized', ...
    'Position', [0.180 0.015 0.090 0.045], 'FontWeight', 'bold', ...
    'Callback', @(~, ~) togglePlay());
uicontrol(fig, 'Style', 'pushbutton', 'String', '>>', 'Units', 'normalized', ...
    'Position', [0.275 0.015 0.060 0.045], 'TooltipString', 'Frame successivo', ...
    'Callback', @(~, ~) stepFrame(1));
uicontrol(fig, 'Style', 'pushbutton', 'String', '>|', 'Units', 'normalized', ...
    'Position', [0.340 0.015 0.060 0.045], 'TooltipString', 'Ultimo frame', ...
    'Callback', @(~, ~) setFrame(nF));

hFps = uicontrol(fig, 'Style', 'popupmenu', 'Units', 'normalized', ...
    'Position', [0.410 0.020 0.070 0.045], ...
    'String', cellstr(compose('%d fps', fpsList(:))), ...
    'Value', nearestIdx(fpsList, fps), 'Callback', @(~, ~) onFps());

if isfinite(trailFrames)
    pastLabel = sprintf('frame passati (%d)', trailFrames);
else
    pastLabel = 'frame passati';
end
hPast = uicontrol(fig, 'Style', 'checkbox', 'String', pastLabel, ...
    'Units', 'normalized', 'Position', [0.495 0.015 0.130 0.045], ...
    'Value', showPast, 'BackgroundColor', 'k', 'ForegroundColor', 'w', ...
    'TooltipString', 'Spenta: solo il frame corrente. Accesa: anche i precedenti (p)', ...
    'Callback', @(s, ~) onPast(get(s, 'Value')));

hColor = uicontrol(fig, 'Style', 'popupmenu', 'Units', 'normalized', ...
    'Position', [0.635 0.020 0.120 0.045], ...
    'String', {'colore: quota', 'colore: tempo'}, ...
    'Value', 1 + strcmpi(colorStart, 'tempo'), 'Callback', @(~, ~) onColor());

hHigh = uicontrol(fig, 'Style', 'checkbox', 'String', 'evidenzia', ...
    'Units', 'normalized', 'Position', [0.765 0.015 0.100 0.045], ...
    'Value', showNew, 'BackgroundColor', 'k', 'ForegroundColor', 'w', ...
    'TooltipString', 'Frame corrente in bianco sopra gli altri', ...
    'Callback', @(s, ~) onHighlight(s));

uicontrol(fig, 'Style', 'checkbox', 'String', 'loop', ...
    'Units', 'normalized', 'Position', [0.875 0.015 0.070 0.045], ...
    'Value', loopPlay, 'BackgroundColor', 'k', 'ForegroundColor', 'w', ...
    'Callback', @(s, ~) onLoop(s));

% Tastiera attiva sia sulla figura sia sui controlli, che altrimenti
% catturerebbero il tasto una volta presi a fuoco.
set(fig, 'WindowKeyPressFcn', @onKey);
set(findobj(fig, 'Type', 'uicontrol'), 'KeyPressFcn', @onKey);

%% 7. Timer di riproduzione
tmr = timer('ExecutionMode', 'fixedSpacing', 'BusyMode', 'drop', ...
    'Period', round(1 / fps, 3), 'TimerFcn', @(~, ~) onTick());
set(fig, 'CloseRequestFcn', @(~, ~) onClose());

onPast(showPast);   % allinea i controlli alla vista iniziale
onColor();          % imposta la scala colore e disegna il primo frame

%% --- funzioni interne -------------------------------------------------

    function setFrame(newK)
        k = min(max(round(newK), 1), nF);
        render();
    end

    function stepFrame(d)
        setPlaying(false);
        setFrame(k + d);
    end

    function render()
        n = cumEnd(k);              % ultimo indice della nuvola al frame k
        s = n - counts(k) + 1;      % primo indice del frame k

        % Primo indice da disegnare: e' l'unica cosa che cambia fra le viste,
        % i punti restano gli stessi.
        if ~showPast
            a = s;                  % solo il frame corrente
        elseif isfinite(trailFrames)
            j = max(1, k - trailFrames + 1);    % scia di trailFrames
            a = cumEnd(j) - counts(j) + 1;
        else
            a = 1;                  % tutti i frame passati
        end

        m = n - a + 1;
        if m <= 0
            set(hAll, 'XData', nan, 'YData', nan, 'ZData', nan, 'CData', 0);
        else
            % Oltre maxRenderPoints si disegna un punto ogni 'stride':
            % serve solo a tenere fluida la riproduzione, i dati non cambiano.
            stride = max(1, ceil(m / maxRenderPoints));
            ii = a:stride:n;
            set(hAll, 'XData', XYZ(ii, 1), 'YData', XYZ(ii, 2), ...
                'ZData', XYZ(ii, 3), 'CData', cData(ii));
        end

        % Senza frame passati l'evidenziazione coprirebbe di bianco tutto
        % cio' che si vede, quindi li' non ha senso.
        if showNew && showPast && counts(k) > 0
            set(hNew, 'XData', XYZ(s:n, 1), 'YData', XYZ(s:n, 2), ...
                'ZData', XYZ(s:n, 3), 'Visible', 'on');
        else
            set(hNew, 'Visible', 'off');
        end

        set(hSlider, 'Value', k);
        set(hStatus, 'String', sprintf( ...
            'frame %d/%d   t = %6.2f s   punti frame %d   a schermo %d', ...
            k, nF, tRel(k), counts(k), max(m, 0)));
        if ~showPast
            set(hTitle, 'String', sprintf('frame %d/%d, %d punti', k, nF, m));
        elseif isfinite(trailFrames)
            set(hTitle, 'String', sprintf('frame %d, scia di %d, %d punti', ...
                k, min(trailFrames, k), m));
        else
            set(hTitle, 'String', sprintf('%d frame accumulati, %d punti', k, n));
        end
        drawnow limitrate
    end

    function onTick()
        if k >= nF
            if loopPlay
                setFrame(1);
            else
                setPlaying(false);
            end
            return
        end
        setFrame(k + 1);
    end

    function togglePlay()
        if ~playing && k >= nF
            k = 1;    % ripartenza automatica se si e' fermi alla fine
        end
        setPlaying(~playing);
    end

    function setPlaying(tf)
        playing = tf;
        if tf
            if strcmp(tmr.Running, 'off')
                start(tmr);
            end
            set(hPlay, 'String', 'Pausa');
        else
            if strcmp(tmr.Running, 'on')
                stop(tmr);
            end
            set(hPlay, 'String', 'Play');
        end
    end

    function onFps()
        fps = fpsList(get(hFps, 'Value'));
        wasPlaying = playing;
        setPlaying(false);
        tmr.Period = round(1 / fps, 3);
        setPlaying(wasPlaying);
    end

    function onColor()
        if get(hColor, 'Value') == 1
            cData = cQuota;
            clim(ax, [lo(3) - eps, hi(3) + eps]);
        else
            cData = cTempo;
            clim(ax, [1, max(nF, 1 + eps)]);
        end
        render();
    end

    function onPast(tf)
        showPast = logical(tf);
        set(hPast, 'Value', showPast);   % tiene allineata la checkbox al tasto p
        % Senza frame passati non c'e' nulla da evidenziare.
        if showPast
            set(hHigh, 'Enable', 'on');
        else
            set(hHigh, 'Enable', 'off');
        end
        render();
    end

    function onHighlight(s)
        showNew = logical(get(s, 'Value'));
        render();
    end

    function onLoop(s)
        loopPlay = logical(get(s, 'Value'));
    end

    function onKey(~, evt)
        switch evt.Key
            case 'space',      togglePlay();
            case 'rightarrow', stepFrame(1);
            case 'leftarrow',  stepFrame(-1);
            case 'home',       stepFrame(1 - k);
            case 'end',        stepFrame(nF - k);
            case 'p',          onPast(~showPast);
        end
    end

    function onClose()
        if isvalid(tmr)
            stop(tmr);
            delete(tmr);
        end
        delete(fig);
    end
end

%% --- funzioni di utilita' ---------------------------------------------

function st = sliderStepFor(nF)
% Passo dello slider: un frame con le frecce, dieci con il click sulla barra.
if nF > 1
    st = [1 / (nF - 1), min(1, 10 / (nF - 1))];
else
    st = [1 1];
end
end

function i = nearestIdx(v, x)
[~, i] = min(abs(v - x));
end
