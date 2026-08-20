%% Correzione mappa su bag FAST-LIO gia' processata, SENZA loop closure
%
% Variante di LoopClosure_v2.m che rimuove interamente la ricerca/verifica
% dei loop (Scan Context + ICP + vincoli di loop nel pose graph) e usa SOLO:
%   - il vincolo di gravita' (riallineamento assetto sul pavimento + fattore
%     di gravita' nel pose graph)
%   - i vincoli temporali (taglio finestra traiettoria, troncamento alla
%     prima divergenza odometria/velocita' implausibile)
%   - le altre correzioni di deriva non basate sui loop (yaw sui muri, quota
%     Z diretta dal pavimento)
% In piu' aggiunge un filtro di rimozione outlier (statistico, pcdenoise),
% sia a livello di singolo keyframe sia sulla mappa finale ricostruita.
%
% APPROCCIO
% 1. Si leggono le pose da /Odometry
% 2. Si riportano le nuvole nel frame body invertendo la posa (un-transform)
% 3. Si selezionano keyframe, si filtrano gli outlier per nuvola
% 4. Si riallinea l'assetto a gravita' (pavimento) e lo yaw (muri)
% 5. Si costruisce un pose graph con SOLI vincoli sequenziali + gravita'
% 6. Si ottimizza e si ricostruisce la mappa con le pose corrette
%
% REQUISITI: ROS Toolbox, Lidar Toolbox, Navigation Toolbox, Computer Vision Toolbox

clear
close all
clc

%% 1. Parametri
bagPath = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\rosbag2_2026_07_30-17_50_45_0.db3";

% Selezione keyframe: un nuovo keyframe quando ci si e' spostati di
% kfDistance metri OPPURE ruotati di kfAngle gradi rispetto al precedente.
kfDistance = 1.0;    % m
kfAngle    = 15;     % gradi

% Taglio della traiettoria a una finestra [inizio, fine] in secondi.
% Vincolo TEMPORALE: utile sia per scartare la coda del percorso (es. il
% ritorno che si sovrappone al corridoio di andata, generando doppioni
% nella mappa) sia per isolare un tratto, senza indovinare una soglia
% spaziale: una coordinata X puo' essere attraversata piu' volte in modo
% imprevedibile (curve, avanti-indietro locale), il tempo trascorso no.
useTimeCut      = true;
timeCutStartSec = 10;   % s, scarta i keyframe con tempo trascorso < questo (NaN = dall'inizio)
timeCutEndSec   = 295;  % s, scarta i keyframe con tempo trascorso > questo (NaN = fino alla fine)

% Downsampling applicato a ogni keyframe
kfVoxel = 0.20;   % m

% --- Filtro outlier (nuovo) -------------------------------------------
% Rimozione statistica degli outlier: per ogni punto si guarda la distanza
% media dai suoi NumNeighbors vicini piu' prossimi; i punti la cui distanza
% media supera media + Threshold*deviazione_standard (calcolate su tutta la
% nuvola) vengono scartati. Applicato in due punti della pipeline:
%   - per keyframe, appena estratto (aiuta anche i fit successivi: pavimento
%     6b, normali dei muri 6c);
%   - sulla mappa finale ricostruita, dove compaiono anche outlier "di
%     giunzione" tra scansioni diverse che non esistono nel singolo keyframe.
useOutlierFilterKF    = true;
outlierKFNumNeighbors = 8;      % punti vicini usati per la statistica
outlierKFStdFactor    = 0.5;    % soglia in deviazioni standard, piu' basso = piu' aggressivo

useOutlierFilterMap    = true;
outlierMapNumNeighbors = 20;
outlierMapStdFactor    = 0.5;

% Riallineamento assetto sul pavimento: vincolo di GRAVITA'.
% Correzione della deriva di roll/pitch usando il pavimento come riferimento
% di gravita'. Da disattivare solo se l'ambiente NON ha un pavimento piano
% (esterni, terreno irregolare, rampe continue).
useGravityAlign = true;
floorBand    = 0.30;   % m, spessore della fascia bassa in cui cercare il pavimento
floorTol     = 0.06;   % m, tolleranza di planarita' del fit
floorMaxTilt = 50;     % gradi, inclinazione max accettata per il piano trovato

% Correzione della deriva di YAW sui muri.
% Il pavimento vincola roll e pitch ma NON lo yaw: i corridoi restano non
% perpendicolari in pianta. I muri sono il riferimento per il terzo DOF.
% ATTENZIONE: a differenza del pavimento, questa correzione ASSUME che
% l'edificio sia ortogonale. Controllare sempre la dispersione stampata: se
% non cala nettamente, l'ipotesi non regge e va disattivata.
useYawAlign = true;
wallMaxNz   = 0.2;   % |nz| sotto cui una normale e' considerata di muro
yawRefKF    = 40;    % keyframe iniziali usati come riferimento di azimut
yawSmooth   = 9;     % finestra di lisciatura della stima, in keyframe

% Vincolo di gravita' DENTRO il pose graph.
% Senza, l'ottimizzazione puo' comunque discostare l'assetto dal
% riallineamento fatto sopra per soddisfare al meglio la sola catena
% odometrica (rumore locale). poseGraph3D non ha fattori unari (prior), ma
% il nodo 1 e' fissato dall'ottimizzatore: un vincolo 1->k con misura pari
% alla posa assoluta desiderata di k si comporta come un prior su k. Per
% vincolare SOLO roll e pitch, lasciando liberi x, y, z e yaw, si usa una
% information matrix anisotropa: peso alto su rx,ry e peso quasi nullo (ma
% positivo, deve restare definita positiva) sugli altri DOF.
useGravityFactor = true;
infoGravRP   = 50;     % peso su roll/pitch (confrontabile con infoOdom = 100)
infoGravFree = 1e-6;   % peso sui DOF liberi (x,y,z,yaw): quasi nullo ma > 0

% Vincolo di quota Z sulla traiettoria, dallo stesso pavimento di sopra.
% floorDist e' una misura LOCALE fresca ad ogni keyframe (non porta dentro
% la deriva accumulata): corregge la deriva Z residua che, senza loop
% closure, nient'altro vincola.
% floorFitMin: frazione minima di keyframe con fit pavimento valido perche'
% il vincolo si attivi; sotto quella soglia il pavimento non e' abbastanza
% osservato e si lascia solo il vincolo di roll/pitch di sopra.
floorFitMin = 0.5;
infoGravZ   = 1e-6;

% Peso dei vincoli odometrici nel pose graph.
infoOdom = 100;

% Voxel per la mappa finale
mapVoxel = 0.05;    % m

% Crop geometrico (ROI) della mappa finale.
% Coordinate nel frame mappa; usare Inf/-Inf per lasciare un asse libero.
% Lo script stampa l'estensione della mappa prima di applicarlo, cosi' da
% poter scegliere i limiti al primo giro con useMapROI = false.
useMapROI = true;
mapROI = [-Inf Inf, ...     % X min max
          -Inf Inf, ...     % Y min max
          -1 4];             % Z min max

% Rilevamento divergenza odometria (IMU/FAST-LIO che perde il tracking).
% Vincolo TEMPORALE/cinematico: un salto di posizione tra due messaggi
% consecutivi fisicamente impossibile per la velocita' del sensore segna il
% punto da cui le pose non sono piu' fisiche e vanno scartate, non corrette.
odomMaxSpeed = 5.0;    % m/s, velocita' massima plausibile del sensore
odomMaxJump  = 5.0;    % m, salto massimo assoluto tollerato indipendentemente da dt

%% 2. Lettura bag
bag = ros2bagreader(bagPath);

selOdom  = select(bag, 'Topic', '/Odometry');
selCloud = select(bag, 'Topic', '/cloud_registered');

fprintf('Odometry:         %d messaggi\n', selOdom.NumMessages);
fprintf('cloud_registered: %d messaggi\n', selCloud.NumMessages);

odomMsgs  = readMessages(selOdom);
cloudMsgs = readMessages(selCloud);

%% 3. Conversione odometria in trasformazioni omogenee
% nav_msgs/Odometry: pose.pose.position + pose.pose.orientation (quaternione)
nOdom = numel(odomMsgs);
posesRaw = repmat(rigidtform3d, nOdom, 1);
tOdom = zeros(nOdom, 1);

for i = 1:nOdom
    m = odomMsgs{i};
    p = m.pose.pose.position;
    q = m.pose.pose.orientation;

    % quat2rotm vuole l'ordine [w x y z], ROS espone x,y,z,w
    R = quat2rotm([q.w q.x q.y q.z]);
    t = [p.x p.y p.z];

    posesRaw(i) = rigidtform3d(R, t);
    tOdom(i) = double(m.header.stamp.sec) + double(m.header.stamp.nanosec)*1e-9;
end

%% 3b. Troncamento alla prima divergenza dell'odometria (vincolo temporale)
% Un salto di posizione troppo grande in troppo poco tempo non e' deriva:
% e' l'IMU che ha perso il tracking (superficie riflettente, urto, tratto
% senza feature). Da quel messaggio in poi tutte le pose sono inattendibili
% e vengono scartate.
trans = vertcat(posesRaw.Translation);
dStep = vecnorm(diff(trans), 2, 2);
dt    = diff(tOdom);
dt(dt <= 0) = eps;   % evita divisioni per zero/negative su timestamp non monotoni
speedStep = dStep ./ dt;

divergeIdx = find(speedStep > odomMaxSpeed | dStep > odomMaxJump, 1, 'first');

if ~isempty(divergeIdx)
    fprintf(2, ['\nATTENZIONE: divergenza odometria rilevata al messaggio %d/%d ' ...
        '(salto %.1f m in %.3f s, velocita'' implicita %.1f m/s).\n' ...
        'Pose troncate da qui in poi: %d messaggi scartati su %d.\n'], ...
        divergeIdx + 1, nOdom, dStep(divergeIdx), dt(divergeIdx), ...
        speedStep(divergeIdx), nOdom - divergeIdx, nOdom);

    posesRaw = posesRaw(1:divergeIdx);
    tOdom    = tOdom(1:divergeIdx);
    nOdom    = divergeIdx;
end

fprintf('Durata odometria utilizzabile: %.1f s\n', tOdom(end) - tOdom(1));

% Timestamp delle nuvole, per l'associazione
nCloud = numel(cloudMsgs);
tCloud = zeros(nCloud, 1);
for i = 1:nCloud
    h = cloudMsgs{i}.header.stamp;
    tCloud(i) = double(h.sec) + double(h.nanosec)*1e-9;
end

%% 4. Associazione nuvola <-> posa per timestamp
% Non si assume l'allineamento per indice: si cerca la posa piu' vicina nel
% tempo a ciascuna nuvola e si scartano gli accoppiamenti troppo distanti.
maxDt = 0.05;   % s
pairCloudIdx = [];
pairOdomIdx  = [];

for i = 1:nCloud
    [dt, j] = min(abs(tOdom - tCloud(i)));
    if dt <= maxDt
        pairCloudIdx(end+1) = i;   %#ok<SAGROW>
        pairOdomIdx(end+1)  = j;   %#ok<SAGROW>
    end
end

fprintf('Coppie nuvola/posa associate: %d (scartate %d)\n', ...
    numel(pairCloudIdx), nCloud - numel(pairCloudIdx));

if isempty(pairCloudIdx)
    error(['Nessuna associazione trovata. Verificare che i timestamp dei due ' ...
        'topic siano coerenti, oppure alzare maxDt.']);
end

%% 5. Selezione keyframe
% Un nuovo keyframe quando si supera la soglia in traslazione o rotazione.
kfSel = 1;   % il primo e' sempre keyframe
lastT = posesRaw(pairOdomIdx(1)).Translation;
lastR = posesRaw(pairOdomIdx(1)).R;

for k = 2:numel(pairCloudIdx)
    T = posesRaw(pairOdomIdx(k)).Translation;
    R = posesRaw(pairOdomIdx(k)).R;

    dTrans = norm(T - lastT);
    % angolo della rotazione relativa, da traccia della matrice
    dR = lastR' * R;
    dAng = abs(rad2deg(acos(max(-1, min(1, (trace(dR) - 1) / 2)))));

    if dTrans >= kfDistance || dAng >= kfAngle
        kfSel(end+1) = k;   %#ok<SAGROW>
        lastT = T;
        lastR = R;
    end
end

nKF = numel(kfSel);
fprintf('Keyframe selezionati: %d su %d frame\n', nKF, numel(pairCloudIdx));

%% 5b. Taglio della traiettoria a una finestra [inizio, fine] in secondi
% Vincolo TEMPORALE, vedi nota su useTimeCut in Sezione 1. Va PRIMA
% dell'estrazione nuvole (Sezione 6): i keyframe scartati non vengono
% nemmeno letti dalla bag. A differenza di un taglio su coordinata
% spaziale, il tempo trascorso e' monotono per costruzione: nessun rischio
% di attraversamenti multipli o ambigui. NaN su un estremo lascia quel lato
% aperto.
if useTimeCut
    tKF = tOdom(pairOdomIdx(kfSel));
    elapsed = tKF - tKF(1);   % secondi dall'inizio DELLA BAG, non della finestra
    keep = true(size(elapsed));
    if ~isnan(timeCutStartSec), keep = keep & elapsed >= timeCutStartSec; end
    if ~isnan(timeCutEndSec),   keep = keep & elapsed <= timeCutEndSec;   end

    if all(keep)
        warning(['useTimeCut attivo ma la finestra [%.1f, %.1f] s copre l''intera ' ...
            'traiettoria (0-%.1f s): nessun taglio applicato.'], ...
            timeCutStartSec, timeCutEndSec, elapsed(end));
    elseif ~any(keep)
        error(['La finestra [%.1f, %.1f] s non contiene nessun keyframe (traiettoria: ' ...
            '0-%.1f s). Controllare i limiti.'], timeCutStartSec, timeCutEndSec, elapsed(end));
    else
        keepIdx = find(keep);
        fprintf('Taglio traiettoria: tenuti i keyframe tra %.1f e %.1f s (%d/%d keyframe)\n', ...
            elapsed(keepIdx(1)), elapsed(keepIdx(end)), numel(keepIdx), numel(kfSel));
        kfSel = kfSel(keepIdx);
        nKF = numel(kfSel);
    end
end

%% 6. Estrazione nuvole nel frame body + filtro outlier per keyframe
% /cloud_registered e' nel frame mappa: si inverte la posa per tornare al
% frame sensore. Il filtro outlier (Sezione 1) e' applicato dopo il
% downsampling: piu' veloce, e la densita' uniforme del voxel grid rende la
% statistica dei vicini piu' stabile.
kfClouds = cell(nKF, 1);
kfPoses  = repmat(rigidtform3d, nKF, 1);

fprintf('Estrazione nuvole keyframe...\n');
nOutlierKFTotal = 0;
for k = 1:nKF
    ci = pairCloudIdx(kfSel(k));
    oi = pairOdomIdx(kfSel(k));

    xyz = rosReadXYZ(cloudMsgs{ci});
    xyz = xyz(all(isfinite(xyz), 2), :);
    pcMap = pointCloud(xyz);

    % un-transform: dal frame mappa al frame body
    pcBody = pctransform(pcMap, invert(posesRaw(oi)));

    pcDown = pcdownsample(pcBody, 'gridAverage', kfVoxel);

    if useOutlierFilterKF && pcDown.Count > outlierKFNumNeighbors
        nBefore = pcDown.Count;
        pcDown = pcdenoise(pcDown, ...
            'NumNeighbors', outlierKFNumNeighbors, ...
            'Threshold', outlierKFStdFactor);
        nOutlierKFTotal = nOutlierKFTotal + (nBefore - pcDown.Count);
    end

    kfClouds{k} = pcDown;
    kfPoses(k)  = posesRaw(oi);
end
if useOutlierFilterKF
    fprintf('  filtro outlier per keyframe: %d punti rimossi in totale\n', nOutlierKFTotal);
end

% Copia dell'odometria GREZZA, senza alcun vincolo (ne' gravita', ne' yaw,
% ne' Z, ne' pose graph): serve solo come riferimento "PRIMA" in Sezione 9,
% per mostrare l'effetto di TUTTE le correzioni insieme rispetto a nessuna
% correzione. kfPoses viene invece modificato in-place dalle sezioni
% seguenti (rigidtform3d e' una value class: questa e' una copia vera).
kfPosesOdomRaw = kfPoses;

%% 6b. Riallineamento dell'assetto sul piano del pavimento (vincolo di gravita')
% FAST-LIO e' allineato a gravita' (l'IMU la osserva), quindi la normale del
% pavimento, riportata nel frame mappa, deve restare verticale lungo tutto
% il percorso. Quando invece si inclina progressivamente, quella e' deriva
% di assetto: la mappa "ruota" e un corridoio piano sembra scendere.
%
% Si impone quindi roll e pitch dal pavimento (2 DOF, privi di deriva per
% costruzione) e si lasciano yaw e spostamento all'odometria, che su quelli
% e' affidabile. La traiettoria viene re-integrata con gli assetti corretti.
if useGravityAlign
    fprintf('Riallineamento assetto sul pavimento...\n');

    % La fascia di ricerca parte da un PERCENTILE basso, non da min(z): un
    % singolo punto spurio sotto il pavimento sposterebbe la fascia nel
    % vuoto e il fit fallirebbe.
    nBody = nan(nKF, 3);
    % floorDist(k): distanza con segno origine-sensore -> piano pavimento,
    % nel frame BODY. Riusata in Sezione 6d per il vincolo di quota Z sulla
    % traiettoria, indipendente dal fit di normale qui sopra.
    floorDist = nan(nKF, 1);
    for k = 1:nKF
        loc = kfClouds{k}.Location;
        if size(loc,1) < 80, continue; end
        zref = prctile(loc(:,3), 2);
        cand = loc(loc(:,3) > zref - 0.12 & loc(:,3) < zref + floorBand, :);
        if size(cand,1) < 50, continue; end
        try
            [model, inl] = pcfitplane(pointCloud(cand), floorTol, [0 0 1], floorMaxTilt);
            if numel(inl) < 40, continue; end
        catch
            continue
        end
        n = model.Normal(:);
        if n(3) < 0, n = -n; end
        nBody(k,:) = n' / norm(n);
        floorDist(k) = -model.Parameters(4) / norm(model.Parameters(1:3));
    end
    validFloor = ~any(isnan(nBody), 2);
    fprintf('  normali pavimento valide: %d su %d keyframe\n', nnz(validFloor), nKF);

    % Rotazione di correzione dove il pavimento e' stato misurato
    qC = nan(nKF, 4);
    tiltBefore = nan(nKF, 1);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        nMap = kfPoses(k).R * nBody(k,:)';
        if nMap(3) < 0, nMap = -nMap; end
        nMap = nMap / norm(nMap);
        tiltBefore(k) = rad2deg(acos(max(-1, min(1, nMap(3)))));

        ax = cross(nMap, [0;0;1]);
        s  = norm(ax);
        c  = dot(nMap, [0;0;1]);
        if s > 1e-8
            ax  = ax / s;
            ang = atan2(s, c);
            K   = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
            C   = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);   % Rodrigues
        else
            C = eye(3);
        end
        qC(k,:) = rotm2quat(C);
    end

    % Nei buchi la correzione viene INTERPOLATA tra i due estremi validi:
    % congelarla all'ultimo valore noto lascia riaccumulare l'errore.
    vi = find(validFloor);
    if isempty(vi)
        error(['Nessun pavimento trovato in nessun keyframe: impossibile ' ...
            'riallineare. Disattivare useGravityAlign o rivedere floorBand.']);
    end
    for k = 1:nKF
        if validFloor(k), continue; end
        prev = vi(find(vi < k, 1, 'last'));
        next = vi(find(vi > k, 1, 'first'));
        if isempty(prev)
            qC(k,:) = qC(next,:);
        elseif isempty(next)
            qC(k,:) = qC(prev,:);
        else
            t = (k - prev) / (next - prev);
            qC(k,:) = slerpQuat(qC(prev,:), qC(next,:), t);
        end
    end

    Rc = cell(nKF,1);
    for k = 1:nKF
        Rc{k} = quat2rotm(qC(k,:)) * kfPoses(k).R;
    end

    % Re-integrazione delle posizioni: lo spostamento in frame body viene
    % dall'odometria, la direzione in cui applicarlo dall'assetto corretto.
    pOld = vertcat(kfPoses.Translation);
    pNew = zeros(nKF,3);
    pNew(1,:) = pOld(1,:);
    for k = 2:nKF
        dLocal = kfPoses(k-1).R' * (pOld(k,:) - pOld(k-1,:))';
        pNew(k,:) = pNew(k-1,:) + (Rc{k-1} * dLocal)';
    end

    for k = 1:nKF
        kfPoses(k) = rigidtform3d(Rc{k}, pNew(k,:));
    end

    % Tilt residuo: e' la verifica che la correzione abbia fatto il suo
    % lavoro. Se non scende vicino a zero, il pavimento non e' un buon
    % riferimento in questo ambiente (rampe, terreno irregolare).
    tiltAfter = nan(nKF,1);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        nMap = Rc{k} * nBody(k,:)';
        if nMap(3) < 0, nMap = -nMap; end
        tiltAfter(k) = rad2deg(acos(max(-1, min(1, nMap(3)))));
    end

    fprintf('  inclinazione pavimento: mediana %.2f -> %.2f deg, max %.2f -> %.2f deg\n', ...
        median(tiltBefore(~isnan(tiltBefore))), median(tiltAfter(~isnan(tiltAfter))), ...
        max(tiltBefore), max(tiltAfter));
    fprintf('  deriva Z traiettoria: %.2f m -> %.2f m\n', ...
        max(pOld(:,3))-min(pOld(:,3)), max(pNew(:,3))-min(pNew(:,3)));
end

%% 6c. Correzione della deriva di YAW sulla direzione dei muri
% Il pavimento vincola solo 2 DOF su 3: la normale di un piano orizzontale
% e' invariante per rotazione attorno alla verticale, quindi dice dove e'
% "su" ma non dove e' "nord". Lo yaw resta libero di derivare, e il sintomo
% e' che i corridoi non si incontrano piu' ad angolo retto in pianta.
%
% Riferimento per lo yaw sono i MURI: si prendono le normali quasi
% orizzontali e se ne calcola l'azimut dominante, ripiegato modulo 90 gradi
% (cosi' i quattro lati di un corridoio ortogonale cadono sullo stesso
% valore). La correzione riporta quell'azimut al valore di riferimento.
%
% ATTENZIONE, questa NON e' una misura pura come il pavimento: assume che
% l'edificio abbia una direzione dominante coerente. Verificare sempre le
% due stampe di controllo: se la dispersione NON cala nettamente, i muri
% non appartengono a una sola famiglia ortogonale e va disattivata.
if useYawAlign
    fprintf('Correzione deriva di yaw sui muri...\n');

    azi = nan(nKF,1);
    for k = 1:nKF
        pc = kfClouds{k};
        if pc.Count < 200, continue; end
        try
            nrm = pcnormals(pc, 20);
        catch
            continue
        end
        nMap   = (kfPoses(k).R * nrm')';
        isWall = abs(nMap(:,3)) < wallMaxNz;     % normale orizzontale => muro
        if nnz(isWall) < 50, continue; end
        a = atan2(nMap(isWall,2), nMap(isWall,1));
        % x4 porta il periodo da 90 a 360 gradi: cosi' la media circolare e'
        % ben definita e non soffre del salto 0/90.
        azi(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
    end
    validWall = ~isnan(azi);
    fprintf('  azimut stimato su %d keyframe su %d\n', nnz(validWall), nKF);

    if nnz(validWall) < 10
        warning(['Troppi pochi keyframe con muri: correzione yaw saltata. ' ...
            'Ambiente probabilmente aperto o poco strutturato.']);
    else
        z = nan(nKF,1) + 1i*nan;
        z(validWall) = exp(1i*4*deg2rad(azi(validWall)));

        % Riferimento: media circolare dei primi keyframe, prima che la
        % deriva si manifesti. Ancorare all'inizio conserva l'orientamento
        % originale della mappa.
        nRef = min(yawRefKF, nKF);
        zRef = mean(z(1:nRef), 'omitnan');

        % Lisciatura circolare: la stima per singolo keyframe e' rumorosa e
        % non va inseguita, la deriva e' un fenomeno lento.
        zS = nan(nKF,1) + 1i*nan;
        hw = floor(yawSmooth/2);
        for k = 1:nKF
            w = z(max(1,k-hw):min(nKF,k+hw));
            w = w(~isnan(w));
            if ~isempty(w), zS(k) = mean(w); end
        end
        vw = find(~isnan(zS));
        for k = 1:nKF
            if ~isnan(zS(k)), continue; end
            [~, i] = min(abs(vw - k));
            zS(k) = zS(vw(i));
        end

        dYaw = zeros(nKF,1);
        for k = 1:nKF
            d = rad2deg(angle(zS(k) / zRef)) / 4;
            dYaw(k) = mod(d + 45, 90) - 45;     % riporta in [-45, 45]
        end

        pOldY = vertcat(kfPoses.Translation);
        RcY   = cell(nKF,1);
        for k = 1:nKF
            th = -deg2rad(dYaw(k));
            Cz = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
            RcY{k} = Cz * kfPoses(k).R;
        end

        pNewY = zeros(nKF,3);
        pNewY(1,:) = pOldY(1,:);
        for k = 2:nKF
            dLocal = kfPoses(k-1).R' * (pOldY(k,:) - pOldY(k-1,:))';
            pNewY(k,:) = pNewY(k-1,:) + (RcY{k-1}*dLocal)';
        end

        for k = 1:nKF
            kfPoses(k) = rigidtform3d(RcY{k}, pNewY(k,:));
        end

        % Controllo: la dispersione circolare deve calare nettamente.
        aziAfter = nan(nKF,1);
        for k = 1:nKF
            pc = kfClouds{k};
            if pc.Count < 200, continue; end
            try
                nrm = pcnormals(pc, 20);
            catch
                continue
            end
            nMap   = (RcY{k} * nrm')';
            isWall = abs(nMap(:,3)) < wallMaxNz;
            if nnz(isWall) < 50, continue; end
            a = atan2(nMap(isWall,2), nMap(isWall,1));
            aziAfter(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
        end
        cdisp = @(v) 1 - abs(mean(exp(1i*4*deg2rad(v(~isnan(v))))));

        fprintf('  correzione applicata: da %.2f a %.2f deg\n', min(dYaw), max(dYaw));
        fprintf('  dispersione direzione muri: %.3f -> %.3f  (piu'' basso = piu'' coerente)\n', ...
            cdisp(azi), cdisp(aziAfter));
    end
end

%% 6d. Correzione diretta della quota Z dal pavimento
% floorDist (Sezione 6b) e' una misura locale, non affetta dalla deriva
% Z accumulata: qui si applica direttamente a kfPoses. Il vincolo nel pose
% graph (Sezione 7) resta comunque utile per difendere questa quota durante
% l'ottimizzazione.
%
% Nota: dopo il livellamento di roll/pitch (6b), le rotazioni di 6c sono
% pura rotazione attorno a Z (yaw): non mescolano la componente Z della
% traslazione, quindi applicare questa correzione qui (dopo 6c) invece che
% prima da' lo stesso risultato numerico.
floorFitRate = nnz(validFloor) / nKF;
useGravityZ  = useGravityAlign && floorFitRate >= floorFitMin && validFloor(1);
if useGravityZ
    pZ = vertcat(kfPoses.Translation);
    floorWorldZPre = pZ(:,3) - floorDist;
    fprintf('\nQuota pavimento (pre-correzione): mediana %.3f m, std %.3f m (su %d/%d keyframe)\n', ...
        median(floorWorldZPre(validFloor)), std(floorWorldZPre(validFloor)), nnz(validFloor), nKF);

    z1 = kfPoses(1).Translation(3);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        p = kfPoses(k).Translation;
        p(3) = z1 + (floorDist(k) - floorDist(1));
        kfPoses(k) = rigidtform3d(kfPoses(k).R, p);
    end

    pZ = vertcat(kfPoses.Translation);
    floorWorldZPost = pZ(:,3) - floorDist;
    fprintf('Quota pavimento (post-correzione): mediana %.3f m, std %.3f m (su %d/%d keyframe)\n', ...
        median(floorWorldZPost(validFloor)), std(floorWorldZPost(validFloor)), nnz(validFloor), nKF);
    fprintf('Correzione Z diretta applicata a %d/%d keyframe (peso %.0f%% rilevamento pavimento)\n', ...
        nnz(validFloor), nKF, 100*floorFitRate);
else
    fprintf(['\nPavimento rilevato in %.0f%% dei keyframe (< %.0f%% richiesto) o nodo 1 senza fit: ' ...
        'nessuna correzione Z diretta\n'], 100*floorFitRate, 100*floorFitMin);
end

% Checkpoint: la lettura bag e l'estrazione keyframe sono costose e non
% dipendono dai pesi del pose graph. Si salva qui per poter iterare senza
% rifare tutto da capo.
checkpointFile = fullfile(fileparts(bagPath), 'noloop_checkpoint.mat');
save(checkpointFile, 'kfPoses', 'nKF', 'kfVoxel', 'mapVoxel', 'kfClouds', ...
    'pairCloudIdx', 'pairOdomIdx', 'kfSel', '-v7.3');
fprintf('Checkpoint salvato in: %s\n', checkpointFile);

%% 7. Costruzione del pose graph (solo odometria + gravita', NESSUN loop)
pg = poseGraph3D;

% Information matrix: 21 elementi, triangolo superiore di una 6x6.
% Diagonale = [x y z rx ry rz], valori piu' alti = vincolo piu' rigido.
infoVecOdom = buildInfoVector(infoOdom);

% Vincoli sequenziali dall'odometria
for k = 2:nKF
    Trel = kfPoses(k-1).A \ kfPoses(k).A;
    addRelativePose(pg, tform2measurement(Trel), infoVecOdom, k-1, k);
end

% Vincoli di gravita': senza questi, l'ottimizzazione della sola catena
% odometrica potrebbe comunque discostare l'assetto dal riallineamento
% fatto in 6b/6d.
if useGravityAlign && useGravityFactor
    if useGravityZ
        fprintf('Vincolo Z nel pose graph attivato (peso %g)\n', infoGravZ);
    else
        fprintf('Nessun vincolo Z nel pose graph (vedi Sezione 6d)\n');
    end

    T0inv = kfPoses(1).A \ eye(4);
    nZCorr = 0;
    for k = 2:nKF
        Ak = T0inv * kfPoses(k).A;    % posa di k relativa al nodo 1, gia' livellata
        wZ = infoGravFree;
        if useGravityZ && validFloor(k)
            Ak(3,4) = floorDist(k) - floorDist(1);   % target Z dal pavimento, non da poseZ (niente deriva)
            wZ = infoGravZ;
            nZCorr = nZCorr + 1;
        end
        addRelativePose(pg, tform2measurement(Ak), ...
            buildInfoVectorAniso(infoGravFree, infoGravRP, wZ), 1, k);
    end
    fprintf('Vincoli di gravita aggiunti: %d (peso roll/pitch %g, Z da pavimento su %d/%d)\n', ...
        nKF-1, infoGravRP, nZCorr, nKF-1);
end

fprintf('\nPose graph: %d nodi, %d vincoli (nessun loop)\n', pg.NumNodes, pg.NumEdges);

%% 8. Ottimizzazione
fprintf('Ottimizzazione...\n');
pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');

%% 9. Ricostruzione mappa con pose corrette
% IMPORTANTE: poseGraph3D ancora SEMPRE il nodo 1 all'origine con
% orientamento identita', mentre kfPoses(1) ha posizione e assetto propri.
% Senza riportare il risultato nel frame di partenza, "prima" e "dopo"
% vivono in due frame globali diversi, ruotati tra loro.
nodesOpt = nodeEstimates(pgOpt);

T0 = kfPoses(1).A;               % frame del primo keyframe
posesOpt = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF
    n  = nodesOpt(k, :);         % nodeEstimates restituisce [x y z qw qx qy qz]
    Ak = eye(4);
    Ak(1:3,1:3) = quat2rotm(n(4:7));
    Ak(1:3,4)   = n(1:3)';
    Ak = T0 * Ak;                % riporto nel frame di partenza
    posesOpt(k) = rigidtform3d(Ak(1:3,1:3), Ak(1:3,4)');
end
nodesOpt = [vertcat(posesOpt.Translation), ...
            cell2mat(arrayfun(@(p) rotm2quat(p.R), posesOpt, 'UniformOutput', false))];

allXYZ = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, posesOpt(k));
    allXYZ{k} = pcT.Location;
end

pcOpt = pointCloud(vertcat(allXYZ{:}));
pcOpt = pcdownsample(pcOpt, 'gridAverage', mapVoxel);

% Mappa originale: odometria GREZZA, senza NESSUN vincolo (ne' gravita', ne'
% yaw, ne' Z, ne' pose graph) - vedi kfPosesOdomRaw in Sezione 6. E' il vero
% "PRIMA": il confronto con pcOpt mostra l'effetto di TUTTE le correzioni
% insieme, non solo del pose graph.
allXYZraw = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, kfPosesOdomRaw(k));
    allXYZraw{k} = pcT.Location;
end
pcRaw = pointCloud(vertcat(allXYZraw{:}));
pcRaw = pcdownsample(pcRaw, 'gridAverage', mapVoxel);

% Filtro outlier sulla mappa finale (Sezione 1): qui compaiono anche
% outlier "di giunzione" tra scansioni diverse che non esistono nel singolo
% keyframe. Applicato a entrambe le mappe, per un confronto coerente.
if useOutlierFilterMap
    nOptBefore = pcOpt.Count;
    nRawBefore = pcRaw.Count;
    pcOpt = pcdenoise(pcOpt, 'NumNeighbors', outlierMapNumNeighbors, 'Threshold', outlierMapStdFactor);
    pcRaw = pcdenoise(pcRaw, 'NumNeighbors', outlierMapNumNeighbors, 'Threshold', outlierMapStdFactor);
    fprintf('\nFiltro outlier sulla mappa:\n');
    fprintf('  mappa corretta: %d -> %d punti (%.1f%% rimosso)\n', ...
        nOptBefore, pcOpt.Count, 100*(nOptBefore - pcOpt.Count)/nOptBefore);
    fprintf('  mappa grezza:   %d -> %d punti (%.1f%% rimosso)\n', ...
        nRawBefore, pcRaw.Count, 100*(nRawBefore - pcRaw.Count)/nRawBefore);
end

%% 9b. Crop geometrico (ROI) sulla mappa finale
% ATTENZIONE al punto in cui si applica. Il ROI va QUI, sulla mappa gia'
% ricostruita, non sulle nuvole keyframe: quelle sono in frame body e
% servono al fit del pavimento/muri. Ritagliarle degraderebbe il
% riallineamento.
fprintf('\n--- Estensione della mappa (per scegliere il ROI) ---\n');
fprintf('  X: %7.2f  %7.2f\n', pcOpt.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcOpt.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcOpt.ZLimits);

if useMapROI
    nBeforeOpt = pcOpt.Count;
    nBeforeRaw = pcRaw.Count;

    % Il crop si applica a ENTRAMBE le mappe: confrontare una zona ritagliata
    % con la mappa intera renderebbe il confronto prima/dopo privo di senso.
    pcOpt = select(pcOpt, findPointsInROI(pcOpt, mapROI));
    pcRaw = select(pcRaw, findPointsInROI(pcRaw, mapROI));

    fprintf('ROI applicato [%g %g, %g %g, %g %g]\n', mapROI);
    fprintf('  mappa corretta: %d -> %d punti (%.1f%% rimosso)\n', ...
        nBeforeOpt, pcOpt.Count, 100*(nBeforeOpt - pcOpt.Count)/nBeforeOpt);
    fprintf('  mappa grezza:   %d -> %d punti (%.1f%% rimosso)\n', ...
        nBeforeRaw, pcRaw.Count, 100*(nBeforeRaw - pcRaw.Count)/nBeforeRaw);

    if pcOpt.Count == 0 || pcRaw.Count == 0
        error(['Il ROI ha svuotato la mappa: nessun punto dentro i limiti. ' ...
            'Controllare che le coordinate siano nel frame mappa stampato sopra.']);
    end
end

%% 10. Confronto
% ATTENZIONE alla metrica. L'escursione Z della MAPPA non misura la deriva:
% ogni singola scansione copre gia' diversi metri in verticale, quindi il
% bounding box resta ampio anche con una traiettoria perfetta. La deriva
% vive nelle POSE, ed e' li' che va misurata.
zTrajRaw = vertcat(kfPosesOdomRaw.Translation);
zTrajRaw = zTrajRaw(:,3);
zTrajOpt = nodesOpt(:,3);

spanTrajRaw = max(zTrajRaw) - min(zTrajRaw);
spanTrajOpt = max(zTrajOpt) - min(zTrajOpt);

fprintf('\n--- Deriva verticale della TRAIETTORIA (metrica corretta) ---\n');
fprintf('Prima:  escursione Z pose %.2f m\n', spanTrajRaw);
fprintf('Dopo:   escursione Z pose %.2f m  (%+.1f%%)\n', ...
    spanTrajOpt, 100*(spanTrajOpt - spanTrajRaw)/spanTrajRaw);
fprintf('Spostamento medio dei nodi (da odometria grezza a tutte le correzioni): %.3f m\n', ...
    mean(vecnorm(nodesOpt(:,1:3) - vertcat(kfPosesOdomRaw.Translation), 2, 2)));

% Escursione Z della mappa, riportata solo come riferimento: NON e' un
% indicatore di deriva, vedi commento sopra.
fprintf('\n--- Estensione Z della mappa (NON indicatore di deriva) ---\n');
fprintf('Prima:  %7.2f  %7.2f   (escursione %.2f m)\n', ...
    pcRaw.ZLimits, diff(pcRaw.ZLimits));
fprintf('Dopo:   %7.2f  %7.2f   (escursione %.2f m)\n', ...
    pcOpt.ZLimits, diff(pcOpt.ZLimits));

figure('Color', 'k', 'Name', 'Confronto odometria grezza / tutte le correzioni');

subplot(2,1,1);
pcshow(pcRaw, 'MarkerSize', 20);
title(sprintf('PRIMA (odometria grezza, nessun vincolo), escursione Z %.2f m', diff(pcRaw.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

subplot(2,1,2);
pcshow(pcOpt, 'MarkerSize', 20);
title(sprintf('DOPO (gravita'' + yaw + Z + pose graph), escursione Z %.2f m', diff(pcOpt.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

% Traiettoria prima (odometria grezza) e dopo (tutte le correzioni)
figure('Name', 'Traiettoria');
trajRaw = vertcat(kfPosesOdomRaw.Translation);
plot3(trajRaw(:,1), trajRaw(:,2), trajRaw(:,3), 'r-', 'LineWidth', 1.5);
hold on;
plot3(nodesOpt(:,1), nodesOpt(:,2), nodesOpt(:,3), 'g-', 'LineWidth', 1.5);
legend('Prima', 'Dopo', 'Location', 'best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Traiettoria keyframe');
axis equal; grid on;

%% 11. Salvataggio
outFile = fullfile(fileparts(bagPath), 'noloop_corrected_map.pcd');
pcwrite(pcOpt, outFile, 'Encoding', 'binary');
fprintf('\nMappa salvata in:\n  %s\n', outFile);

%% Funzioni di supporto
function v = buildInfoVectorAniso(wFree, wRP, wZ)
    % Information matrix anisotropa: peso su roll e pitch, quasi nullo sul
    % resto. L'ordine della diagonale e' [x y z rx ry rz]. wFree deve essere
    % > 0: addRelativePose rifiuta matrici non definite positive. wZ
    % opzionale (default wFree): peso su Z quando il vincolo di quota
    % e' attivo per il keyframe corrente.
    if nargin < 3, wZ = wFree; end
    M = diag([wFree wFree wZ wRP wRP wFree]);
    v = zeros(1, 21);
    n = 0;
    for i = 1:6
        for j = i:6
            n = n + 1;
            v(n) = M(i, j);
        end
    end
end

function v = buildInfoVector(w)
    % Information matrix 6x6 diagonale, restituita come i 21 elementi del
    % triangolo superiore nell'ordine richiesto da addRelativePose.
    % L'ordine e' per righe: (1,1)...(1,6),(2,2)...(2,6),...,(6,6).
    % La maschera triu con indicizzazione lineare di MATLAB e' per colonne,
    % quindi metterebbe la diagonale nelle posizioni sbagliate.
    M = diag([w w w w w w]);
    v = zeros(1, 21);
    n = 0;
    for i = 1:6
        for j = i:6
            n = n + 1;
            v(n) = M(i, j);
        end
    end
end

function meas = tform2measurement(A)
    % Da matrice omogenea 4x4 a [x y z qw qx qy qz]
    R = A(1:3, 1:3);
    t = A(1:3, 4)';
    q = rotm2quat(R);
    meas = [t q];
end

function q = slerpQuat(q0, q1, t)
    % Interpolazione sferica tra due quaternioni. Scritta a mano perche'
    % quatinterp richiede l'Aerospace Toolbox, non tra i requisiti.
    q0 = q0 / norm(q0);
    q1 = q1 / norm(q1);

    c = dot(q0, q1);
    if c < 0            % percorso piu' corto sulla sfera
        q1 = -q1;
        c  = -c;
    end

    if c > 0.9995       % quasi allineati: lineare, evita la divisione instabile
        q = q0 + t*(q1 - q0);
        q = q / norm(q);
        return
    end

    th0 = acos(max(-1, min(1, c)));
    th  = th0 * t;
    q2  = q1 - q0*c;
    q2  = q2 / norm(q2);
    q   = q0*cos(th) + q2*sin(th);
end
