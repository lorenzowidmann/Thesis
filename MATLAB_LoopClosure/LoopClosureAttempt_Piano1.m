
%% Loop closure su bag FAST-LIO gia' processata
%
% PROBLEMA CHE RISOLVE
% /cloud_registered e' l'output di FAST-LIO, quindi i punti sono gia' nel
% frame mappa MA con la deriva incorporata nelle trasformazioni. Il sintomo
% tipico e' un corridoio piano che nella mappa "sale" di diversi metri.
%
% APPROCCIO
% 1. Si leggono le pose da /Odometry
% 2. Si riportano le nuvole nel frame body invertendo la posa (un-transform)
% 3. Si selezionano keyframe
% 4. Si cercano loop con Scan Context
% 5. Si verificano i loop con ICP, scartando i falsi positivi
% 6. Si costruisce un pose graph con vincoli sequenziali + loop
% 7. Si ottimizza e si ricostruisce la mappa con le pose corrette
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

% Downsampling applicato a ogni keyframe, serve sia a Scan Context che a ICP
kfVoxel    = 0.20;   % m

% Riallineamento assetto sul pavimento (vedi sezione 6b).
% Correzione della deriva di roll/pitch usando il pavimento come riferimento
% di gravita'. Da disattivare solo se l'ambiente NON ha un pavimento piano
% (esterni, terreno irregolare, rampe continue).
useGravityAlign = true;
floorBand    = 0.30;   % m, spessore della fascia bassa in cui cercare il pavimento
floorTol     = 0.06;   % m, tolleranza di planarita' del fit
floorMaxTilt = 30;     % gradi, inclinazione max accettata per il piano trovato

% Correzione della deriva di YAW sui muri (sezione 6c).
% Il pavimento vincola roll e pitch ma NON lo yaw: i corridoi restano non
% perpendicolari in pianta. I muri sono il riferimento per il terzo DOF.
% MISURATO su questo dataset: azimut dei muri 89.45 deg nella prima meta'
% contro 82.22 nella seconda (scalino di ~7 deg allo stesso keyframe dove si
% e' rotto roll/pitch). Dopo la correzione le due meta' coincidono e la
% dispersione scende da 0.082 a 0.014.
%
% ATTENZIONE: a differenza del pavimento, questa correzione ASSUME che
% l'edificio sia ortogonale. Controllare sempre la dispersione stampata: se
% non cala nettamente, l'ipotesi non regge e va disattivata.
useYawAlign = true;
wallMaxNz   = 0.2;   % |nz| sotto cui una normale e' considerata di muro
yawRefKF    = 40;    % keyframe iniziali usati come riferimento di azimut
yawSmooth   = 9;     % finestra di lisciatura della stima, in keyframe

% Snap ortogonale della traiettoria (sezione 11), post-processing finale.
%
% NATURA DI QUESTO PASSO, da leggere prima di usarlo: NON e' una stima ne'
% un vincolo pesato. E' una FORZATURA rigida, applicata sempre e senza
% eccezioni: ogni tratto rettilineo viene ruotato finche' la sua direzione
% non e' un multiplo esatto di 90 gradi rispetto al primo tratto. Non
% guarda i muri, non guarda l'ICP, non guarda nessun altro segnale locale:
% impone e basta.
%
% SCELTA CONSAPEVOLE, NON UN BUG: se una svolta reale del percorso fisico
% non fosse esattamente a 90 gradi, questo passo la forza comunque al
% multiplo piu' vicino, INTRODUCENDO un errore rispetto alla realta'. Il
% trade-off e' stato scelto deliberatamente: in questo edificio i corridoi
% sono noti come ortogonali, e si preferisce una traiettoria pulita a
% costo di eventuali svolte non esattamente rette. Chi riusa questo script
% in un edificio non ortogonale (ali a 45 gradi, corridoi curvi) deve
% mettere useOrthogonalSnap = false, altrimenti la geometria esce
% falsificata in modo silenzioso e plausibile.
%
% Le pose originali NON vengono sovrascritte (nodesOpt/posesOpt restano
% intatti): il risultato va in nodesOrtho/posesOrtho, cosi' il confronto
% prima/dopo del pose graph resta leggibile e questo passo si vede a parte.
%
% MISURATO su questo dataset: la segmentazione grezza si frantuma, perche'
% 13 passi su 136 sono sotto 10 cm (keyframe presi per ROTAZIONE, vedi
% kfAngle, non per spostamento) e li' l'heading e' puro rumore (fino a 180
% deg di variazione tra passi consecutivi): senza filtri escono 29 "tratti",
% molti da 2-7 cm, ognuno con il proprio snap che poi si propaga a valle.
% Con i due filtri sotto ne escono 7, tutti fra 10 e 26 m, coerenti con la
% struttura fisica del percorso.
useOrthogonalSnap = true;
orthoMinStep    = 0.15;  % m, passi piu' corti non VOTANO sulla direzione del tratto
                          % (restano nella traiettoria e contribuiscono alla posizione:
                          % sono keyframe di sola rotazione, il loro heading e' rumore)
orthoSegMaxTurn = 25;     % gradi, scarto dalla direzione media corrente oltre cui inizia
                          % un nuovo tratto (una svolta vera e' ~90 deg, il rumore entro
                          % un rettilineo resta molto sotto: 25 separa bene i due casi)
orthoMinSegLen  = 3.0;    % m, tratti piu' corti vengono fusi nel precedente invece di
                          % ricevere uno snap proprio: sotto questa lunghezza la direzione
                          % stimata non e' affidabile e l'errore si propagherebbe a valle
orthoWarnCorr   = 10;     % gradi, sopra questa correzione il tratto viene segnalato: e'
                          % piu' della deriva globale gia' corretta dalla 6c (~7 deg), quindi
                          % il tratto potrebbe non essere davvero a 90 gradi nella realta'
                          % e va verificato a mano

% Correzione di consenso sui muri (sezione 11b), dopo lo snap ortogonale.
% Le pareti fisicamente uguali, viste da keyframe diversi, possono ancora
% cadere a coordinate X o Y leggermente diverse anche dopo lo snap
% ortogonale (che sistema SOLO l'orientazione, non la posizione): e' il
% residuo di deriva odometrica. Qui si cerca il "consenso" tra le
% osservazioni della stessa parete da piu' keyframe e si sposta ogni
% keyframe (sola traslazione X/Y) verso quel consenso.
%
% ATTENZIONE, stesso principio delle altre correzioni "ortogonali" di
% questo script: ASSUME che le pareti appartengano a famiglie discrete
% allineate a X/Y (garantito solo se useOrthogonalSnap ha gia' girato). Se
% in un punto reale la parete non e' dritta (rientranza, porta aperta,
% pilastro), viene comunque forzata nella famiglia piu' vicina.
%
% VERDETTO onesto, MISURATO su questo dataset -- non e' un miglioramento
% netto come lo snap ortogonale, e va letto cosi':
%   - Il criterio "media pesata sui punti tra famiglie multiple sullo
%     stesso asse" (deciso con l'utente) funziona SOLO scartando prima le
%     famiglie lontane: senza wallMaxDist, il 79% dei keyframe (asse X)
%     toccava famiglie fino a 41m di distanza (es. il muro di fondo di un
%     corridoio lungo, visto dall'inizio), con correzioni in disaccordo di
%     decine di cm sullo STESSO keyframe. Col filtro il conflitto residuo
%     tra pareti vicine (es. i due lati dello stesso corridoio) resta ma
%     e' molto piu' piccolo.
%   - La correzione GREZZA per keyframe salta fino a oltre 1m tra keyframe
%     adiacenti (~1m di distanza sul percorso): applicata cosi' com'e',
%     PEGGIORA la nitidezza locale della mappa invece di migliorarla
%     (misurato: fascia campione x in [5,15]m, std laterale del muro
%     0.220 -> 0.238m). La lisciatura sotto (stesso principio di yawSmooth
%     in 6c) elimina i salti (max 1.14 -> 0.22m) ma riduce il guadagno
%     reale a marginale: sulla mappa ricostruita (non sul solo proxy
%     analitico) lo std laterale delle famiglie cala di pochi punti
%     percentuali su alcune pareti, resta piatto o peggiora leggermente su
%     altre. Il rumore intra-keyframe (spessore fisico della singola
%     scansione, ~0.25-0.33m di std in questo dataset) domina lo spessore
%     osservato molto piu' del disallineamento tra le mediane dei
%     keyframe, e questa correzione tocca solo il secondo.
%   - In breve: utile ma non risolutivo. Non aspettarsi lo stesso salto di
%     qualita' dello snap ortogonale di sezione 11.
useWallSnap       = true;
wallLocalGapMerge = 0.30;   % m, gap tra offset per considerare due punti-parete
                             % nello stesso keyframe la stessa parete locale
wallLocalMinPts   = 30;     % punti minimi per un cluster-parete locale valido
wallMaxDist       = 10.0;   % m, oltre questa distanza dal keyframe una parete
                             % locale viene scartata: MISURATO che senza questo
                             % filtro il 79% dei keyframe (asse X) tocca famiglie
                             % fino a 41m, con correzioni in forte disaccordo
wallFamilyTol     = 0.25;   % m, gap tra offset per considerare due cluster
                             % locali (di keyframe diversi) la stessa famiglia
wallFamilyMinKF   = 4;      % keyframe minimi di supporto per tenere una famiglia
wallSmooth        = 9;      % keyframe, finestra di lisciatura della correzione
                             % grezza (stesso principio di yawSmooth in 6c):
                             % senza, i salti tra keyframe adiacenti peggiorano
                             % la mappa invece di correggerla, vedi sopra
wallWarnCorr      = 0.5;    % m, sopra questa correzione il keyframe viene
                             % segnalato: possibile errore di clustering o
                             % geometria non standard in quel punto

% Vincolo di gravita' DENTRO il pose graph (sezione 9). Senza, l'ottimizzazione
% disfa in parte il riallineamento della 6b per soddisfare i loop.
% MISURATO su questo dataset (devZ / tilt mediano / spostamento medio nodi):
%   6b senza ottimizzare : 1.84 m  0.74 deg    -
%   loop senza gravita   : 2.21 m  1.37 deg  0.21 m   <- l'ottimizzatore re-inclina
%   loop + gravita  50   : 1.82 m  0.66 deg  0.31 m
%   loop + gravita 5000  : 1.79 m  0.69 deg  2.19 m   <- non ne vale la pena
%
% Il peso NON va alzato a piacere: il guadagno su deriva e tilt satura
% attorno a 50, mentre lo spostamento dei nodi cresce senza limite. Passare
% da 50 a 5000 compra 3 cm di planarita' verticale e costa 1.9 m di
% spostamento in XY, che non abbiamo modo di validare.
useGravityFactor = true;
infoGravRP   = 50;     % peso su roll/pitch (confrontabile con infoOdom = 100)
infoGravFree = 1e-6;   % peso sui DOF liberi (x,y,z,yaw): quasi nullo ma > 0

% Applicazione dei vincoli di loop nel pose graph.
% MISURATO su questo dataset (rosbag2_2026_07_30-17_50_45): i 13 loop validi
% concordano con l'odometria entro 0.15 m e 3.4 deg, quindi portano poca
% informazione correttiva. L'effetto e' marginale e le due metriche non
% concordano:
%     solo riallineamento : deriva Z pose 1.84 m, span Z mappa 9.21 m
%     + loop closure      : deriva Z pose 2.21 m, span Z mappa 8.87 m
% Qui la deriva vera era di assetto (vedi 6b), non accumulo su ri-passaggi:
% e' il riallineamento a fare il lavoro, non i loop. Su un dataset con veri
% ri-passaggi e deriva accumulata i loop contano molto di piu'.
useLoopClosure = true;

% Rilevamento loop
scDistThreshold  = 0.15;   % soglia distanza Scan Context, piu' basso = piu' selettivo
scNumExcluded    = 30;     % keyframe recenti esclusi dalla ricerca
scMaxDetections  = 3;      % candidati per keyframe

% Ricerca loop per prossimita' spaziale (complementare a Scan Context)
proxRadius       = 3.0;    % m, raggio XY entro cui due keyframe sono "stesso luogo"
proxMinGap       = 40;     % keyframe minimi di distacco per parlare di ri-passaggio
proxMaxCand      = 300;    % tetto ai candidati di prossimita', i piu' vicini

% Verifica ICP dei loop candidati
icpMaxRMSE       = 0.30;   % m, sopra questa soglia il loop viene scartato
icpMaxDistance   = 1.0;    % m, distanza massima tra corrispondenze

% Filtro di coerenza con l'odometria.
% Un RMSE basso NON basta a fidarsi di un loop: in corridoi con strutture
% ripetitive (porte, colonne a passo costante) l'ICP puo' agganciarsi al
% "repeat" sbagliato e convergere con RMSE ottimo ma trasformazione del
% tutto errata. Il controllo giusto e' il confronto 6-DOF con la stima
% odometrica: la loop closure serve a correggere una deriva di decine di
% centimetri, non a ribaltare la traiettoria di metri. Un vincolo che
% contraddice l'odometria oltre queste soglie e' un falso positivo, non una
% correzione.
%
% NB: filtrare la sola componente Z non basta. Un match sbagliato puo'
% avere Z plausibile e comunque essere errato di metri in XY e di gradi in
% rotazione, distorcendo l'intero grafo una volta propagato.
loopMaxTransErr = 2.0;    % m, scarto max in traslazione rispetto all'odometria
loopMaxRotErr   = 10.0;   % gradi, scarto max in rotazione

% Peso relativo dei vincoli nel pose graph.
% ATTENZIONE: il peso che conta e' quello TOTALE, non quello per singolo
% vincolo. Con N keyframe ci sono N-1 vincoli odometrici ma tipicamente
% solo una manciata di loop: se infoLoop non compensa questo squilibrio
% numerico, l'ottimizzatore ignora di fatto i loop anche se sono corretti,
% e la soluzione resta quella odometrica di partenza (nessuna correzione
% visibile). Qui il peso di ogni loop viene scalato per il suo RMSE ICP
% reale: un match piu' preciso di sigma0 pesa di piu', uno vicino alla
% soglia pesa di meno.
infoOdom = 100;
infoLoop = 500;    % peso di riferimento per un loop con rmse = sigma0
sigma0Loop = kfVoxel;   % m, incertezza di riferimento (== risoluzione voxel)

% Voxel per la mappa finale
mapVoxel = 0.05;    % m

% Crop geometrico (ROI) della mappa finale, applicato nella sezione 12b.
% Coordinate nel frame mappa; usare Inf/-Inf per lasciare un asse libero.
% Lo script stampa l'estensione della mappa prima di applicarlo, cosi' da
% poter scegliere i limiti al primo giro con useMapROI = false.
% Esempi: [-Inf Inf, -Inf Inf, -Inf 2.5] taglia solo sopra i 2.5 m
%         [0 40, -5 15, -Inf Inf]        isola un tratto in pianta
useMapROI = true;
mapROI = [-Inf 42, ...     % X min max
          -Inf Inf, ...     % Y min max
          -1 4];        % Z min max

% Rilevamento divergenza odometria (IMU/FAST-LIO che perde il tracking).
% Il sintomo e' un salto di posizione tra due messaggi consecutivi
% fisicamente impossibile per la velocita' del sensore: da quel punto in
% poi le pose non sono piu' fisiche e vanno scartate, non corrette.
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

%% 3b. Troncamento alla prima divergenza dell'odometria
% Un salto di posizione troppo grande in troppo poco tempo non e' deriva:
% e' l'IMU che ha perso il tracking (superficie riflettente, urto, tratto
% senza feature). Da quel messaggio in poi tutte le pose sono inattendibili
% e vengono scartate, non "corrette" con la loop closure.
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

%% 6. Estrazione nuvole nel frame body
% /cloud_registered e' nel frame mappa: si inverte la posa per tornare al
% frame sensore, che e' quello che serve sia a Scan Context (descrittore
% sensor-centric) sia a ICP.
kfClouds = cell(nKF, 1);
kfPoses  = repmat(rigidtform3d, nKF, 1);

fprintf('Estrazione nuvole keyframe...\n');
for k = 1:nKF
    ci = pairCloudIdx(kfSel(k));
    oi = pairOdomIdx(kfSel(k));

    xyz = rosReadXYZ(cloudMsgs{ci});
    xyz = xyz(all(isfinite(xyz), 2), :);
    pcMap = pointCloud(xyz);

    % un-transform: dal frame mappa al frame body
    pcBody = pctransform(pcMap, invert(posesRaw(oi)));

    kfClouds{k} = pcdownsample(pcBody, 'gridAverage', kfVoxel);
    kfPoses(k)  = posesRaw(oi);
end

%% 6b. Riallineamento dell'assetto sul piano del pavimento
% FAST-LIO e' allineato a gravita' (l'IMU la osserva), quindi la normale del
% pavimento, riportata nel frame mappa, deve restare verticale lungo tutto
% il percorso. Quando invece si inclina progressivamente, quella e' deriva
% di assetto: la mappa "ruota" e un corridoio piano sembra scendere.
%
% Questa deriva la loop closure NON la vede: se i loop cadono tutti dentro
% la stessa meta' del percorso (prima o dopo l'evento di deriva), ogni meta'
% resta coerente con se stessa e nessun vincolo attraversa la rottura.
% Serve un riferimento esterno, e il pavimento lo e': e' una direzione
% fisica fissa, osservata direttamente dal LiDAR.
%
% Si impone quindi roll e pitch dal pavimento (2 DOF, privi di deriva per
% costruzione) e si lasciano yaw e spostamento all'odometria, che su quelli
% e' affidabile. La traiettoria viene re-integrata con gli assetti corretti.
if useGravityAlign
    fprintf('Riallineamento assetto sul pavimento...\n');

    % La fascia di ricerca parte da un PERCENTILE basso, non da min(z): un
    % singolo punto spurio sotto il pavimento sposterebbe la fascia nel
    % vuoto e il fit fallirebbe (con min(z) falliva su 25 keyframe su 137).
    nBody = nan(nKF, 3);
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
    % congelarla all'ultimo valore noto lascia riaccumulare l'errore, e i
    % buchi possono essere lunghi (qui fino a 16 keyframe consecutivi).
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
% l'edificio abbia una direzione dominante coerente (muri paralleli o
% perpendicolari). Se l'edificio avesse davvero un'ala a 45 gradi, questa
% correzione la raddrizzerebbe a torto, falsificando la geometria.
% Verificare sempre le due stampe di controllo: se la dispersione NON cala
% nettamente, i muri non appartengono a una sola famiglia ortogonale e
% l'ipotesi non regge per questo edificio: in tal caso disattivare.
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

%% 7. Rilevamento loop con Scan Context
% NOTA: verificare le firme con "doc scanContextLoopDetector" se MATLAB
% segnala argomenti non validi. I nomi dei parametri sono cambiati tra release.
fprintf('Calcolo descrittori Scan Context...\n');

loopDetector = scanContextLoopDetector;
loopCandidates = [];   % [from to]

for k = 1:nKF
    descriptor = scanContextDescriptor(kfClouds{k});

    if k > scNumExcluded
        [loopIds, ~] = detectLoop(loopDetector, descriptor, ...
            'DistanceThreshold', scDistThreshold, ...
            'NumExcludedDescriptors', scNumExcluded, ...
            'MaxDetections', scMaxDetections);

        for m = 1:numel(loopIds)
            loopCandidates(end+1, :) = [loopIds(m), k];   %#ok<SAGROW>
        end
    end

    addDescriptor(loopDetector, k, descriptor);
end

fprintf('Loop candidati da Scan Context: %d\n', size(loopCandidates, 1));

%% 7b. Diagnostica ri-passaggi e candidati per prossimita' spaziale
% Scan Context confronta l'aspetto della scansione: in un corridoio, dove
% ogni tratto somiglia a ogni altro, e' poco affidabile. Criterio
% complementare: due keyframe lontani nel tempo ma vicini nello spazio sono
% con ogni probabilita' un ri-passaggio.
%
% Si misura la distanza nel SOLO piano XY. La deriva qui e' prevalentemente
% in Z (il corridoio "sale"): includere Z allontanerebbe artificialmente
% proprio i ri-passaggi che stiamo cercando.

kfXYZ = vertcat(kfPoses.Translation);

% Matrice delle distanze XY senza pdist2 (evita la Statistics Toolbox)
dx  = kfXYZ(:,1) - kfXYZ(:,1)';
dy  = kfXYZ(:,2) - kfXYZ(:,2)';
dXY = sqrt(dx.^2 + dy.^2);

[II, JJ] = ndgrid(1:nKF, 1:nKF);
gapIdx = abs(II - JJ);

revisit = triu(dXY <= proxRadius & gapIdx >= proxMinGap, 1);

spanXYZ = max(kfXYZ, [], 1) - min(kfXYZ, [], 1);

fprintf('\n--- Diagnostica ri-passaggi ---\n');
fprintf('Lunghezza percorso:  %.1f m\n', sum(vecnorm(diff(kfXYZ), 2, 2)));
fprintf('Estensione XY:       %.1f x %.1f m\n', spanXYZ(1), spanXYZ(2));
fprintf('Escursione Z pose:   %.2f m  (deriva sospetta)\n', spanXYZ(3));
fprintf('Coppie a >=%d keyframe di distacco e <=%.1f m in XY: %d\n', ...
    proxMinGap, proxRadius, nnz(revisit));

[ri, rj] = find(revisit);
proxCandidates = [ri, rj];

% Se i ri-passaggi sono tantissimi (percorso che si sovrappone a lungo) si
% tengono i piu' stretti, altrimenti l'ICP diventa proibitivo.
if size(proxCandidates, 1) > proxMaxCand
    dSel = dXY(sub2ind([nKF nKF], ri, rj));
    [~, ord] = sort(dSel, 'ascend');
    proxCandidates = proxCandidates(ord(1:proxMaxCand), :);
    fprintf('  (ridotti ai %d piu'' vicini)\n', proxMaxCand);
end

if isempty(proxCandidates)
    fprintf(2, ['\nATTENZIONE: il percorso non ripassa MAI su se stesso.\n' ...
        'La loop closure non puo'' funzionare, su nessun asse: non esiste\n' ...
        'nessuna osservazione che leghi due punti lontani del percorso,\n' ...
        'quindi il grafo non ha modo di "sapere" che la quota e'' sbagliata.\n' ...
        'Un grafo con soli vincoli sequenziali ha come ottimo esatto\n' ...
        'l''odometria di partenza: per questo PRIMA e DOPO coincidono.\n' ...
        'Per raddrizzare questa mappa serve un vincolo esterno (es. il piano\n' ...
        'del pavimento osservato dal LiDAR), non una loop closure.\n']);
end

% Unione dei due insiemi di candidati
if isempty(loopCandidates)
    allCandidates = proxCandidates;
elseif isempty(proxCandidates)
    allCandidates = loopCandidates;
else
    allCandidates = unique([double(loopCandidates); proxCandidates], 'rows');
end
fprintf('Candidati totali da verificare: %d\n\n', size(allCandidates, 1));

%% 8. Verifica dei loop con ICP
% Scan Context produce falsi positivi, tipicamente in ambienti ripetitivi
% come i corridoi. ICP li scarta: se le due scansioni non si allineano
% davvero, l'RMSE resta alto.
loopConstraints = {};   % {fromIdx, toIdx, rigidtform3d relativa, rmse}

fprintf('Verifica ICP dei candidati...\n');
for c = 1:size(allCandidates, 1)
    i = allCandidates(c, 1);
    j = allCandidates(c, 2);

    % Stima iniziale dall'odometria: posa relativa da i a j.
    Arel = kfPoses(i).A \ kfPoses(j).A;

    % Su un ri-passaggio vero questa stima e' inquinata proprio dalla deriva
    % che vogliamo correggere: se la quota e' sbagliata di metri, ICP parte
    % fuori dal bacino di convergenza e fallisce sempre. Si prova quindi
    % anche una variante con la componente Z azzerata, e si tiene la
    % migliore delle due.
    ArelFlat = Arel;
    ArelFlat(3, 4) = 0;

    initGuesses = {Arel, ArelFlat};
    bestRmse  = inf;
    bestTform = [];

    % ICP coarse-to-fine: con una stima iniziale sbagliata di metri (la
    % deriva che vogliamo correggere), 'InlierDistance' stretto trova troppo
    % poche corrispondenze e non converge mai: fallisce anche quando le due
    % nuvole SONO in realta' sovrapponibili. Un primo passaggio con
    % corrispondenze larghe porta l'allineamento nel bacino di convergenza
    % giusto; il secondo, stretto, lo rifinisce.
    coarseInlierDistance = max(icpMaxDistance * 6, 6.0);

    for g = 1:numel(initGuesses)
        try
            [tfCoarse, ~, ~] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', rigidtform3d(initGuesses{g}), ...
                'InlierDistance', coarseInlierDistance, ...
                'MaxIterations', 50);

            [tf, ~, rmse] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', tfCoarse, ...
                'InlierDistance', icpMaxDistance);

            if rmse < bestRmse
                bestRmse  = rmse;
                bestTform = tf;
            end
        catch ME
            fprintf('  ICP fallito su %d -> %d: %s\n', i, j, ME.message);
        end
    end

    if isempty(bestTform)
        continue
    end

    % Scarto 6-DOF tra la misura ICP e la stima odometrica
    Terr     = Arel \ bestTform.A;
    transErr = norm(Terr(1:3, 4));
    rotErr   = abs(rad2deg(acos(max(-1, min(1, (trace(Terr(1:3,1:3)) - 1) / 2)))));

    if bestRmse > icpMaxRMSE
        fprintf('  loop SCARTATO  %d -> %d  (rmse %.3f m, sopra soglia)\n', ...
            i, j, bestRmse);
    elseif transErr > loopMaxTransErr || rotErr > loopMaxRotErr
        fprintf(['  loop SCARTATO  %d -> %d  (rmse %.3f m ok, ma contraddice ' ...
            'l''odometria di %.1f m / %.1f deg: falso positivo)\n'], ...
            i, j, bestRmse, transErr, rotErr);
    else
        loopConstraints{end+1} = {i, j, bestTform, bestRmse};   %#ok<SAGROW>
        fprintf('  loop accettato %d -> %d  (rmse %.3f m, scarto %.2f m / %.1f deg)\n', ...
            i, j, bestRmse, transErr, rotErr);
    end
end

nLoops = numel(loopConstraints);
fprintf('Loop verificati e accettati: %d\n', nLoops);

if nLoops == 0
    if isempty(proxCandidates)
        warning(['Nessun loop accettato E nessun ri-passaggio geometrico: ' ...
            'il percorso non torna mai sui propri passi. La mappa NON e'' ' ...
            'correggibile con la loop closure. Vedi la diagnostica sopra.']);
    else
        warning(['Nessun loop accettato, ma %d ri-passaggi geometrici ' ...
            'esistono: e'' l''ICP a rifiutarli. Le soglie sono troppo ' ...
            'strette oppure la stima iniziale e'' gia'' troppo sbagliata ' ...
            '(con %.1f m di deriva in Z, Tinit dall''odometria puo'' essere ' ...
            'fuori dal bacino di convergenza). Provare ad alzare ' ...
            'icpMaxRMSE e icpMaxDistance.'], ...
            size(proxCandidates, 1), spanXYZ(3));
    end
end

% Checkpoint: quanto sopra (lettura bag, keyframe, ICP) e' costoso e non
% dipende dai pesi del pose graph. Si salva qui per poter iterare sui pesi
% senza rifare tutto da capo.
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
save(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops', 'nKF', ...
    'kfVoxel', 'mapVoxel', 'kfClouds', 'pairCloudIdx', 'pairOdomIdx', 'kfSel', '-v7.3');
fprintf('Checkpoint salvato in: %s\n', checkpointFile);

%% 8b. Esportazione dei candidati per la verifica Stage 2 (Python, ORB/RANSAC)
% Esporta i loop accettati in sezione 8 nello stesso formato CSV/TUM usato
% dalla pipeline C++ FAST-LIO-SAM-SC-QN (vedi
% FAST-LIO-SAM-SC-QN/fast_lio_sam_sc_qn/loop_closure_analysis/plot_loop_closures.py),
% cosi' verify_loops_appearance.py e label_loops_manual.py, gia' esistenti e
% non modificati da questo script, possono girare anche sui candidati di
% QUESTA pipeline MATLAB senza alcuna modifica lato Python.
%
% Indici: loopConstraints{c}{1} e {2} sono indici 1-based in kfPoses (loop
% MATLAB 'for k = 1:nKF' in sezione 7). poses_tum.txt della pipeline C++
% usa invece l'ordine di riga come idx keyframe 0-based (riga 1 di dati =
% idx 0, nessun offset oltre l'eventuale riga di commento '#'). Si
% sottrae quindi 1 in scrittura: sbagliare questo passaggio disallinea
% silenziosamente ogni lookup dell'immagine ZED a valle, senza errore
% visibile.
fprintf('\nEsportazione candidati Stage 2 (formato CSV/TUM)...\n');

% loop_attempts_matlab.csv: solo i loop accettati (status sempre
% 'accepted', coerente con l'input Stage 2 della pipeline C++, che e'
% accepted-only; nessuno status 'rejected' qui, fuori scopo).
loopCsvFile = fullfile(fileparts(bagPath), 'loop_attempts_matlab.csv');
fidCsv = fopen(loopCsvFile, 'w');
fprintf(fidCsv, 'status,src_idx,dst_idx,score\n');
for c = 1:nLoops
    i    = loopConstraints{c}{1};
    j    = loopConstraints{c}{2};
    rmse = loopConstraints{c}{4};
    fprintf(fidCsv, 'accepted,%d,%d,%.6g\n', i - 1, j - 1, rmse);
end
fclose(fidCsv);

% poses_tum_matlab.txt: una riga per keyframe, ordine = idx 0-based.
% Timestamp per-keyframe: stessa associazione gia' usata in sezione 6 per
% costruire kfPoses(k) da posesRaw(oi); pairOdomIdx, kfSel e tOdom sono
% ancora nel workspace, invariati da nessuna sezione successiva, quindi
% non e' una ricostruzione approssimata ma lo stesso identico lookup.
posesTumFile = fullfile(fileparts(bagPath), 'poses_tum_matlab.txt');
fidTum = fopen(posesTumFile, 'w');
fprintf(fidTum, '#timestamp x y z qx qy qz qw\n');
for k = 1:nKF
    oi = pairOdomIdx(kfSel(k));
    ts = tOdom(oi);
    t  = kfPoses(k).Translation;
    q  = rotm2quat(kfPoses(k).R);   % MATLAB: [qw qx qy qz]
    fprintf(fidTum, '%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n', ...
        ts, t(1), t(2), t(3), q(2), q(3), q(4), q(1));   % TUM: x y z qx qy qz qw
end
fclose(fidTum);

fprintf('  loop esportati:     %d  -> %s\n', nLoops, loopCsvFile);
fprintf('  keyframe esportati: %d  -> %s\n', nKF, posesTumFile);

%% 9. Costruzione del pose graph
pg = poseGraph3D;

% Information matrix: 21 elementi, triangolo superiore di una 6x6.
% Diagonale = [x y z rx ry rz], valori piu' alti = vincolo piu' rigido.
infoVecOdom = buildInfoVector(infoOdom);

% Vincoli sequenziali dall'odometria
for k = 2:nKF
    Trel = kfPoses(k-1).A \ kfPoses(k).A;
    addRelativePose(pg, tform2measurement(Trel), infoVecOdom, k-1, k);
end

% Vincoli di loop: peso individuale scalato sull'RMSE ICP reale, non
% fisso. Un loop con rmse << sigma0 e' una misura molto affidabile e deve
% pesare piu' della singola pletora di vincoli odometrici; un loop vicino
% alla soglia di accettazione pesa meno.
if ~useLoopClosure
    fprintf(['Loop closure DISATTIVATA (useLoopClosure = false): il grafo usa\n' ...
        'i soli vincoli sequenziali sulle pose gia'' riallineate a gravita''.\n']);
    nLoops = 0;
end

fprintf('Pesi dei vincoli di loop (infoOdom = %d per confronto):\n', infoOdom);
for c = 1:nLoops
    i     = loopConstraints{c}{1};
    j     = loopConstraints{c}{2};
    tform = loopConstraints{c}{3};
    rmse  = loopConstraints{c}{4};

    infoLoopEff = infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
    fprintf('  %d -> %d : rmse %.3f m -> info %.0f\n', i, j, rmse, infoLoopEff);

    infoVecLoopC = buildInfoVector(infoLoopEff);
    addRelativePose(pg, tform2measurement(tform.A), infoVecLoopC, i, j);
end

% Vincoli di gravita'.
% Senza questi, il riallineamento della sezione 6b viene in parte DISFATTO
% dall'ottimizzazione: il grafo non contiene nulla che dica "il pavimento e'
% orizzontale", quindi l'ottimizzatore e' libero di re-inclinare l'assetto
% pur di soddisfare i loop.
%
% poseGraph3D non ha fattori unari (prior), ma il nodo 1 e' fissato
% dall'ottimizzatore: un vincolo 1->k con misura pari alla posa assoluta
% desiderata di k si comporta come un prior su k. Per vincolare SOLO roll e
% pitch, lasciando liberi x, y, z e yaw, si usa una information matrix
% anisotropa: peso alto su rx,ry e peso quasi nullo (ma positivo, deve
% restare definita positiva) sugli altri DOF.
if useGravityAlign && useGravityFactor
    T0inv = kfPoses(1).A \ eye(4);
    for k = 2:nKF
        Ak = T0inv * kfPoses(k).A;    % posa di k relativa al nodo 1, gia' livellata
        addRelativePose(pg, tform2measurement(Ak), ...
            buildInfoVectorAniso(infoGravFree, infoGravRP), 1, k);
    end
    fprintf('Vincoli di gravita aggiunti: %d (peso roll/pitch %g)\n', nKF-1, infoGravRP);
end

fprintf('\nPose graph: %d nodi, %d vincoli (%d loop)\n', ...
    pg.NumNodes, pg.NumEdges, nLoops);

%% 10. Ottimizzazione
fprintf('Ottimizzazione...\n');
pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');

% IMPORTANTE: poseGraph3D ancora SEMPRE il nodo 1 all'origine con
% orientamento identita', mentre kfPoses(1) ha posizione e assetto propri.
% Senza riportare il risultato nel frame di partenza, "prima" e "dopo"
% vivono in due frame globali diversi, ruotati tra loro: il confronto degli
% span in Z diventa privo di senso (una rotazione globale cambia lo span
% anche a traiettoria identica) e la mappa esce ruotata rispetto
% all'originale.
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

%% 11. Snap ortogonale della traiettoria (post-processing finale)
% Vedi il commento esteso in sezione 1 accanto a useOrthogonalSnap. In
% breve: questo passo FORZA ogni tratto rettilineo a una direzione che e'
% un multiplo esatto di 90 gradi rispetto al primo tratto. Non e' una
% stima e non e' un vincolo pesato che negozia con altre misure: e' una
% imposizione rigida, applicata sempre, senza eccezioni e senza guardare
% ne' i muri ne' l'ICP. Se una svolta reale non fosse a 90 gradi, viene
% forzata lo stesso: e' il trade-off scelto deliberatamente (corridoi noti
% come ortogonali, si preferisce una traiettoria pulita).
%
% nodesOpt/posesOpt NON vengono toccati: il risultato va in
% nodesOrtho/posesOrtho, cosi' il confronto prima/dopo del pose graph
% resta intatto e l'effetto di questo passo si vede separatamente.
nodesOrtho = [];
posesOrtho = repmat(rigidtform3d, 0, 1);

if useOrthogonalSnap
    fprintf('\nSnap ortogonale della traiettoria...\n');

    pOrthoOld = nodesOpt(:, 1:3);
    dStepXY   = diff(pOrthoOld(:, 1:2), 1, 1);
    stepLenXY = vecnorm(dStepXY, 2, 2);
    hStepXY   = atan2(dStepXY(:, 2), dStepXY(:, 1));
    nStepXY   = numel(hStepXY);

    % I passi troppo corti sono keyframe presi per ROTAZIONE (kfAngle), non
    % per spostamento: la loro direzione e' rumore e non deve decidere
    % l'orientamento di un tratto. Restano comunque nella traiettoria e
    % contribuiscono alla posizione, semplicemente non votano.
    reliableStep = stepLenXY >= orthoMinStep;
    fprintf('  passi usati per stimare la direzione (>= %.2f m): %d su %d\n', ...
        orthoMinStep, nnz(reliableStep), nStepXY);

    if nnz(reliableStep) < 2
        warning(['Troppi pochi passi affidabili per segmentare la traiettoria: ' ...
            'snap ortogonale saltato. Abbassare orthoMinStep o verificare ' ...
            'che la traiettoria non sia degenere.']);
    else
        % --- Segmentazione in tratti rettilinei ---
        % Un nuovo tratto inizia quando la direzione del passo si discosta
        % oltre orthoSegMaxTurn dalla media circolare del tratto corrente.
        segId  = zeros(nStepXY, 1);
        segCur = 0;
        segAcc = 0;
        for k = 1:nStepXY
            if ~reliableStep(k), continue; end
            if segCur == 0
                segCur   = 1;
                segAcc   = stepLenXY(k) * exp(1i*hStepXY(k));
                segId(k) = segCur;
                continue
            end
            dev = abs(rad2deg(angle(exp(1i*(hStepXY(k) - angle(segAcc))))));
            if dev > orthoSegMaxTurn
                segCur = segCur + 1;
                segAcc = stepLenXY(k) * exp(1i*hStepXY(k));
            else
                segAcc = segAcc + stepLenXY(k) * exp(1i*hStepXY(k));
            end
            segId(k) = segCur;
        end

        % I passi non affidabili ereditano il tratto del passo affidabile
        % precedente: seguono la traiettoria senza averne deciso la direzione.
        lastGoodSeg = 1;
        for k = 1:nStepXY
            if segId(k) > 0
                lastGoodSeg = segId(k);
            else
                segId(k) = lastGoodSeg;
            end
        end

        % --- Fusione dei tratti troppo corti ---
        % Sotto orthoMinSegLen la direzione stimata non e' affidabile, e uno
        % snap sbagliato li' non resta locale: la ricongiunzione lo propaga
        % a TUTTA la traiettoria a valle.
        nMergedSeg = 0;
        merging    = true;
        while merging
            merging = false;
            nSeg = max(segId);
            for s = 1:nSeg
                if sum(stepLenXY(segId == s)) >= orthoMinSegLen || nSeg <= 1
                    continue
                end
                if s > 1
                    segId(segId == s) = s - 1;
                else
                    segId(segId == s) = 2;
                end
                u = unique(segId);
                remap = containers.Map(num2cell(u), num2cell(1:numel(u)));
                for k = 1:nStepXY
                    segId(k) = remap(segId(k));
                end
                nMergedSeg = nMergedSeg + 1;
                merging    = true;
                break
            end
        end
        nSeg = max(segId);
        fprintf('  tratti rettilinei trovati: %d (fusi %d tratti sotto %.1f m)\n', ...
            nSeg, nMergedSeg, orthoMinSegLen);

        % --- Heading medio per tratto, arrotondato a multipli di 90 ---
        % Media circolare pesata sulla lunghezza del passo (stesso principio
        % della 6c, dove la media circolare serviva per l'azimut dei muri):
        % un passo lungo ha direzione molto meno rumorosa di uno corto.
        % VERIFICATO su questo dataset che usare invece la direzione della
        % corda inizio->fine del tratto da lo stesso risultato entro 0.55
        % deg, quindi la scelta tra le due non e' critica qui.
        hSeg = zeros(nSeg, 1);
        for s = 1:nSeg
            m = (segId == s) & reliableStep;
            if ~any(m), m = (segId == s); end
            hSeg(s) = angle(sum(stepLenXY(m) .* exp(1i*hStepXY(m))));
        end

        % Riferimento: il PRIMO tratto. Ancora l'orientamento globale gia'
        % stabilito dalla 6c invece di introdurre un'origine arbitraria.
        hRefOrtho = hSeg(1);
        dSeg      = zeros(nSeg, 1);
        nWarnSeg  = 0;

        fprintf('  riferimento (tratto 1): %.2f deg\n', rad2deg(hRefOrtho));
        fprintf('  tratto    kf        lungh.    heading  ->     snap   correzione\n');
        for s = 1:nSeg
            relDeg  = rad2deg(angle(exp(1i*(hSeg(s) - hRefOrtho))));
            snapDeg = round(relDeg / 90) * 90;
            corrDeg = snapDeg - relDeg;
            dSeg(s) = deg2rad(corrDeg);

            stepsOfSeg = find(segId == s);
            segLenM    = sum(stepLenXY(stepsOfSeg));
            flagStr    = '';
            if abs(corrDeg) > orthoWarnCorr
                flagStr  = '   <-- VERIFICARE A MANO';
                nWarnSeg = nWarnSeg + 1;
            end
            fprintf('    %2d    %3d-%3d   %7.2f m  %8.2f  -> %8.2f  %+7.2f deg%s\n', ...
                s, stepsOfSeg(1), stepsOfSeg(end) + 1, segLenM, ...
                rad2deg(hSeg(s)), rad2deg(hRefOrtho) + snapDeg, corrDeg, flagStr);
        end

        % --- Ricongiunzione dei tratti ---
        % Stessa re-integrazione gia' usata in 6b/6c (pOld/pNew): lo
        % spostamento di ogni passo viene dalla traiettoria ottimizzata, la
        % direzione in cui applicarlo dalla rotazione del tratto a cui il
        % passo appartiene. I tratti restano attaccati per costruzione (il
        % punto finale di un tratto ruotato E' il punto iniziale del
        % successivo), senza bookkeeping esplicito degli estremi.
        pOrthoNew = zeros(nKF, 3);
        pOrthoNew(1, :) = pOrthoOld(1, :);
        RorthoAll = cell(nKF, 1);

        th = dSeg(segId(1));
        Cz = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
        RorthoAll{1} = Cz * quat2rotm(nodesOpt(1, 4:7));

        for k = 2:nKF
            th = dSeg(segId(k-1));   % tratto a cui appartiene il passo k-1 -> k
            Cz = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
            pOrthoNew(k, :) = pOrthoNew(k-1, :) + ...
                (Cz * (pOrthoOld(k, :) - pOrthoOld(k-1, :))')';
            % la stessa rotazione di yaw va applicata anche all'ASSETTO, non
            % solo alla posizione, altrimenti le nuvole restano ruotate
            % rispetto alla traiettoria e la mappa esce peggio di prima
            RorthoAll{k} = Cz * quat2rotm(nodesOpt(k, 4:7));
        end

        posesOrtho = repmat(rigidtform3d, nKF, 1);
        for k = 1:nKF
            posesOrtho(k) = rigidtform3d(RorthoAll{k}, pOrthoNew(k, :));
        end
        nodesOrtho = [vertcat(posesOrtho.Translation), ...
                      cell2mat(arrayfun(@(p) rotm2quat(p.R), posesOrtho, 'UniformOutput', false))];

        % --- Verifica: dopo lo snap ogni tratto deve essere esattamente a
        % un multiplo di 90 gradi dal riferimento. Se questo residuo non e'
        % ~0 la ricongiunzione non ha fatto quello che doveva.
        dSnapCheck = diff(pOrthoNew(:, 1:2), 1, 1);
        hSnapCheck = atan2(dSnapCheck(:, 2), dSnapCheck(:, 1));
        resMax = 0;
        for s = 1:nSeg
            m = (segId == s) & reliableStep;
            if ~any(m), continue; end
            hNew   = angle(sum(stepLenXY(m) .* exp(1i*hSnapCheck(m))));
            relNew = rad2deg(angle(exp(1i*(hNew - hRefOrtho))));
            resMax = max(resMax, abs(relNew - round(relNew/90)*90));
        end
        fprintf('  residuo max dal multiplo di 90 dopo lo snap: %.4f deg (atteso ~0)\n', resMax);

        dNodeOrtho = vecnorm(nodesOrtho(:, 1:3) - nodesOpt(:, 1:3), 2, 2);
        fprintf('  spostamento nodi rispetto al pose graph: media %.2f m, max %.2f m\n', ...
            mean(dNodeOrtho), max(dNodeOrtho));

        if nWarnSeg > 0
            fprintf(2, ['  ATTENZIONE: %d tratto/i con correzione oltre %.0f deg.\n' ...
                '  E'' piu'' della deriva globale gia'' corretta dalla 6c: quei tratti\n' ...
                '  potrebbero NON essere davvero a 90 gradi nella realta'', e lo snap\n' ...
                '  li sta forzando comunque. Verificare a mano sulla mappa prima di\n' ...
                '  fidarsi di quella zona.\n'], nWarnSeg, orthoWarnCorr);
        end
    end
end

%% 11b. Correzione di consenso sui muri (wall snap)
% Vedi il commento esteso in sezione 1 accanto a useWallSnap, in particolare
% il VERDETTO misurato su questo dataset: guadagno reale marginale, non un
% netto miglioramento come lo snap ortogonale di sezione 11.
%
% nodesOrtho/posesOrtho NON vengono toccati: il risultato va in
% nodesWallSnap/posesWallSnap.
nodesWallSnap = [];
posesWallSnap = repmat(rigidtform3d, 0, 1);

if useWallSnap
    if ~useOrthogonalSnap || isempty(nodesOrtho)
        warning(['useWallSnap attivo ma nodesOrtho non disponibile (richiede ' ...
            'useOrthogonalSnap attivo in sezione 11, con almeno un tratto ' ...
            'trovato): step saltato. Il wall snap assume muri gia'' allineati ' ...
            'a X/Y globali, precondizione che solo lo snap ortogonale garantisce.']);
    else
        fprintf('\nCorrezione di consenso sui muri...\n');

        % --- Passo 1: piani-parete locali per keyframe ---
        % Stesso pattern di 6c per normali/classificazione muro (pcnormals,
        % wallMaxNz), esteso con: classificazione famiglia X/Y in base a
        % quale componente della normale domina, poi clustering 1D per gap
        % sull'offset (posizione lungo quell'asse) per separare pareti
        % diverse viste dallo STESSO keyframe (es. i due lati di un
        % corridoio). Il filtro wallMaxDist scarta le pareti troppo
        % lontane: vedi il commento in sezione 1 sul perche' e' necessario.
        wallLocal = struct('kf', {}, 'axis', {}, 'offset', {}, 'npts', {});
        nSkippedWall = 0;
        for k = 1:nKF
            pc = kfClouds{k};
            if pc.Count < 200, nSkippedWall = nSkippedWall + 1; continue; end
            try
                nrm = pcnormals(pc, 20);
            catch
                nSkippedWall = nSkippedWall + 1; continue
            end
            R = posesOrtho(k).R;
            t = posesOrtho(k).Translation;
            nMap = (R * nrm')';
            isWall = abs(nMap(:,3)) < wallMaxNz;
            if nnz(isWall) < 50, nSkippedWall = nSkippedWall + 1; continue; end

            ptsMap = (R * pc.Location(isWall,:)')' + t;
            nW = nMap(isWall,:);
            isXfam = abs(nW(:,1)) >= abs(nW(:,2));   % normale piu' lungo X -> famiglia X

            for axLabel = ["X", "Y"]
                if axLabel == "X", m = isXfam; col = 1; else, m = ~isXfam; col = 2; end
                if nnz(m) < wallLocalMinPts, continue; end
                offs = sort(ptsMap(m, col));
                gaps = find(diff(offs) > wallLocalGapMerge);
                edges = [0; gaps; numel(offs)];
                for c = 1:numel(edges)-1
                    rng = (edges(c)+1):edges(c+1);
                    if numel(rng) < wallLocalMinPts, continue; end
                    localOffset = median(offs(rng));
                    if abs(t(col) - localOffset) > wallMaxDist, continue; end
                    wallLocal(end+1) = struct('kf', k, 'axis', char(axLabel), ...
                        'offset', localOffset, 'npts', numel(rng)); %#ok<SAGROW>
                end
            end
        end
        fprintf('  piani-parete locali trovati: %d (%d/%d keyframe senza normali/pochi punti muro)\n', ...
            numel(wallLocal), nSkippedWall, nKF);

        if numel(wallLocal) < 2 * wallFamilyMinKF
            warning(['Troppo pochi piani-parete locali per formare famiglie: ' ...
                'wall snap saltato. Ambiente probabilmente poco strutturato ' ...
                'o wallMaxDist troppo stretto.']);
        else
            % --- Passo 2-3: famiglie tra keyframe, separatamente per asse ---
            % Stesso principio della media circolare di 6c, qui su un asse
            % lineare (offset), quindi clustering per gap invece di media
            % circolare: due cluster locali entro wallFamilyTol sono la
            % stessa parete fisica, oltre sono famiglie diverse. Famiglie
            % con troppo pochi keyframe di supporto sono scartate (rumore,
            % non una vera parete condivisa).
            famAxis = strings(0, 1);
            famOffset = zeros(0, 1);
            famMembers = {};
            for axLabel = ["X", "Y"]
                idx = find(strcmp({wallLocal.axis}, axLabel));
                if isempty(idx), continue; end
                offs = [wallLocal(idx).offset];
                [offsSorted, ord] = sort(offs);
                idxSorted = idx(ord);
                famBreaks = find(diff(offsSorted) > wallFamilyTol);
                edges = [0, famBreaks, numel(offsSorted)];
                for f = 1:numel(edges)-1
                    members = idxSorted((edges(f)+1):edges(f+1));
                    kfs = unique([wallLocal(members).kf]);
                    if numel(kfs) < wallFamilyMinKF, continue; end

                    % Passo 4: consenso = mediana pesata sui punti dei
                    % cluster locali contribuenti (robusta a outlier, non
                    % una semplice media).
                    offsM = [wallLocal(members).offset];
                    wM    = [wallLocal(members).npts];
                    [vs, ordW] = sort(offsM);
                    ws = wM(ordW);
                    cw = cumsum(ws) / sum(ws);
                    wMed = vs(find(cw >= 0.5, 1, 'first'));

                    famAxis(end+1, 1)   = axLabel;
                    famOffset(end+1, 1) = wMed;
                    famMembers{end+1, 1} = members;
                end
            end
            nFamX = nnz(famAxis == "X");
            nFamY = nnz(famAxis == "Y");
            fprintf('  famiglie di pareti (>= %d keyframe di supporto): %d (X: %d, Y: %d)\n', ...
                wallFamilyMinKF, numel(famOffset), nFamX, nFamY);
            for f = 1:numel(famOffset)
                kfs = unique([wallLocal(famMembers{f}).kf]);
                npts = sum([wallLocal(famMembers{f}).npts]);
                fprintf('    fam %2d (%s): offset consenso %7.2f, keyframe %3d, punti %6d\n', ...
                    f, famAxis(f), famOffset(f), numel(kfs), npts);
            end

            % --- Passo 5: correzione grezza per keyframe ---
            % Media pesata sui punti tra le famiglie toccate sullo stesso
            % asse (deciso con l'utente dopo aver verificato sui dati reali
            % che ne' "solo la piu' supportata" ne' una media cieca SENZA
            % il filtro wallMaxDist reggono, vedi commento in sezione 1).
            touchXc = cell(nKF, 2);   % {: ,1}=correzioni, {: ,2}=pesi(npts)
            touchYc = cell(nKF, 2);
            for f = 1:numel(famOffset)
                for m = reshape(famMembers{f}, 1, [])   % forza riga: un elemento per iterazione
                    r = wallLocal(m);
                    corr = famOffset(f) - r.offset;
                    if r.axis == 'X'
                        touchXc{r.kf,1} = [touchXc{r.kf,1}, corr];
                        touchXc{r.kf,2} = [touchXc{r.kf,2}, r.npts];
                    else
                        touchYc{r.kf,1} = [touchYc{r.kf,1}, corr];
                        touchYc{r.kf,2} = [touchYc{r.kf,2}, r.npts];
                    end
                end
            end
            corrXraw = nan(nKF, 1);
            corrYraw = nan(nKF, 1);
            for k = 1:nKF
                if ~isempty(touchXc{k,1})
                    corrXraw(k) = sum(touchXc{k,1} .* touchXc{k,2}) / sum(touchXc{k,2});
                end
                if ~isempty(touchYc{k,1})
                    corrYraw(k) = sum(touchYc{k,1} .* touchYc{k,2}) / sum(touchYc{k,2});
                end
            end
            fprintf('  keyframe con correzione diretta: X %d/%d, Y %d/%d\n', ...
                nnz(~isnan(corrXraw)), nKF, nnz(~isnan(corrYraw)), nKF);

            % Lisciatura a finestra mobile (stesso principio di yawSmooth in
            % 6c) PRIMA di riempire i buchi: senza, i salti tra keyframe
            % adiacenti superano 1m e peggiorano la mappa, vedi il commento
            % in sezione 1.
            hw = floor(wallSmooth / 2);
            corrX = nan(nKF, 1);
            corrY = nan(nKF, 1);
            for k = 1:nKF
                w = corrXraw(max(1,k-hw):min(nKF,k+hw)); w = w(~isnan(w));
                if ~isempty(w), corrX(k) = mean(w); end
                w = corrYraw(max(1,k-hw):min(nKF,k+hw)); w = w(~isnan(w));
                if ~isempty(w), corrY(k) = mean(w); end
            end

            % --- Passo 6: buchi e ricongiunzione ---
            % Buchi (keyframe senza famiglia toccata su un asse): stesso
            % pattern di interpolazione lineare tra vicini validi gia' usato
            % in 6b per il pavimento (qui su uno scalare, non un
            % quaternione, quindi non serve slerpQuat).
            corrX = fillGapsLinear(corrX);
            corrY = fillGapsLinear(corrY);

            % Applicazione: sola traslazione X/Y (l'orientazione e' gia'
            % sistemata dallo snap di sezione 11). A differenza di 6b/6c/11
            % la correzione qui e' una TRASLAZIONE, non una rotazione: non
            % serve la reintegrazione via spostamenti locali ruotati (quel
            % meccanismo serve quando cambia la DIREZIONE con cui un passo
            % va sommato). Sommare direttamente pOld(k)+corr(k) e'
            % algebricamente equivalente a quella reintegrazione (somma
            % telescopica) per una correzione additiva, e la propagazione
            % ai keyframe a valle e' comunque garantita: corr(k) e'
            % liscia/interpolata (mai azzerata nei buchi), quindi un
            % keyframe corretto sposta con continuita' anche i successivi.
            pWallOld = nodesOrtho(:, 1:3);
            pWallNew = pWallOld + [corrX, corrY, zeros(nKF, 1)];

            posesWallSnap = repmat(rigidtform3d, nKF, 1);
            for k = 1:nKF
                posesWallSnap(k) = rigidtform3d(posesOrtho(k).R, pWallNew(k, :));
            end
            nodesWallSnap = [pWallNew, nodesOrtho(:, 4:7)];

            % --- Passo 8: diagnostica ---
            fprintf('  correzione X: mediana %.3f m, max %.3f m\n', ...
                median(abs(corrX)), max(abs(corrX)));
            fprintf('  correzione Y: mediana %.3f m, max %.3f m\n', ...
                median(abs(corrY)), max(abs(corrY)));
            nWarnWall = nnz(abs(corrX) > wallWarnCorr | abs(corrY) > wallWarnCorr);
            if nWarnWall > 0
                fprintf(2, ['  ATTENZIONE: %d keyframe con correzione oltre %.1fm: possibile ' ...
                    'errore di clustering (famiglie unite per sbaglio) o geometria non ' ...
                    'standard in quel punto (rientranza, porta aperta, pilastro). ' ...
                    'Verificare a mano.\n'], nWarnWall, wallWarnCorr);
            end

            % Verifica: std dell'offset per famiglia, proxy analitico veloce
            % (ricalcolo diretto senza rifare pcnormals). ATTENZIONE: questo
            % proxy da' un quadro piu' ottimistico della mappa reale, vedi
            % il VERDETTO in sezione 1 -- il rumore intra-keyframe che
            % domina lo spessore osservato sulla mappa vera non e' incluso
            % qui.
            stdBefore = [];
            stdAfter  = [];
            for f = 1:numel(famOffset)
                if numel(famMembers{f}) < 5, continue; end
                offsB = [wallLocal(famMembers{f}).offset];
                kfsM  = [wallLocal(famMembers{f}).kf];
                if famAxis(f) == "X", c = corrX(kfsM); else, c = corrY(kfsM); end
                stdBefore(end+1) = std(offsB);
                stdAfter(end+1)  = std(offsB + c');
            end
            if ~isempty(stdBefore)
                fprintf('  std offset per famiglia (proxy, NON la mappa reale): mediana %.3f -> %.3f m\n', ...
                    median(stdBefore), median(stdAfter));
            end

            dNodeWall = vecnorm(nodesWallSnap(:,1:3) - nodesOrtho(:,1:3), 2, 2);
            fprintf('  spostamento nodi rispetto allo snap ortogonale: media %.3f m, max %.3f m\n', ...
                mean(dNodeWall), max(dNodeWall));
        end
    end
end

%% 12. Ricostruzione mappa con pose corrette
allXYZ = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, posesOpt(k));
    allXYZ{k} = pcT.Location;
end

pcOpt = pointCloud(vertcat(allXYZ{:}));
pcOpt = pcdownsample(pcOpt, 'gridAverage', mapVoxel);

% Mappa originale, per confronto
allXYZraw = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, kfPoses(k));
    allXYZraw{k} = pcT.Location;
end
pcRaw = pointCloud(vertcat(allXYZraw{:}));
pcRaw = pcdownsample(pcRaw, 'gridAverage', mapVoxel);

% Mappa con le pose passate per lo snap ortogonale (sezione 11). Tenuta
% separata da pcOpt: quella resta il risultato del solo pose graph, cosi'
% il confronto prima/dopo esistente non cambia significato.
pcOrtho = pointCloud.empty;
if useOrthogonalSnap && ~isempty(nodesOrtho)
    allXYZortho = cell(nKF, 1);
    for k = 1:nKF
        pcT = pctransform(kfClouds{k}, posesOrtho(k));
        allXYZortho{k} = pcT.Location;
    end
    pcOrtho = pointCloud(vertcat(allXYZortho{:}));
    pcOrtho = pcdownsample(pcOrtho, 'gridAverage', mapVoxel);
end

% Mappa con le pose passate per la correzione di consenso sui muri
% (sezione 11b). Tenuta separata da pcOrtho: vedi il VERDETTO in sezione 1,
% il guadagno atteso qui e' marginale, non netto come lo snap ortogonale.
pcWallSnap = pointCloud.empty;
if useWallSnap && ~isempty(nodesWallSnap)
    allXYZwall = cell(nKF, 1);
    for k = 1:nKF
        pcT = pctransform(kfClouds{k}, posesWallSnap(k));
        allXYZwall{k} = pcT.Location;
    end
    pcWallSnap = pointCloud(vertcat(allXYZwall{:}));
    pcWallSnap = pcdownsample(pcWallSnap, 'gridAverage', mapVoxel);
end

%% 12b. Crop geometrico (ROI) sulla mappa finale
% ATTENZIONE al punto in cui si applica. Il ROI va QUI, sulla mappa gia'
% ricostruita, non sulle nuvole keyframe: quelle sono in frame body e
% servono a Scan Context (descrittore che vuole la scansione intera), a ICP
% (che ha bisogno di geometria per agganciarsi) e al fit del pavimento
% (sezione 6b). Ritagliarle degraderebbe registrazione e riallineamento.
%
% Le coordinate sono nel frame mappa, lo stesso della traiettoria: usare
% l'estensione stampata qui sotto per scegliere i limiti. Inf/-Inf lasciano
% l'asse libero.
fprintf('\n--- Estensione della mappa (per scegliere il ROI) ---\n');
fprintf('  X: %7.2f  %7.2f\n', pcOpt.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcOpt.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcOpt.ZLimits);

if useMapROI
    nBeforeOpt = pcOpt.Count;
    nBeforeRaw = pcRaw.Count;

    % Il crop si applica a TUTTE le mappe: confrontare una zona ritagliata
    % con la mappa intera renderebbe il confronto prima/dopo privo di senso.
    pcOpt = select(pcOpt, findPointsInROI(pcOpt, mapROI));
    pcRaw = select(pcRaw, findPointsInROI(pcRaw, mapROI));

    fprintf('ROI applicato [%g %g, %g %g, %g %g]\n', mapROI);
    fprintf('  mappa corretta: %d -> %d punti (%.1f%% rimosso)\n', ...
        nBeforeOpt, pcOpt.Count, 100*(nBeforeOpt - pcOpt.Count)/nBeforeOpt);
    fprintf('  mappa grezza:   %d -> %d punti (%.1f%% rimosso)\n', ...
        nBeforeRaw, pcRaw.Count, 100*(nBeforeRaw - pcRaw.Count)/nBeforeRaw);

    if ~isempty(pcOrtho)
        nBeforeOrtho = pcOrtho.Count;
        pcOrtho = select(pcOrtho, findPointsInROI(pcOrtho, mapROI));
        fprintf('  mappa snap ortogonale: %d -> %d punti (%.1f%% rimosso)\n', ...
            nBeforeOrtho, pcOrtho.Count, 100*(nBeforeOrtho - pcOrtho.Count)/nBeforeOrtho);
    end

    if ~isempty(pcWallSnap)
        nBeforeWall = pcWallSnap.Count;
        pcWallSnap = select(pcWallSnap, findPointsInROI(pcWallSnap, mapROI));
        fprintf('  mappa wall snap:       %d -> %d punti (%.1f%% rimosso)\n', ...
            nBeforeWall, pcWallSnap.Count, 100*(nBeforeWall - pcWallSnap.Count)/nBeforeWall);
    end

    if pcOpt.Count == 0 || pcRaw.Count == 0
        error(['Il ROI ha svuotato la mappa: nessun punto dentro i limiti. ' ...
            'Controllare che le coordinate siano nel frame mappa stampato sopra.']);
    end
end

%% 13. Confronto
% ATTENZIONE alla metrica. L'escursione Z della MAPPA non misura la deriva:
% ogni singola scansione copre gia' diversi metri in verticale (il LiDAR
% vede pavimento e soffitto, e negli atri molto piu' in alto), quindi il
% bounding box resta ampio anche con una traiettoria perfetta. La deriva
% vive nelle POSE, ed e' li' che va misurata.
zTrajRaw = vertcat(kfPoses.Translation);
zTrajRaw = zTrajRaw(:,3);
zTrajOpt = nodesOpt(:,3);

spanTrajRaw = max(zTrajRaw) - min(zTrajRaw);
spanTrajOpt = max(zTrajOpt) - min(zTrajOpt);

fprintf('\n--- Deriva verticale della TRAIETTORIA (metrica corretta) ---\n');
fprintf('Prima:  escursione Z pose %.2f m\n', spanTrajRaw);
fprintf('Dopo:   escursione Z pose %.2f m  (%+.1f%%)\n', ...
    spanTrajOpt, 100*(spanTrajOpt - spanTrajRaw)/spanTrajRaw);
fprintf('Spostamento medio dei nodi: %.3f m\n', ...
    mean(vecnorm(nodesOpt(:,1:3) - vertcat(kfPoses.Translation), 2, 2)));

% Escursione Z della mappa, riportata solo come riferimento: NON e' un
% indicatore di deriva, vedi commento sopra.
fprintf('\n--- Estensione Z della mappa (NON indicatore di deriva) ---\n');
fprintf('Prima:  %7.2f  %7.2f   (escursione %.2f m)\n', ...
    pcRaw.ZLimits, diff(pcRaw.ZLimits));
fprintf('Dopo:   %7.2f  %7.2f   (escursione %.2f m)\n', ...
    pcOpt.ZLimits, diff(pcOpt.ZLimits));

% Con gli snap attivi le mappe da confrontare arrivano a quattro: ogni
% subplot in piu' mostra la stessa mappa dopo un passo successivo, per
% vedere a occhio se i corridoi si sono raddrizzati/allineati.
showOrtho = ~isempty(pcOrtho);
showWall  = ~isempty(pcWallSnap);
nSub = 2 + double(showOrtho) + double(showWall);

figure('Color', 'k', 'Name', 'Confronto prima/dopo loop closure');

subplot(nSub,1,1);
pcshow(pcRaw, 'MarkerSize', 20);
title(sprintf('PRIMA, escursione Z %.2f m', diff(pcRaw.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

subplot(nSub,1,2);
pcshow(pcOpt, 'MarkerSize', 20);
title(sprintf('DOPO, escursione Z %.2f m', diff(pcOpt.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

nextSub = 3;
if showOrtho
    subplot(nSub,1,nextSub);
    pcshow(pcOrtho, 'MarkerSize', 20);
    title(sprintf('DOPO SNAP ORTOGONALE, escursione Z %.2f m', diff(pcOrtho.ZLimits)), ...
        'Color', 'w');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    axis equal; colormap(gca, turbo);
    nextSub = nextSub + 1;
end

if showWall
    subplot(nSub,1,nextSub);
    pcshow(pcWallSnap, 'MarkerSize', 20);
    title(sprintf('DOPO WALL SNAP, escursione Z %.2f m', diff(pcWallSnap.ZLimits)), ...
        'Color', 'w');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    axis equal; colormap(gca, turbo);
end

% Traiettoria prima e dopo
figure('Name', 'Traiettoria');
trajRaw = vertcat(kfPoses.Translation);
plot3(trajRaw(:,1), trajRaw(:,2), trajRaw(:,3), 'r-', 'LineWidth', 1.5);
hold on;
plot3(nodesOpt(:,1), nodesOpt(:,2), nodesOpt(:,3), 'g-', 'LineWidth', 1.5);
legendEntries = {'Prima', 'Dopo'};
if ~isempty(nodesOrtho)
    plot3(nodesOrtho(:,1), nodesOrtho(:,2), nodesOrtho(:,3), 'b-', 'LineWidth', 2);
    legendEntries{end+1} = 'Dopo snap ortogonale';
end
if ~isempty(nodesWallSnap)
    plot3(nodesWallSnap(:,1), nodesWallSnap(:,2), nodesWallSnap(:,3), 'm-', 'LineWidth', 1.5);
    legendEntries{end+1} = 'Dopo wall snap';
end
legend(legendEntries, 'Location', 'best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Traiettoria keyframe');
axis equal; grid on;

% Nitidezza dei muri, prima/dopo il wall snap, sulla MAPPA RICOSTRUITA
% (non il proxy analitico di sezione 11b): per ogni famiglia di pareti
% trovata li', std laterale dei punti reali entro una fascia di 0.6m dal
% consenso. E' la metrica diretta richiesta per verificare il guadagno,
% vedi il VERDETTO in sezione 1 sul perche' e' tipicamente marginale.
if showWall && exist('famAxis', 'var') && exist('famOffset', 'var')
    fprintf('\n--- Nitidezza pareti sulla mappa reale, prima/dopo wall snap ---\n');
    for f = 1:numel(famOffset)
        if numel(famMembers{f}) < 15, continue; end
        off = famOffset(f);
        if famAxis(f) == "X"
            col = 1;
        else
            col = 2;
        end
        mO = abs(pcOrtho.Location(:,col) - off) < 0.6;
        mW = abs(pcWallSnap.Location(:,col) - off) < 0.6;
        fprintf('  fam %2d (%s, offset %.2f): std ortho %.3f m (n=%d)  ->  std wall snap %.3f m (n=%d)\n', ...
            f, famAxis(f), off, std(pcOrtho.Location(mO,col)), nnz(mO), ...
            std(pcWallSnap.Location(mW,col)), nnz(mW));
    end
end

%% 14. Salvataggio
outFile = fullfile(fileparts(bagPath), 'loop_closed_map.pcd');
pcwrite(pcOpt, outFile, 'Encoding', 'binary');
fprintf('\nMappa salvata in:\n  %s\n', outFile);

% La mappa con lo snap ortogonale va in un file SEPARATO: e' il risultato
% di una forzatura geometrica (vedi sezione 11), non dello stesso tipo di
% correzione del pose graph, e va tenuta distinguibile.
if ~isempty(pcOrtho)
    outFileOrtho = fullfile(fileparts(bagPath), 'loop_closed_map_ortho.pcd');
    pcwrite(pcOrtho, outFileOrtho, 'Encoding', 'binary');
    fprintf('Mappa con snap ortogonale salvata in:\n  %s\n', outFileOrtho);
end

% La mappa con la correzione sui muri va anch'essa in un file SEPARATO,
% stesso motivo di pcOrtho: e' un passo di post-processing distinto, e va
% tenuta distinguibile (vedi il VERDETTO in sezione 1 prima di trattarla
% come "la" mappa definitiva).
if ~isempty(pcWallSnap)
    outFileWall = fullfile(fileparts(bagPath), 'loop_closed_map_wallsnap.pcd');
    pcwrite(pcWallSnap, outFileWall, 'Encoding', 'binary');
    fprintf('Mappa con wall snap salvata in:\n  %s\n', outFileWall);
end

%% Funzioni di supporto
function v = buildInfoVectorAniso(wFree, wRP)
    % Information matrix anisotropa: peso su roll e pitch, quasi nullo sul
    % resto. L'ordine della diagonale e' [x y z rx ry rz]. wFree deve essere
    % > 0: addRelativePose rifiuta matrici non definite positive.
    M = diag([wFree wFree wFree wRP wRP wFree]);
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

function c = fillGapsLinear(c)
    % Interpolazione lineare tra i due valori validi piu' vicini, stesso
    % pattern usato in 6b per il pavimento (li' su un quaternione via
    % slerpQuat, qui su uno scalare quindi basta una combinazione lineare).
    % Ai bordi, dove manca un lato, tiene costante il valore valido piu'
    % vicino invece di estrapolare.
    vi = find(~isnan(c));
    if isempty(vi), return; end
    for k = 1:numel(c)
        if ~isnan(c(k)), continue; end
        prev = vi(find(vi < k, 1, 'last'));
        next = vi(find(vi > k, 1, 'first'));
        if isempty(prev)
            c(k) = c(next);
        elseif isempty(next)
            c(k) = c(prev);
        else
            t = (k - prev) / (next - prev);
            c(k) = c(prev) + t * (c(next) - c(prev));
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