%% Prova 3: fascia pavimento robusta + interpolazione dei buchi
%
% Due difetti della prova 2:
%  1. la fascia di ricerca partiva da min(z): un singolo punto spurio sotto
%     il pavimento la spostava nel vuoto e il fit falliva (25 keyframe su
%     137, con 1-45 punti candidati invece di centinaia). Qui si usa un
%     percentile basso, insensibile agli outlier isolati.
%  2. nei buchi la correzione restava congelata all'ultimo valore noto; con
%     16 keyframe consecutivi mancanti in coda l'errore si riaccumulava.
%     Qui la correzione viene interpolata (slerp) tra i due estremi validi.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
load(fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat'), ...
    'kfPoses', 'nKF', 'kfClouds', 'mapVoxel');

floorBand = 0.35; floorTol = 0.06; floorMaxTilt = 30;

%% 1. Normale pavimento con fascia robusta
nBody = nan(nKF,3);
for k = 1:nKF
    loc = kfClouds{k}.Location;
    if size(loc,1) < 80, continue; end
    zref = prctile(loc(:,3), 2);                       % robusto agli outlier
    sel  = loc(:,3) > zref - 0.12 & loc(:,3) < zref + floorBand;
    cand = loc(sel, :);
    if size(cand,1) < 50, continue; end
    try
        [model, inl] = pcfitplane(pointCloud(cand), floorTol, [0 0 1], floorMaxTilt);
        if numel(inl) < 40, continue; end
    catch
        continue
    end
    n = model.Normal(:); if n(3) < 0, n = -n; end
    nBody(k,:) = n' / norm(n);
end
valid = ~any(isnan(nBody),2);
fprintf('Normali pavimento valide: %d su %d  (prova 2: 112)\n', nnz(valid), nKF);

%% 2. Correzione dove misurata
qC = nan(nKF,4);
for k = 1:nKF
    if ~valid(k), continue; end
    nMap = kfPoses(k).R * nBody(k,:)';
    if nMap(3) < 0, nMap = -nMap; end
    nMap = nMap/norm(nMap);
    ax = cross(nMap,[0;0;1]); s = norm(ax); c = dot(nMap,[0;0;1]);
    if s > 1e-8
        ax = ax/s; ang = atan2(s,c);
        K = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
        C = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);
    else
        C = eye(3);
    end
    qC(k,:) = rotm2quat(C);
end

%% 3. Interpolazione (slerp) sui buchi
vi = find(valid);
for k = 1:nKF
    if valid(k), continue; end
    prev = vi(find(vi < k, 1, 'last'));
    next = vi(find(vi > k, 1, 'first'));
    if isempty(prev), qC(k,:) = qC(next,:);
    elseif isempty(next), qC(k,:) = qC(prev,:);
    else
        t = (k - prev) / (next - prev);
        qC(k,:) = slerpQuat(qC(prev,:), qC(next,:), t);
    end
end

%% 4. Applicazione e re-integrazione
Rc = cell(nKF,1);
for k = 1:nKF, Rc{k} = quat2rotm(qC(k,:)) * kfPoses(k).R; end

pOld = vertcat(kfPoses.Translation);
pNew = zeros(nKF,3); pNew(1,:) = pOld(1,:);
for k = 2:nKF
    dLocal = kfPoses(k-1).R' * (pOld(k,:) - pOld(k-1,:))';
    pNew(k,:) = pNew(k-1,:) + (Rc{k-1}*dLocal)';
end

posesNew = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF, posesNew(k) = rigidtform3d(Rc{k}, pNew(k,:)); end

%% 5. Verifica
tiltA = nan(nKF,1);
for k = 1:nKF
    if ~valid(k), continue; end
    nMap = Rc{k} * nBody(k,:)'; if nMap(3)<0, nMap=-nMap; end
    tiltA(k) = rad2deg(acos(max(-1,min(1,nMap(3)))));
end

acc = cell(nKF,1);
for k = 1:nKF, acc{k} = pctransform(kfClouds{k}, posesNew(k)).Location; end
mapNew = pcdownsample(pointCloud(vertcat(acc{:})), 'gridAverage', mapVoxel);

fprintf('\n--- Risultato ---\n');
fprintf('Deriva Z pose:  %.2f m  (grezza 4.77, prova 2: 1.81)\n', ...
    max(pNew(:,3))-min(pNew(:,3)));
fprintf('Span Z mappa:   %.2f m  (grezza 14.01, prova 2: 10.04)\n', ...
    diff(mapNew.ZLimits));
fprintf('Tilt pavimento residuo: mediana %.2f deg, max %.2f deg\n', ...
    median(tiltA(~isnan(tiltA))), max(tiltA));

%% Slerp senza Aerospace Toolbox
function q = slerpQuat(q0, q1, t)
    q0 = q0/norm(q0); q1 = q1/norm(q1);
    c = dot(q0, q1);
    if c < 0, q1 = -q1; c = -c; end    % percorso piu' corto
    if c > 0.9995                       % quasi allineati: lineare + normalizza
        q = q0 + t*(q1 - q0);
        q = q/norm(q);
        return
    end
    th0 = acos(max(-1,min(1,c)));
    th  = th0*t;
    q2  = q1 - q0*c;  q2 = q2/norm(q2);
    q   = q0*cos(th) + q2*sin(th);
end
