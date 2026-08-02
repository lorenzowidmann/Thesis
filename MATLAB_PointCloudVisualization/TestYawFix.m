%% Correzione della deriva di yaw usando la direzione dei muri
%
% ASSUNZIONE, da dichiarare esplicitamente: si assume che l'edificio abbia
% una direzione dominante coerente (muri paralleli o perpendicolari fra
% loro). E' vero nella grande maggioranza degli edifici, ma NON e' una
% misura: e' un'ipotesi sul mondo. Se l'edificio avesse davvero un corridoio
% a 45 gradi, questa correzione lo raddrizzerebbe a torto.
%
% Cio' che la rende difendibile QUI e' che l'azimut dei muri fa uno scalino
% di ~8 gradi esattamente al keyframe dove sappiamo gia' che si e' rotto
% anche roll/pitch. Un edificio non ruota a meta' corridoio.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
load(fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat'), ...
    'kfPoses', 'nKF', 'kfClouds', 'mapVoxel');

yawSmooth = 9;    % finestra (keyframe) per lisciare la stima

%% 1. Azimut dominante dei muri, per keyframe
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
    isWall = abs(nMap(:,3)) < 0.2;          % normale orizzontale => muro
    if nnz(isWall) < 50, continue; end
    a = atan2(nMap(isWall,2), nMap(isWall,1));
    % ripiegatura mod 90 deg: x4 porta il periodo a 360 deg
    azi(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
end
valid = ~isnan(azi);
fprintf('Azimut stimato su %d keyframe su %d\n', nnz(valid), nKF);

%% 2. Riferimento e correzione, in dominio ripiegato
% Si lavora sull'angolo x4 per evitare il salto 0/90.
z = nan(nKF,1) + 1i*nan;
z(valid) = exp(1i*4*deg2rad(azi(valid)));

% riferimento: media circolare dei primi keyframe, prima della rottura
nRef  = min(40, nKF);
zRef  = mean(z(1:nRef), 'omitnan');
aRef  = mod(rad2deg(angle(zRef))/4, 90);
fprintf('Azimut di riferimento (primi %d kf): %.2f deg\n', nRef, aRef);

% lisciatura circolare con finestra mobile
zS = nan(nKF,1) + 1i*nan;
h  = floor(yawSmooth/2);
for k = 1:nKF
    lo = max(1, k-h); hi = min(nKF, k+h);
    w  = z(lo:hi);
    w  = w(~isnan(w));
    if isempty(w), continue; end
    zS(k) = mean(w);
end

% buchi: si riempiono col valore valido piu' vicino
vi = find(~isnan(zS));
for k = 1:nKF
    if ~isnan(zS(k)), continue; end
    [~, i] = min(abs(vi - k));
    zS(k) = zS(vi(i));
end

% correzione di yaw: differenza rispetto al riferimento, riportata in [-45,45]
dYaw = zeros(nKF,1);
for k = 1:nKF
    d = rad2deg(angle(zS(k) / zRef)) / 4;   % differenza nel dominio ripiegato
    dYaw(k) = mod(d + 45, 90) - 45;
end
fprintf('Correzione di yaw: min %.2f deg, max %.2f deg\n', min(dYaw), max(dYaw));

%% 3. Applicazione: rotazione attorno alla verticale del mondo
Rc = cell(nKF,1);
for k = 1:nKF
    th = -deg2rad(dYaw(k));
    Cz = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
    Rc{k} = Cz * kfPoses(k).R;
end

% re-integrazione delle posizioni con gli assetti corretti
pOld = vertcat(kfPoses.Translation);
pNew = zeros(nKF,3); pNew(1,:) = pOld(1,:);
for k = 2:nKF
    dLocal = kfPoses(k-1).R' * (pOld(k,:) - pOld(k-1,:))';
    pNew(k,:) = pNew(k-1,:) + (Rc{k-1}*dLocal)';
end

posesNew = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF, posesNew(k) = rigidtform3d(Rc{k}, pNew(k,:)); end

%% 4. Verifica: quanto sono coerenti i muri dopo?
aziAfter = nan(nKF,1);
for k = 1:nKF
    pc = kfClouds{k};
    if pc.Count < 200, continue; end
    try
        nrm = pcnormals(pc, 20);
    catch
        continue
    end
    nMap = (Rc{k} * nrm')';
    isWall = abs(nMap(:,3)) < 0.2;
    if nnz(isWall) < 50, continue; end
    a = atan2(nMap(isWall,2), nMap(isWall,1));
    aziAfter(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
end

cm  = @(v) mod(rad2deg(angle(mean(exp(1i*4*deg2rad(v(~isnan(v)))))))/4, 90);
% dispersione circolare: 1 - |R|, va da 0 (allineati) a 1 (casuali)
cdisp = @(v) 1 - abs(mean(exp(1i*4*deg2rad(v(~isnan(v))))));

fprintf('\n--- Coerenza della direzione dei muri ---\n');
fprintf('%-14s %12s %12s %14s\n', '', 'meta 1', 'meta 2', 'dispersione');
h1 = 1:floor(nKF/2); h2 = floor(nKF/2)+1:nKF;
fprintf('%-14s %11.2f° %11.2f° %14.3f\n', 'prima', ...
    cm(azi(h1)), cm(azi(h2)), cdisp(azi));
fprintf('%-14s %11.2f° %11.2f° %14.3f\n', 'dopo', ...
    cm(aziAfter(h1)), cm(aziAfter(h2)), cdisp(aziAfter));

acc = cell(nKF,1);
for k = 1:nKF, acc{k} = pctransform(kfClouds{k}, posesNew(k)).Location; end
mapNew = pcdownsample(pointCloud(vertcat(acc{:})), 'gridAverage', mapVoxel);
accO = cell(nKF,1);
for k = 1:nKF, accO{k} = pctransform(kfClouds{k}, kfPoses(k)).Location; end
mapOld = pcdownsample(pointCloud(vertcat(accO{:})), 'gridAverage', mapVoxel);

fprintf('\nSpan mappa X: %.2f -> %.2f m\n', diff(mapOld.XLimits), diff(mapNew.XLimits));
fprintf('Span mappa Y: %.2f -> %.2f m\n', diff(mapOld.YLimits), diff(mapNew.YLimits));
fprintf('Span mappa Z: %.2f -> %.2f m\n', diff(mapOld.ZLimits), diff(mapNew.ZLimits));

figure('Color','k','Name','Yaw: prima / dopo');
subplot(1,2,1); pcshow(mapOld,'MarkerSize',12); view(2); axis equal;
title('PRIMA','Color','w'); xlabel('X'); ylabel('Y');
subplot(1,2,2); pcshow(mapNew,'MarkerSize',12); view(2); axis equal;
title('DOPO','Color','w'); xlabel('X'); ylabel('Y');
