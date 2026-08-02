%% Prova 2: riallineamento ASSOLUTO dell'assetto sul pavimento
%
% Differenza rispetto alla prova 1: li' si correggevano i vincoli relativi e
% si lasciava all'ottimizzatore il risultato, che poteva disfare la
% correzione per soddisfare gli altri vincoli (tutti a peso pieno sui 6 DOF).
% Qui l'assetto assoluto di ogni keyframe viene imposto dal pavimento
% osservato (roll/pitch, 2 DOF, privi di deriva per costruzione), mentre
% yaw e modulo dello spostamento restano dall'odometria, che su quelli e'
% affidabile. La traiettoria viene poi re-integrata con gli assetti corretti.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'nKF', 'kfClouds', 'loopConstraints', 'nLoops', 'kfVoxel');

%% 1. Normale del pavimento (frame body)
nBody = nan(nKF, 3);
for k = 1:nKF
    loc = kfClouds{k}.Location;
    zmin = min(loc(:,3));
    cand = loc(loc(:,3) < zmin + 0.30, :);
    if size(cand,1) < 50, continue; end
    try
        [model, inl] = pcfitplane(pointCloud(cand), 0.06, [0 0 1], 30);
        if numel(inl) < 40, continue; end
    catch
        continue
    end
    n = model.Normal(:); if n(3) < 0, n = -n; end
    nBody(k,:) = n' / norm(n);
end
valid = ~any(isnan(nBody), 2);
fprintf('Normali pavimento valide: %d su %d\n', nnz(valid), nKF);

%% 2. Assetto assoluto corretto
Rc = cell(nKF,1);
Cprev = eye(3);
for k = 1:nKF
    R = kfPoses(k).R;
    if valid(k)
        nMap = R * nBody(k,:)';
        if nMap(3) < 0, nMap = -nMap; end
        nMap = nMap / norm(nMap);
        tg = [0;0;1];
        ax = cross(nMap, tg); s = norm(ax); c = dot(nMap, tg);
        if s > 1e-8
            ax = ax/s; ang = atan2(s,c);
            K = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
            Cprev = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);
        else
            Cprev = eye(3);
        end
    end
    Rc{k} = Cprev * R;    % nei buchi si mantiene l'ultima correzione nota
end

%% 3. Re-integrazione delle posizioni con gli assetti corretti
pOld = vertcat(kfPoses.Translation);
pNew = zeros(nKF,3);
pNew(1,:) = pOld(1,:);
for k = 2:nKF
    dLocal = kfPoses(k-1).R' * (pOld(k,:) - pOld(k-1,:))';   % spostamento in frame body
    pNew(k,:) = pNew(k-1,:) + (Rc{k-1} * dLocal)';
end

%% 4. Pose graph sulle pose corrette + loop
infoOdom = 100; infoLoop = 500; sigma0Loop = kfVoxel;
pg = poseGraph3D;
for k = 2:nKF
    A0 = eye(4); A0(1:3,1:3) = Rc{k-1}; A0(1:3,4) = pNew(k-1,:)';
    A1 = eye(4); A1(1:3,1:3) = Rc{k};   A1(1:3,4) = pNew(k,:)';
    addRelativePose(pg, tform2measurement(A0\A1), buildInfoVector(infoOdom), k-1, k);
end
for c = 1:nLoops
    i = loopConstraints{c}{1}; j = loopConstraints{c}{2};
    tf = loopConstraints{c}{3}; rmse = loopConstraints{c}{4};
    infoEff = infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
    addRelativePose(pg, tform2measurement(tf.A), buildInfoVector(infoEff), i, j);
end
pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');
nodesOpt = nodeEstimates(pgOpt);

%% 5. Verifica
fprintf('\n--- Deriva verticale della traiettoria ---\n');
fprintf('Odometria grezza:            %.2f m\n', max(pOld(:,3))-min(pOld(:,3)));
fprintf('Dopo riallineamento:         %.2f m\n', max(pNew(:,3))-min(pNew(:,3)));
fprintf('Dopo riallineamento + loop:  %.2f m\n', max(nodesOpt(:,3))-min(nodesOpt(:,3)));

tiltA = nan(nKF,1);
for k = 1:nKF
    if ~valid(k), continue; end
    R = quat2rotm(nodesOpt(k,4:7));
    nMap = R * nBody(k,:)'; if nMap(3) < 0, nMap = -nMap; end
    tiltA(k) = rad2deg(acos(max(-1,min(1,nMap(3)))));
end
fprintf('\nInclinazione pavimento: mediana %.2f deg, max %.2f deg\n', ...
    median(tiltA(~isnan(tiltA))), max(tiltA));

function v = buildInfoVector(w)
    M = diag([w w w w w w]); v = zeros(1,21); n = 0;
    for i = 1:6
        for j = i:6, n = n+1; v(n) = M(i,j); end
    end
end
function meas = tform2measurement(A)
    meas = [A(1:3,4)', rotm2quat(A(1:3,1:3))];
end
