%% Prova: correzione dell'assetto tramite la normale del pavimento
%
% La normale del pavimento e' una direzione fisica FISSA. Se al keyframe
% k-1 la vedo come n(k-1) nel mio frame body, e al keyframe k come n(k),
% allora la rotazione relativa vera R_rel deve soddisfare
%     R_rel * n(k) = n(k-1)
% Questo vincola 2 DOF (roll e pitch); il terzo (yaw attorno alla normale)
% resta all'odometria, che sullo yaw non deriva in modo problematico.
%
% Si corregge quindi ogni rotazione relativa dell'odometria con la rotazione
% minima che soddisfa il vincolo, e si ricostruisce il pose graph.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
load(checkpointFile, 'kfPoses', 'nKF', 'kfClouds', 'loopConstraints', 'nLoops', 'kfVoxel');

%% 1. Normale del pavimento per ogni keyframe (frame body)
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
    n = model.Normal(:);
    if n(3) < 0, n = -n; end
    nBody(k,:) = n' / norm(n);
end
valid = ~any(isnan(nBody), 2);
fprintf('Normali pavimento valide: %d su %d\n', nnz(valid), nKF);

%% 2. Vincoli relativi corretti
infoOdom   = 100;
infoLoop   = 500;
sigma0Loop = kfVoxel;

pg = poseGraph3D;
nCorrected = 0;

for k = 2:nKF
    Arel = kfPoses(k-1).A \ kfPoses(k).A;
    Rrel = Arel(1:3,1:3);
    trel = Arel(1:3,4);

    if valid(k) && valid(k-1)
        v  = Rrel * nBody(k,:)';       % normale di k portata nel frame di k-1
        tg = nBody(k-1,:)';            % dove dovrebbe finire
        v  = v / norm(v);  tg = tg / norm(tg);

        ax = cross(v, tg);
        s  = norm(ax);
        c  = dot(v, tg);
        if s > 1e-8
            ax = ax / s;
            ang = atan2(s, c);
            K = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
            C = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);   % Rodrigues
            Rrel = C * Rrel;
            nCorrected = nCorrected + 1;
        end
    end

    A = eye(4); A(1:3,1:3) = Rrel; A(1:3,4) = trel;
    addRelativePose(pg, tform2measurement(A), buildInfoVector(infoOdom), k-1, k);
end
fprintf('Rotazioni relative corrette col pavimento: %d su %d\n', nCorrected, nKF-1);

for c = 1:nLoops
    i = loopConstraints{c}{1};
    j = loopConstraints{c}{2};
    tf = loopConstraints{c}{3};
    rmse = loopConstraints{c}{4};
    infoEff = infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
    addRelativePose(pg, tform2measurement(tf.A), buildInfoVector(infoEff), i, j);
end

pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');
nodesOpt = nodeEstimates(pgOpt);

%% 3. Verifica
zRaw = vertcat(kfPoses.Translation); zRaw = zRaw(:,3);
zOpt = nodesOpt(:,3);
fprintf('\n--- Deriva verticale della traiettoria ---\n');
fprintf('Prima: %.2f m\n', max(zRaw)-min(zRaw));
fprintf('Dopo:  %.2f m\n', max(zOpt)-min(zOpt));

% Inclinazione residua del pavimento con le nuove pose
tiltAfter = nan(nKF,1);
for k = 1:nKF
    if ~valid(k), continue; end
    R = quat2rotm(nodesOpt(k,4:7));
    nMap = R * nBody(k,:)';
    if nMap(3) < 0, nMap = -nMap; end
    tiltAfter(k) = rad2deg(acos(max(-1,min(1,nMap(3)))));
end
fprintf('\nInclinazione pavimento DOPO: mediana %.2f deg, max %.2f deg\n', ...
    median(tiltAfter(~isnan(tiltAfter))), max(tiltAfter));

%% Funzioni
function v = buildInfoVector(w)
    M = diag([w w w w w w]);
    v = zeros(1,21); n = 0;
    for i = 1:6
        for j = i:6
            n = n+1; v(n) = M(i,j);
        end
    end
end

function meas = tform2measurement(A)
    meas = [A(1:3,4)', rotm2quat(A(1:3,1:3))];
end
