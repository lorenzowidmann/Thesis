%% Diagnostica: candidati scartati per disaccordo con l'odometria (sezione 8)
%
% NON modifica LoopClosureAttempt_Piano1.m, non cambia nessuna soglia.
% Riusa il checkpoint gia' salvato (kfClouds/kfPoses), non rilegge il bag
% da 3GB. Ripete la logica di rilevamento (sezione 7 Scan Context, 7b
% prossimita') e verifica ICP (sezione 8), ma stavolta registra TUTTI gli
% esiti (accettato / RMSE alto / disaccordo con odometria / ICP fallito),
% non solo gli accettati che finiscono in loopConstraints.
%
% Perche': i candidati scartati per "disaccordo con l'odometria"
% (loopMaxTransErr/loopMaxRotErr) sono quelli che la loop closure servirebbe
% davvero a correggere, se il disaccordo e' vera deriva accumulata e non un
% falso positivo geometrico. Un loop lontano nel percorso (gap keyframe
% grande) scartato per questo motivo va controllato a mano, non buttato.
%
% Solo diagnostica: stampa un riepilogo a console, non esporta CSV/immagini.

clear
close all
clc

checkpointFile = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\loop_closure_checkpoint.mat";
load(checkpointFile, 'kfPoses', 'kfClouds', 'nKF');

% Stessi parametri di LoopClosureAttempt_Piano1.m (sezioni 7, 7b, 8),
% nessuna soglia cambiata.
scDistThreshold  = 0.15;
scNumExcluded    = 30;
scMaxDetections  = 3;
proxRadius       = 3.0;
proxMinGap       = 40;
proxMaxCand      = 300;
icpMaxRMSE       = 0.30;
icpMaxDistance   = 1.0;
loopMaxTransErr  = 2.0;
loopMaxRotErr    = 10.0;

%% Rilevamento loop con Scan Context (= sezione 7 dello script originale)
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

%% Candidati per prossimita' spaziale (= sezione 7b)
kfXYZ = vertcat(kfPoses.Translation);
dx  = kfXYZ(:,1) - kfXYZ(:,1)';
dy  = kfXYZ(:,2) - kfXYZ(:,2)';
dXY = sqrt(dx.^2 + dy.^2);
[II, JJ] = ndgrid(1:nKF, 1:nKF);
gapIdx = abs(II - JJ);
revisit = triu(dXY <= proxRadius & gapIdx >= proxMinGap, 1);
[ri, rj] = find(revisit);
proxCandidates = [ri, rj];
if size(proxCandidates, 1) > proxMaxCand
    dSel = dXY(sub2ind([nKF nKF], ri, rj));
    [~, ord] = sort(dSel, 'ascend');
    proxCandidates = proxCandidates(ord(1:proxMaxCand), :);
end
fprintf('Candidati per prossimita'' XY: %d\n', size(proxCandidates, 1));

if isempty(loopCandidates)
    allCandidates = proxCandidates;
elseif isempty(proxCandidates)
    allCandidates = loopCandidates;
else
    allCandidates = unique([double(loopCandidates); proxCandidates], 'rows');
end
fprintf('Candidati totali da verificare: %d\n\n', size(allCandidates, 1));

%% Verifica ICP (= sezione 8), ma logga TUTTI gli esiti, non solo gli accettati
results = struct('i', {}, 'j', {}, 'gap', {}, 'rmse', {}, 'transErr', {}, ...
                  'rotErr', {}, 'outcome', {});

fprintf('Verifica ICP dei candidati (solo diagnostica, nessuna soglia cambiata)...\n');
for c = 1:size(allCandidates, 1)
    i = allCandidates(c, 1);
    j = allCandidates(c, 2);

    Arel = kfPoses(i).A \ kfPoses(j).A;
    ArelFlat = Arel;
    ArelFlat(3, 4) = 0;
    initGuesses = {Arel, ArelFlat};
    bestRmse  = inf;
    bestTform = [];
    coarseInlierDistance = max(icpMaxDistance * 6, 6.0);

    for g = 1:numel(initGuesses)
        try
            [tfCoarse, ~, ~] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', rigidtform3d(initGuesses{g}), ...
                'InlierDistance', coarseInlierDistance, 'MaxIterations', 50);
            [tf, ~, rmse] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', tfCoarse, 'InlierDistance', icpMaxDistance);
            if rmse < bestRmse
                bestRmse  = rmse;
                bestTform = tf;
            end
        catch
            % come nello script originale: ICP fallito su questa init guess,
            % si prova l'altra
        end
    end

    if isempty(bestTform)
        results(end+1) = struct('i', i, 'j', j, 'gap', j - i, 'rmse', NaN, ...
            'transErr', NaN, 'rotErr', NaN, 'outcome', "icp_failed"); %#ok<SAGROW>
        continue
    end

    Terr     = Arel \ bestTform.A;
    transErr = norm(Terr(1:3, 4));
    rotErr   = abs(rad2deg(acos(max(-1, min(1, (trace(Terr(1:3,1:3)) - 1) / 2)))));

    if bestRmse > icpMaxRMSE
        outcome = "rmse_alto";
    elseif transErr > loopMaxTransErr || rotErr > loopMaxRotErr
        outcome = "disaccordo_odometria";
    else
        outcome = "accettato";
    end

    results(end+1) = struct('i', i, 'j', j, 'gap', j - i, 'rmse', bestRmse, ...
        'transErr', transErr, 'rotErr', rotErr, 'outcome', outcome); %#ok<SAGROW>
end

%% Riepilogo
outcomes = string({results.outcome});
fprintf('\n--- Riepilogo verifica ICP (%d candidati totali) ---\n', numel(results));
fprintf('  accettati:                    %d\n', nnz(outcomes == "accettato"));
fprintf('  scartati, RMSE alto:          %d\n', nnz(outcomes == "rmse_alto"));
fprintf('  scartati, disacc. odometria:  %d\n', nnz(outcomes == "disaccordo_odometria"));
fprintf('  ICP fallito:                  %d\n', nnz(outcomes == "icp_failed"));

rejOdom = results(outcomes == "disaccordo_odometria");
if isempty(rejOdom)
    fprintf('\nNessun candidato scartato per disaccordo con l''odometria.\n');
else
    gaps = [rejOdom.gap];
    [~, ord] = sort(gaps, 'descend');
    rejOdom = rejOdom(ord);

    fprintf(['\n--- Candidati scartati per disaccordo con l''odometria, ' ...
        'ordinati per gap keyframe (decrescente) ---\n']);
    fprintf('%6s  %6s  %6s  %8s  %11s  %11s\n', ...
        'i', 'j', 'gap', 'rmse(m)', 'transErr(m)', 'rotErr(deg)');
    for k = 1:numel(rejOdom)
        r = rejOdom(k);
        fprintf('%6d  %6d  %6d  %8.3f  %11.2f  %11.1f\n', ...
            r.i, r.j, r.gap, r.rmse, r.transErr, r.rotErr);
    end
end
