function [tf, rmse] = checkPair2(i, j, kfPoses, kfClouds)
    Arel = kfPoses(i).A \ kfPoses(j).A;
    ArelFlat = Arel; ArelFlat(3,4) = 0;
    best = inf; tf = [];
    for g = {Arel, ArelFlat}
        try
            tfC = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', rigidtform3d(g{1}), ...
                'InlierDistance', 6.0, 'MaxIterations', 50);
            [t, ~, r] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', tfC, 'InlierDistance', 1.0);
            if r < best
                best = r;
                tf = t;
            end
        catch
            % ICP fallito con questa inizializzazione: si prova l'altra,
            % o si resta a best = inf se falliscono entrambe
        end
    end
    rmse = best;   % rimane Inf se ICP e' fallito con entrambe le init
end

Bidx = [109 110 111 112 113 114];
Cidx = [123 124 125 126 127 128 129 130 131 132 133 134];
found = [];
for i = Bidx
    for j = Cidx
        [~, r] = checkPair2(i, j, kfPoses, kfClouds);
        if r < 0.30
            fprintf('  *** CANDIDATO: %d -> %d rmse %.3f ***\n', i, j, r);
            found(end+1, :) = [i, j, r]; %#ok<AGROW>
        end
    end
end
fprintf('Candidati trovati: %d\n', size(found,1));

i = 112; j = 134;
Arel = kfPoses(i).A \ kfPoses(j).A;
[tf, rmse] = checkPair2(i, j, kfPoses, kfClouds);

Terr = Arel \ tf.A;
transErr = norm(Terr(1:3,4));
rotErr = abs(rad2deg(acos(max(-1, min(1, (trace(Terr(1:3,1:3)) - 1)/2)))));

fprintf('%d -> %d : rmse %.3f m, scarto odometria %.2f m / %.1f deg\n', ...
    i, j, rmse, transErr, rotErr);