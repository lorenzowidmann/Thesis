load('C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\loop_closure_checkpoint.mat');

checkPair2(72, 133, kfPoses, kfClouds);   % A-C, gia' scartato come falso positivo
checkPair2(74, 134, kfPoses, kfClouds);   % A-C, idem
checkPair2(112, 128, kfPoses, kfClouds);  % B-C, mai testato
checkPair2(113, 130, kfPoses, kfClouds);  % B-C, mai testato

function [tf, rmse] = checkPair2(i, j, kfPoses, kfClouds)
    Arel = kfPoses(i).A \ kfPoses(j).A;
    ArelFlat = Arel; ArelFlat(3,4) = 0;
    best = inf; tf = [];
    for g = {Arel, ArelFlat}
        tfC = pcregistericp(kfClouds{j}, kfClouds{i}, ...
            'InitialTransform', rigidtform3d(g{1}), 'InlierDistance', 6.0, 'MaxIterations', 50);
        [t, ~, r] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
            'InitialTransform', tfC, 'InlierDistance', 1.0);
        if r < best, best = r; tf = t; end
    end
    rmse = best;
    fprintf('%d -> %d : rmse %.3f\n', i, j, rmse);
end