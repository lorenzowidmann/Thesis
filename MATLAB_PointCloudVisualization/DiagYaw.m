%% I corridoi non sono a 90 gradi: deriva di yaw o edificio non ortogonale?
%
% Il pavimento (sezione 6b) vincola roll e pitch, non lo yaw: la normale di
% un piano orizzontale non dice nulla sulla rotazione attorno alla verticale.
% Se c'e' deriva di yaw, la direzione dei MURI deve ruotare progressivamente
% lungo il percorso; se invece l'edificio e' davvero non ortogonale, la
% direzione resta stabile ma con piu' famiglie distinte.
%
% Si usano le normali dei punti: quelle quasi orizzontali appartengono a muri
% verticali. L'azimut viene ripiegato modulo 90 gradi, cosi' i quattro lati
% di un corridoio ortogonale cadono tutti nello stesso valore.
clear
clc

bagPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45';
load(fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat'), ...
    'kfPoses', 'nKF', 'kfClouds');

azi   = nan(nKF,1);   % azimut dominante dei muri, ripiegato in [0,90)
nWall = zeros(nKF,1);

for k = 1:nKF
    pc = kfClouds{k};
    if pc.Count < 200, continue; end

    try
        nrm = pcnormals(pc, 20);
    catch
        continue
    end

    % normale portata nel frame mappa
    nMap = (kfPoses(k).R * nrm')';

    % muri: normale quasi orizzontale
    isWall = abs(nMap(:,3)) < 0.2;
    nWall(k) = nnz(isWall);
    if nWall(k) < 50, continue; end

    a = atan2(nMap(isWall,2), nMap(isWall,1));   % azimut in [-pi,pi]

    % Ripiegatura modulo 90 deg: si moltiplica per 4 l'angolo cosi' il
    % periodo di 90 deg diventa 360 deg e la media circolare e' ben definita.
    z = exp(1i * 4 * a);
    azi(k) = mod(rad2deg(angle(mean(z))) / 4, 90);
end

ok = ~isnan(azi);
fprintf('Keyframe con muri sufficienti: %d su %d\n\n', nnz(ok), nKF);

fprintf('%5s %10s %10s\n', 'kf', 'azimut', 'nWall');
for k = 1:6:nKF
    if ok(k)
        fprintf('%5d %9.2f° %10d\n', k, azi(k), nWall(k));
    end
end

% Deriva: differenza tra prima e seconda meta' del percorso.
% Attenzione al wrap a 90 deg: si confrontano con media circolare.
h1 = azi(ok & (1:nKF)' <= nKF/2);
h2 = azi(ok & (1:nKF)' >  nKF/2);
cm = @(v) mod(rad2deg(angle(mean(exp(1i*4*deg2rad(v)))))/4, 90);
fprintf('\nAzimut medio prima meta:  %.2f deg\n', cm(h1));
fprintf('Azimut medio seconda meta: %.2f deg\n', cm(h2));

d = cm(h2) - cm(h1);
d = mod(d + 45, 90) - 45;    % differenza minima con periodo 90
fprintf('Deriva di yaw stimata:     %.2f deg\n', d);

fprintf('\nDispersione (std entro meta): prima %.2f deg, seconda %.2f deg\n', ...
    std(h1), std(h2));
