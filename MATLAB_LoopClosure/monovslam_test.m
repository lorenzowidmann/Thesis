%% monovslam_test.m
% Monocular visual SLAM (ORB-based) su session_right.mp4 (ZED 2i, HD1080,
% 30 fps) usando la classe monovslam del Computer Vision Toolbox.
%
% Pipeline:
%   1) Legge metadata.json e verifica risoluzione/fps/formato attesi.
%   2) Legge session_right.mp4 con VideoReader, frame by frame.
%   3) Alimenta monovslam con ogni frame (undistorto) -> stima traiettoria.
%   4) Estrae le pose delle key frame con poses(vslam), salva:
%        - plot 2D X-Y (vista dall'alto) in PNG
%        - pose in formato TUM (timestamp x y z qx qy qz qw) in TXT
%
% NOTA IMPORTANTE - SCALA METRICA:
%   Questa e' odometria MONOCULARE: la scala assoluta (metri) NON e'
%   osservabile da una sola camera. monovslam stima una traiettoria a
%   meno di un fattore di scala arbitrario (fissato dalla baseline
%   iniziale di inizializzazione, non da una misura fisica).
%   -> Confrontare questa traiettoria con poses_tum.txt di
%      FAST-LIO-SAM-SC-QN (LiDAR, scala metrica reale) SOLO in FORMA
%      (dopo allineamento tipo Umeyama con stima di scala, o dopo
%      normalizzazione), MAI leggendo le coordinate in metri come se
%      fossero direttamente confrontabili.
%
% Requisiti: Computer Vision Toolbox (classe monovslam, R2023b+).
% Non modifica alcun file esistente del progetto o della sessione dati.

clear; clc; close all;

%% --- Config percorsi ---
sessionDir = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_155047';
videoFile  = fullfile(sessionDir, 'session_right.mp4');
metaFile   = fullfile(sessionDir, 'metadata.json');

outDir  = sessionDir;   % output salvati nella cartella sessione (nuovi file)
pngOut  = fullfile(outDir, 'monovslam_trajectory_topview.png');
tumOut  = fullfile(outDir, 'monovslam_poses_tum.txt');

%% --- Verifica Computer Vision Toolbox / classe monovslam ---
if isempty(which('monovslam')) || isempty(which('cameraIntrinsics'))
    error(['monovslam_test:missingToolbox', newline, ...
        'Computer Vision Toolbox (classe monovslam) non trovato/non licenziato. ', ...
        'Verificare installazione/licenza (necessario R2023b o successivo).']);
end

%% --- Legge metadata.json e verifica risoluzione/fps/formato ---
assert(isfile(metaFile), 'metadata.json non trovato: %s', metaFile);
meta = jsondecode(fileread(metaFile));

fprintf('metadata.json: camera=%s, resolution=%s, fps=%g, mp4=%s, export_eye=%s\n', ...
    meta.camera.model, meta.camera.resolution, meta.camera.fps, ...
    meta.recording.mp4_path, meta.recording.export_eye);

assert(strcmpi(meta.camera.resolution, 'HD1080'), ...
    'Risoluzione attesa HD1080 (1920x1080), trovata "%s" in metadata.json.', meta.camera.resolution);
assert(meta.camera.fps == 30, ...
    'FPS atteso 30, trovato %g in metadata.json.', meta.camera.fps);
assert(isfile(videoFile), 'Video non trovato: %s', videoFile);
assert(strcmp(meta.recording.mp4_path, 'session_right.mp4'), ...
    'metadata.json indica mp4_path="%s", atteso "session_right.mp4".', meta.recording.mp4_path);

%% --- Intrinseci camera ZED 2i, occhio destro, HD1080 ---
% Fonte: ClaudeCode/lvt2calib/data/camera_info/zed_right_intrinsic.yaml
% (calibrazione MATLAB, 25 immagini, errore reproiezione medio 0.2151 px,
%  risoluzione 1920x1080 - coerente con questa registrazione).
fx = 1412.3362; fy = 1414.4716;
cx = 1012.8503; cy = 569.7181;
k1 = -0.1558253; k2 = 0.009026829;         % distorsione radiale
p1 = 0.0006208599; p2 = 0.0005667587;      % distorsione tangenziale
imageSize = [1080 1920];                    % [height width]

intrinsicsDistorted = cameraIntrinsics([fx fy], [cx cy], imageSize, ...
    RadialDistortion=[k1 k2], TangentialDistortion=[p1 p2]);

%% --- VideoReader ---
vr = VideoReader(videoFile);
fprintf('Video: %s\n  Resolution: %dx%d, FrameRate: %.3f fps, Duration: %.2f s\n', ...
    videoFile, vr.Width, vr.Height, vr.FrameRate, vr.Duration);
assert(vr.Width == imageSize(2) && vr.Height == imageSize(1), ...
    'Risoluzione video (%dx%d) diversa dagli intrinseci (%dx%d).', ...
    vr.Width, vr.Height, imageSize(2), imageSize(1));

%% --- Loop principale: addFrame frame by frame ---
vslam = monovslam(intrinsicsDistorted); % Intrinsics con distorsione: monovslam
                                         % usa cameraIntrinsics.undistortImage
                                         % internamente per rettificare i punti.

frameTimestamps = zeros(0, 1); % secondi dall'inizio del video, indicizzati per ordine di addFrame
frameIdx = 0;
tStart = tic;

% NOTA: isDone(vslam) e' true sull'oggetto appena costruito (prima di
% qualunque addFrame), quindi va controllato SOLO dopo aver aggiunto
% almeno un frame (mai come guardia iniziale del while).
while hasFrame(vr)
    t = vr.CurrentTime;      % timestamp del frame PRIMA di leggerlo
    I = readFrame(vr);
    frameIdx = frameIdx + 1;
    frameTimestamps(frameIdx, 1) = t;

    addFrame(vslam, I);

    if mod(frameIdx, 200) == 0
        fprintf('  frame %d, t=%.2f s, elapsed=%.1f s\n', frameIdx, t, toc(tStart));
    end

    if isDone(vslam)
        fprintf('  isDone=true al frame %d (t=%.2f s): stop elaborazione.\n', frameIdx, t);
        break;
    end
end

fprintf('Elaborazione completata: %d frame letti in %.1f s.\n', frameIdx, toc(tStart));

status = checkStatus(vslam);
fprintf('Stato finale monovslam: %s\n', string(status));
if isDone(vslam)
    warning('monovslam_test:trackingLost', ...
        'monovslam ha interrotto il tracking prima della fine del video (tracking perso in modo irreversibile).');
end

%% --- Estrazione pose delle key frame ---
[camPoses, keyFrameIDs] = poses(vslam);
nPoses = numel(camPoses);
assert(nPoses > 0, 'Nessuna posa stimata: la SLAM monoculare non e'' riuscita a inizializzare/tracciare.');
fprintf('Pose stimate (key frame): %d\n', nPoses);

poseTimestamps = frameTimestamps(keyFrameIDs); % secondi relativi all'inizio del video

X = zeros(nPoses, 1); Y = zeros(nPoses, 1); Z = zeros(nPoses, 1);
Q = zeros(nPoses, 4); % [qx qy qz qw]
for i = 1:nPoses
    T = camPoses(i).Translation;
    X(i) = T(1); Y(i) = T(2); Z(i) = T(3);
    qwxyz = rotm2quat(camPoses(i).R);   % MATLAB: [qw qx qy qz]
    Q(i, :) = [qwxyz(2) qwxyz(3) qwxyz(4) qwxyz(1)]; % -> TUM: [qx qy qz qw]
end

%% --- Plot 2D X-Y (vista dall'alto) ---
fig = figure('Visible', 'off', 'Color', 'w');
plot(X, Y, '-o', 'MarkerSize', 3, 'LineWidth', 1.2);
hold on;
plot(X(1), Y(1), 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot(X(end), Y(end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'End');
axis equal; grid on;
xlabel('X (unita'' arbitrarie, scala monoculare non metrica)');
ylabel('Y (unita'' arbitrarie, scala monoculare non metrica)');
title({'Traiettoria monovslam - vista dall''alto (X-Y)', ...
    'ATTENZIONE: odometria monoculare, scala NON metrica'});
legend('Traiettoria', 'Start', 'End', 'Location', 'best');
exportgraphics(fig, pngOut, 'Resolution', 200);
close(fig);
fprintf('Plot 2D salvato in: %s\n', pngOut);

%% --- Salvataggio pose in formato TUM ---
% timestamp assoluto = started_utc (da metadata.json) + tempo relativo nel video.
% Permette un confronto (dopo allineamento di scala/forma) con poses_tum.txt
% di FAST-LIO-SAM-SC-QN, che usa timestamp assoluti.
t0 = datetime(meta.session.started_utc, ...
    'InputFormat', 'uuuu-MM-dd''T''HH:mm:ss.SSSSSS''Z''', 'TimeZone', 'UTC');
t0Posix = posixtime(t0);
absTimestamps = t0Posix + poseTimestamps;

fid = fopen(tumOut, 'w');
assert(fid ~= -1, 'Impossibile aprire %s in scrittura.', tumOut);
fprintf(fid, '# timestamp x y z qx qy qz qw\n');
fprintf(fid, '# monovslam (monoculare, scala NON metrica) su %s\n', videoFile);
fprintf(fid, '# key_frame_id relativo ad addFrame; video_time_s = timestamp - %.6f\n', t0Posix);
for i = 1:nPoses
    fprintf(fid, '%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n', ...
        absTimestamps(i), X(i), Y(i), Z(i), Q(i,1), Q(i,2), Q(i,3), Q(i,4));
end
fclose(fid);
fprintf('Pose TUM-like salvate in: %s\n', tumOut);

fprintf('\nFATTO. Ricorda: traiettoria monoculare a scala arbitraria -> confronto con LiDAR solo in forma.\n');
