%% Online Semantic Correlation Learning (OSCL) demo
% Change dataFile to run another dataset with the same MAT-file format.

clearvars;
clc;

codeFolder = fileparts(mfilename('fullpath'));
addpath(codeFolder);

%% Settings
% dataFile = fullfile(codeFolder, 'NUSWIDE21.mat');
dataFile = fullfile(codeFolder, 'MIRFlickr.mat');
resultFolder = fullfile(codeFolder, 'results');

options.ell = 10;                         % Latent dimension, ell
options.lambdaX = 1e+1;                  % Image regularizer, lambda_x
options.lambdaY = 1e+1;                  % Text regularizer, lambda_y
options.maxK = 2000;                     % Precision at ranks 1:maxK
options.computePrecisionRecall = true;
options.verbose = true;

%% Load data
% The released dataset uses L for labels and Test for query data. The
% aliases below adopt the X/Y/Z notation used in the manuscript and code.
data = load(dataFile, 'XChunk', 'YChunk', 'LChunk', ...
    'XTest', 'YTest', 'LTest');

XChunk = data.XChunk;
YChunk = data.YChunk;
ZChunk = data.LChunk;
XQuery = data.XTest;
YQuery = data.YTest;
ZQuery = data.LTest;

fprintf('Dataset: %s\n', dataFile);
fprintf('%d training chunks and %d query pairs\n\n', ...
    numel(XChunk), size(XQuery, 1));

%% Run OSCL
[results, models, options] = evaluate_OSCL( ...
    XChunk, YChunk, ZChunk, XQuery, YQuery, ZQuery, options);

numRounds = numel(results);
summary = table((1:numRounds)', [results.numTrainingSamples]', ...
    arrayfun(@(r) r.ItoT.mAP, results), ...
    arrayfun(@(r) r.TtoI.mAP, results), ...
    [results.trainingTime]', ...
    [results.accumulatedTrainingTime]', ...
    'VariableNames', {'Round', 'TrainingPairs', 'ItoT_mAP', ...
    'TtoI_mAP', 'TrainingTimeSeconds', 'AccumulatedTrainingTimeSeconds'});
disp(summary);
