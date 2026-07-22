function [results, models, options] = evaluate_OSCL( ...
    XChunk, YChunk, ZChunk, XQuery, YQuery, ZQuery, options)
%EVALUATE_OSCL Train and evaluate OSCL as paired data chunks arrive.
%
% XChunk{t}, YChunk{t}, and ZChunk{t} contain rows of paired image
% features, text features, and multi-label annotations at round t.
% XQuery, YQuery, and ZQuery contain the corresponding query set.

if nargin < 7
    options = struct();
end
options = apply_defaults(options, numel(XChunk));
validate_data(XChunk, YChunk, ZChunk, XQuery, YQuery, ZQuery, options);

numRounds = options.numRounds;
numSamples = sum(cellfun(@(X) size(X, 1), XChunk(1:numRounds)));
databaseX = zeros(numSamples, size(XQuery, 2), 'like', XChunk{1});
databaseY = zeros(numSamples, size(YQuery, 2), 'like', YChunk{1});
databaseZ = zeros(numSamples, size(ZQuery, 2), 'like', ZChunk{1});

direction = struct('mAP', [], 'k', [], 'precisionAtK', [], ...
    'precision', [], 'recall', [], 'numValidQueries', []);
result = struct('round', [], 'numTrainingSamples', [], ...
    'trainingTime', [], 'accumulatedTrainingTime', [], ...
    'encodingTime', [], 'retrievalTime', [], ...
    'ItoT', direction, 'TtoI', direction);
results = repmat(result, numRounds, 1);
models = repmat(struct('W', [], 'G', [], 'meanX', [], 'meanY', []), ...
    numRounds, 1);

state = [];
lastRow = 0;
accumulatedTrainingTime = 0;

for t = 1:numRounds
    Xnew = XChunk{t};
    Ynew = YChunk{t};
    Znew = ZChunk{t};

    if options.verbose
        fprintf('OSCL round %d/%d: %d new pairs\n', ...
            t, numRounds, size(Xnew, 1));
    end

    timer = tic;
    [state, W, G] = train_OSCL(Xnew', Ynew', Znew', state, options);
    results(t).trainingTime = toc(timer);
    accumulatedTrainingTime = accumulatedTrainingTime + results(t).trainingTime;
    results(t).accumulatedTrainingTime = accumulatedTrainingTime;

    rows = lastRow + (1:size(Xnew, 1));
    databaseX(rows, :) = Xnew;
    databaseY(rows, :) = Ynew;
    databaseZ(rows, :) = Znew;
    lastRow = rows(end);

    timer = tic;
    XQueryLow = project(XQuery, state.meanX, W);
    YQueryLow = project(YQuery, state.meanY, G);
    XDatabaseLow = project(databaseX(1:lastRow, :), state.meanX, W);
    YDatabaseLow = project(databaseY(1:lastRow, :), state.meanY, G);
    results(t).encodingTime = toc(timer);

    timer = tic;
    rankingItoT = cosine_ranking(XQueryLow, YDatabaseLow);
    rankingTtoI = cosine_ranking(YQueryLow, XDatabaseLow);
    labels = databaseZ(1:lastRow, :);
    results(t).ItoT = compute_retrieval_metrics( ...
        rankingItoT, labels, ZQuery, options);
    results(t).TtoI = compute_retrieval_metrics( ...
        rankingTtoI, labels, ZQuery, options);
    results(t).retrievalTime = toc(timer);

    results(t).round = t;
    results(t).numTrainingSamples = lastRow;
    models(t) = struct('W', W, 'G', G, ...
        'meanX', state.meanX, 'meanY', state.meanY);

    if options.verbose
        fprintf(['  I->T mAP %.4f | T->I mAP %.4f | ' ...
            'training %.2f s (accumulated %.2f s)\n\n'], ...
            results(t).ItoT.mAP, results(t).TtoI.mAP, ...
            results(t).trainingTime, results(t).accumulatedTrainingTime);
    end
end
end

function lowDimensionalData = project(data, dataMean, projection)
lowDimensionalData = bsxfun(@minus, data, dataMean') * projection;
end

function ranking = cosine_ranking(query, database)
query = normalize_rows(query);
database = normalize_rows(database);
[~, ranking] = sort(query * database', 2, 'descend');
end

function data = normalize_rows(data)
rowNorms = sqrt(sum(data.^2, 2));
rowNorms(rowNorms < eps(class(rowNorms))) = 1;
data = bsxfun(@rdivide, data, rowNorms);
end

function options = apply_defaults(options, numAvailableRounds)
defaults = struct('ell', 10, 'lambdaX', 1e-1, 'lambdaY', 1e-1, ...
    'maxK', 2000, 'numRounds', numAvailableRounds, ...
    'computePrecisionRecall', true, 'verbose', true, ...
    'rankTolerance', 1e-6);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i}) || isempty(options.(names{i}))
        options.(names{i}) = defaults.(names{i});
    end
end
end

function validate_data(XChunk, YChunk, ZChunk, XQuery, YQuery, ZQuery, options)
if ~iscell(XChunk) || ~iscell(YChunk) || ~iscell(ZChunk) || ...
        numel(XChunk) ~= numel(YChunk) || numel(XChunk) ~= numel(ZChunk)
    error('OSCL:InvalidChunks', ...
        'XChunk, YChunk, and ZChunk must be equally sized cell arrays.');
end
if options.numRounds < 1 || options.numRounds > numel(XChunk) || ...
        options.numRounds ~= floor(options.numRounds)
    error('OSCL:InvalidRounds', 'options.numRounds is invalid.');
end
if options.ell < 1 || options.ell > min(size(XQuery, 2), size(YQuery, 2))
    error('OSCL:InvalidDimension', 'options.ell is invalid.');
end
if size(XQuery, 1) ~= size(YQuery, 1) || size(XQuery, 1) ~= size(ZQuery, 1)
    error('OSCL:InvalidQueries', 'Query matrices must have equal row counts.');
end

for t = 1:options.numRounds
    n = size(XChunk{t}, 1);
    valid = size(YChunk{t}, 1) == n && size(ZChunk{t}, 1) == n && ...
        size(XChunk{t}, 2) == size(XQuery, 2) && ...
        size(YChunk{t}, 2) == size(YQuery, 2) && ...
        size(ZChunk{t}, 2) == size(ZQuery, 2);
    if ~valid
        error('OSCL:InvalidChunk', ...
            'Feature or label dimensions do not agree in chunk %d.', t);
    end
end
end
