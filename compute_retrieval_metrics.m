function metrics = compute_retrieval_metrics( ...
    ranking, databaseZ, queryZ, options)
%COMPUTE_RETRIEVAL_METRICS Evaluate multi-label retrieval rankings.
%
% A database item is relevant if it shares at least one positive label
% with the query. Queries without relevant items are excluded.

numQueries = size(queryZ, 1);
numDatabaseItems = size(databaseZ, 1);
maxK = min(options.maxK, numDatabaseItems);
ranks = 1:numDatabaseItems;
k = 1:maxK;

sumAP = 0;
sumPrecisionAtK = zeros(1, maxK);
sumPrecision = zeros(1, numDatabaseItems);
sumRecall = zeros(1, numDatabaseItems);
numValidQueries = 0;

for q = 1:numQueries
    relevant = databaseZ * queryZ(q, :)' > 0;
    numRelevant = nnz(relevant);
    if numRelevant == 0
        continue;
    end

    rankedRelevant = reshape(relevant(ranking(q, :)), 1, []);
    cumulativeRelevant = cumsum(rankedRelevant);
    relevantRanks = find(rankedRelevant);

    sumAP = sumAP + ...
        sum(cumulativeRelevant(relevantRanks) ./ relevantRanks) / numRelevant;
    sumPrecisionAtK = sumPrecisionAtK + cumulativeRelevant(1:maxK) ./ k;
    if options.computePrecisionRecall
        sumPrecision = sumPrecision + cumulativeRelevant ./ ranks;
        sumRecall = sumRecall + cumulativeRelevant / numRelevant;
    end
    numValidQueries = numValidQueries + 1;
end

if numValidQueries == 0
    error('OSCL:NoRelevantQueries', ...
        'No query has a relevant item in the current database.');
end

metrics = struct('mAP', sumAP / numValidQueries, 'k', k, ...
    'precisionAtK', sumPrecisionAtK / numValidQueries, ...
    'precision', [], 'recall', [], 'numValidQueries', numValidQueries);
if options.computePrecisionRecall
    metrics.precision = sumPrecision / numValidQueries;
    metrics.recall = sumRecall / numValidQueries;
end
end
