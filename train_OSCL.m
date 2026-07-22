function [state, W, G] = train_OSCL(Xnew, Ynew, Znew, state, options)
%TRAIN_OSCL Initialize or update Online Semantic Correlation Learning.
%
% Inputs follow the manuscript orientation: columns are paired samples,
% Xnew and Ynew are image/text features, and Znew contains annotations.
% The returned state is passed unchanged to the next online round.

Znew = normalize_columns(Znew);

%% 1. Initialize or update the compact image/text SVDs
if isempty(state) || isempty(fieldnames(state))
    state.meanX = mean(Xnew, 2);
    state.meanY = mean(Ynew, 2);
    [state.U, state.sigmaX, state.V] = compact_svd( ...
        bsxfun(@minus, Xnew, state.meanX), options.rankTolerance);
    [state.P, state.psiY, state.Q] = compact_svd( ...
        bsxfun(@minus, Ynew, state.meanY), options.rankTolerance);
    state.Z = Znew;
else
    [state.U, state.sigmaX, state.V, state.meanX] = update_OSCL_svd( ...
        Xnew, state.U, state.sigmaX, state.V, state.meanX, ...
        options.rankTolerance);
    [state.P, state.psiY, state.Q, state.meanY] = update_OSCL_svd( ...
        Ynew, state.P, state.psiY, state.Q, state.meanY, ...
        options.rankTolerance);
    state.Z = [state.Z, Znew];
end

%% 2. Compute the OSCL projections W and G
[W, G] = solve_OSCL_projection(state, options);
end

function [U, singularValues, V] = compact_svd(data, rankTolerance)
[U, Sigma, V] = svd(data, 'econ');
singularValues = diag(Sigma);
retain = retain_singular_values(singularValues, rankTolerance);
U = U(:, retain);
singularValues = singularValues(retain);
V = V(:, retain);
end

function retain = retain_singular_values(singularValues, rankTolerance)
energy = sum(singularValues.^2);
if energy <= eps(class(energy))
    error('OSCL:DegenerateFeatures', ...
        'A centered feature matrix has zero numerical variance.');
end
retain = singularValues.^2 >= energy * rankTolerance;
if ~any(retain)
    retain(1) = true;
end
end

function matrix = normalize_columns(matrix)
columnNorms = sqrt(sum(matrix.^2, 1));
columnNorms(columnNorms < eps(class(columnNorms))) = 1;
matrix = bsxfun(@rdivide, matrix, columnNorms);
end
