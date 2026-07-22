function [W, G] = solve_OSCL_projection(state, options)
%SOLVE_OSCL_PROJECTION Compute the OSCL image/text projections W and G.
%
% The variables B, Lambda, W, and G follow the manuscript notation.

U = state.U;
V = state.V;
sigmaX = state.sigmaX;
P = state.P;
Q = state.Q;
psiY = state.psiY;

numSamples = size(state.Z, 2);
S = state.Z' * state.Z + eye(numSamples, 'like', state.Z);
B = V' * S * Q * diag(psiY ./ sqrt(psiY.^2 + options.lambdaY));

Lambda1 = sigmaX ./ (sigmaX.^2 + options.lambdaX);
Lambda = sqrt(Lambda1) ./ sqrt(sigmaX);
LambdaTilde = Lambda1 ./ Lambda;
[Ub, SigmaB] = svd(diag(LambdaTilde) * B, 'econ');
sigmaB = diag(SigmaB);

tolerance = max(size(SigmaB)) * eps(max(sigmaB));
ell = min(options.ell, nnz(sigmaB > tolerance));
if ell < 1
    error('OSCL:DegenerateProjection', ...
        'No nonzero shared projection direction was found.');
elseif ell < options.ell && options.verbose
    warning('OSCL:ReducedDimension', ...
        'Reducing ell from %d to %d for this round.', options.ell, ell);
end

Ub = Ub(:, 1:ell);
W = normalize_columns(U * diag(Lambda) * Ub);
G = P * diag(psiY ./ (psiY.^2 + options.lambdaY)) * Q' * S * V * ...
    diag(sigmaX .* Lambda) * Ub;
G = normalize_columns(bsxfun(@rdivide, G, sigmaB(1:ell)'));
end

function matrix = normalize_columns(matrix)
columnNorms = sqrt(sum(matrix.^2, 1));
columnNorms(columnNorms < eps(class(columnNorms))) = 1;
matrix = bsxfun(@rdivide, matrix, columnNorms);
end
