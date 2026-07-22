function [Unew, singularValuesNew, Vnew, totalMean] = update_OSCL_svd( ...
    Xnew, U, singularValues, V, oldMean, rankTolerance)
%UPDATE_OSCL_SVD Update a compact SVD after a centered chunk arrives.
%
% This is the mean-corrected incremental SVD step described in the OSCL
% manuscript. It updates the retained factors without reconstructing the
% historical feature matrix.

numOld = size(V, 1);
numNew = size(Xnew, 2);
numTotal = numOld + numNew;
newMean = mean(Xnew, 2);
totalMean = (numOld * oldMean + numNew * newMean) / numTotal;

centeredNew = bsxfun(@minus, Xnew, newMean);
meanCorrection = sqrt(numOld * numNew / numTotal) * (oldMean - newMean);
augmentedNew = [centeredNew, meanCorrection];

projected = U' * augmentedNew;
residual = augmentedNew - U * projected;
[E, ~] = qr(residual, 0);

r = numel(singularValues);
smallMatrix = [diag(singularValues), projected; ...
    zeros(size(E, 2), r), E' * residual];
rightTransform = [V', zeros(r, numNew + 1); ...
    zeros(numNew + 1, numOld), eye(numNew + 1)];

[Usmall, SigmaSmall, Vsmall] = svd(smallMatrix, 'econ');
allSingularValues = diag(SigmaSmall);
energy = sum(allSingularValues.^2);
if energy <= eps(class(energy))
    error('OSCL:DegenerateFeatures', ...
        'The updated centered feature matrix has zero numerical variance.');
end
retain = allSingularValues.^2 >= energy * rankTolerance;
if ~any(retain)
    retain(1) = true;
end

Unew = [U, E] * Usmall(:, retain);
singularValuesNew = allSingularValues(retain);
Vnew = rightTransform' * Vsmall(:, retain);

% The last row corresponds to the mean-correction pseudo-sample.
Vnew = Vnew(1:end-1, :);
end
