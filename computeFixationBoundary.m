function [posBoundary, negBoundary, fitInfo] = computeFixationBoundary(gazeValues, nSD, showPlot, titleStr)
% COMPUTEFIXATIONBOUNDARY  Empirically derive a symmetric gaze-position
% threshold separating "looking at source" from "central/away from
% source" fixation, based on a two-component Gaussian Mixture Model
% fit to the pooled distribution of normalized gaze positions.
%
% INPUTS:
%   gazeValues - vector of normalized gaze positions (-1 = left source,
%                +1 = right source), pooled across trials/participants
%                (e.g., for a single condition).
%   nSD        - number of standard deviations from each fitted peak
%                mean used to define the source-zone boundary. The
%                source zone extends from the peak inward (toward
%                center) by nSD * sigma; beyond that = "central."
%                Start with ~2 and visually inspect (showPlot = true)
%                to confirm the boundary lines fall in the valley
%                between the two peaks, not inside a peak. Adjust as
%                needed (commonly somewhere in the 1.5-3 range).
%   showPlot   - logical, if true displays histogram + GMM fit + boundary.
%   titleStr   - (optional) label for the plot title (e.g., condition name).
%
% OUTPUTS:
%   posBoundary - positive threshold; use as: lookR = gaze >= posBoundary
%   negBoundary - negative threshold (-posBoundary); use as: lookL = gaze <= negBoundary
%   fitInfo     - struct with diagnostic fields: muL, muR, sigmaL, sigmaR,
%                 peakMag, combSigma, gmModel

if nargin < 4, titleStr = ''; end
if nargin < 3 || isempty(showPlot), showPlot = false; end
if nargin < 2 || isempty(nSD), nSD = 2; end

gazeValues = gazeValues(:);
gazeValues = gazeValues(~isnan(gazeValues));

if numel(gazeValues) < 20
    warning('computeFixationBoundary: only %d data points; fit may be unreliable.', numel(gazeValues));
end

% --- Fit 2-component Gaussian Mixture Model ---
gmModel = fitgmdist(gazeValues, 2, ...
    'Options', statset('MaxIter', 1000), ...
    'Replicates', 5, ...
    'RegularizationValue', 1e-6);

mu    = gmModel.mu;
sigma = sqrt(squeeze(gmModel.Sigma));

% --- Identify left vs right components by mean position ---
[~, idxL] = min(mu);
[~, idxR] = max(mu);

muL = mu(idxL); sigmaL = sigma(idxL);
muR = mu(idxR); sigmaR = sigma(idxR);

% --- Symmetrize: average peak magnitude & spread across both sides ---
peakMag   = mean([abs(muL), muR]);
combSigma = mean([sigmaL, sigmaR]);

posBoundary = peakMag - nSD * combSigma;
posBoundary = max(min(posBoundary, 1), 0); % clip to sane [0,1] range
negBoundary = -posBoundary;

fitInfo = struct('muL', muL, 'muR', muR, 'sigmaL', sigmaL, 'sigmaR', sigmaR, ...
    'peakMag', peakMag, 'combSigma', combSigma, 'gmModel', gmModel);

if showPlot
    figure('Units', 'inches', 'Position', [1 1 10 7], 'Color', 'w'); hold on;

    histogram(gazeValues, 100, 'Normalization', 'pdf', ...
        'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');

    xGrid = linspace(-1.5, 1.5, 1000);
    pL = gmModel.ComponentProportion(idxL) * normpdf(xGrid, muL, sigmaL);
    pR = gmModel.ComponentProportion(idxR) * normpdf(xGrid, muR, sigmaR);
    plot(xGrid, pL, 'b-', 'LineWidth', 2.5);
    plot(xGrid, pR, 'r-', 'LineWidth', 2.5);
    plot(xGrid, pL + pR, 'k--', 'LineWidth', 1.5);

    xline(posBoundary, 'g-', 'LineWidth', 2.5);
    xline(negBoundary, 'g-', 'LineWidth', 2.5);
    xline(muL, 'b:', 'LineWidth', 1.2);
    xline(muR, 'r:', 'LineWidth', 1.2);

    set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Normalized Gaze Position (-1 = Left Source, +1 = Right Source)', 'FontSize', 18);
    ylabel('Probability Density', 'FontSize', 18);
    legend({'Data', 'Left-Source Fit', 'Right-Source Fit', 'Combined Fit', ...
        sprintf('Boundary (\\pm%.2f)', posBoundary)}, 'FontSize', 12, 'Location', 'best');
    title(sprintf('%s: Fixation Boundary (N=%d SD) = \\pm%.3f', titleStr, nSD, posBoundary), ...
        'FontSize', 18, 'Interpreter', 'tex');
    xlim([-1.5 1.5]);
    grid on;
end

end