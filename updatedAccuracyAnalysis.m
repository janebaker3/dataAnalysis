%% --- Environment Setup ---
participantIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

figurePath = '/Users/janebaker/Desktop/triangles/taskData/figures';
addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
addpath(genpath('../../analysis/'))

nStd = 3;
nHazard = 2;
nConditions = nStd * nHazard;

beta1Cap = 1.5;          % prevents quasi-separation / runaway slope fits
frameRate = 0.05;        % seconds per frame (20 Hz)
midpointSlackLow  = 0.5; % midpoint allowed up to 0.5*maxX BEFORE t=0
midpointSlackHigh = 1.5; % and up to 1.5*maxX AFTER t=0

%% --- Pass 1: extract trial-level data for everyone, no fitting yet ---
allTrialData = cell(nConditions, length(participantIDs));  % [1, framesSince, correct]
allRawAcc = NaN(nConditions, length(participantIDs), 2000);
allActualVals = NaN(nConditions, length(participantIDs), 2);
allOverallAcc = NaN(nConditions, length(participantIDs));
allNSwitches = NaN(nConditions, length(participantIDs));

for p = 1:length(participantIDs)
    pid = participantIDs(p);
    pidStr = sprintf('%03d', pid);
    load(['/Users/janebaker/Desktop/triangles/taskData/finalData/p' pidStr '.mat']);

    blocks = unique([data.blockNumber]);
    blockNum = [data.blockNumber];

    for b = 2:length(blocks)
        thisBlockTask = data(blockNum == blocks(b));
        thisBlockSource = [thisBlockTask.sourceIsLeft];
        thisBlockBelief = [thisBlockTask.keyPressed];
        stdIdx = thisBlockTask(1).stdHrzIdx;
        hIdx = thisBlockTask(1).hazardIdx;
        stdVal = thisBlockTask(1).stdHrz;
        hVal = thisBlockTask(1).H;
        condIdx = (stdIdx - 1) * nHazard + hIdx;

        allActualVals(condIdx, p, 1) = stdVal;
        allActualVals(condIdx, p, 2) = hVal;

        nTrials = length(thisBlockTask);
        trialFramesSince = zeros(nTrials, 1);
        trialIsCorrect = zeros(nTrials, 1);

        countSince = 0;
        for t = 1:nTrials
            if t == 1 || thisBlockSource(t) ~= thisBlockSource(t-1)
                countSince = 1;
            else
                countSince = countSince + 1;
            end
            trialFramesSince(t) = countSince;

            if (thisBlockSource(t) && thisBlockBelief(t) == 'F') || ...
               (~thisBlockSource(t) && thisBlockBelief(t) == 'J')
                trialIsCorrect(t) = 1;
            else
                trialIsCorrect(t) = 0;
            end
        end

        allOverallAcc(condIdx, p) = mean(trialIsCorrect);
        allNSwitches(condIdx, p) = sum(diff(thisBlockSource) ~= 0) + 1;

        maxX = max(trialFramesSince);
        for n = 1:maxX
            allRawAcc(condIdx, p, n) = mean(trialIsCorrect(trialFramesSince == n));
        end

        fitData = [ones(nTrials, 1), trialFramesSince, trialIsCorrect];
        allTrialData{condIdx, p} = [allTrialData{condIdx, p}; fitData];
    end
end

%% --- Pass 2: one-sided IQR outlier detection, per condition, on raw accuracy ---
% "Outlier" = did significantly WORSE than the majority of participants
% *within that same condition*. A whole condition being uniformly harder
% (e.g. high hazard) is expected and never flags anyone -- this only
% catches someone breaking sharply from their own condition's pack.
isOutlier = false(nConditions, length(participantIDs));

for condIdx = 1:nConditions
    vals = allOverallAcc(condIdx, :);
    valid = ~isnan(vals);
    if sum(valid) < 4
        continue % too few participants for IQR to be meaningful
    end
    Q1 = prctile(vals(valid), 25);
    Q3 = prctile(vals(valid), 75);
    IQR_ = Q3 - Q1;
    lowerFence = Q1 - 1.5 * IQR_;
    isOutlier(condIdx, :) = valid & (vals < lowerFence);
end

fprintf('\n=== Outlier exclusions (raw accuracy < Q1 - 1.5*IQR, within-condition) ===\n');
for condIdx = 1:nConditions
    flaggedP = find(isOutlier(condIdx, :));
    if ~isempty(flaggedP)
        avgStdVal = nanmean(allActualVals(condIdx, :, 1));
        avgHVal = nanmean(allActualVals(condIdx, :, 2));
        for p = flaggedP
            fprintf('  Std %.0f H %.3f: P%03d excluded from group avg (acc=%.3f)\n', ...
                avgStdVal, avgHVal, participantIDs(p), allOverallAcc(condIdx, p));
        end
    end
end

%% --- Pass 3a: individual fits (ALL participants, outliers included) ---
allParams = NaN(nConditions, length(participantIDs), 5); % beta0 beta1 lowerAsy upperAsy effAcc
allAsySEMs = NaN(nConditions, length(participantIDs), 2);

fprintf('DEBUG: fitLogisticCapped accepts %d args\n', nargin('fitLogisticCapped'));

for condIdx = 1:nConditions
    for p = 1:length(participantIDs)
        fitData = allTrialData{condIdx, p};
        if isempty(fitData), continue; end
        pidStr = sprintf('%03d', participantIDs(p));
        fprintf('Fitting P%s cond %d...\n', pidStr, condIdx);
        stdVal = nanmean(allActualVals(condIdx, p, 1));
        hVal = nanmean(allActualVals(condIdx, p, 2));

        try
            maxX = max(fitData(:,2));
            [fits, sems] = fitLogisticCapped(fitData, 'lu2', beta1Cap, maxX, midpointSlackLow, midpointSlackHigh);

            beta0 = fits(1);
            beta1 = fits(2);
            lowerAsy = fits(end-1);
            upperAsy = 1 - fits(end);
            lowerAsySEM = sems(end-1);
            upperAsySEM = sems(end);

            if beta1 > 0
                effAcc = upperAsy;
            else
                effAcc = lowerAsy;
            end

            allParams(condIdx, p, :) = [beta0, beta1, lowerAsy, upperAsy, effAcc];
            allAsySEMs(condIdx, p, :) = [lowerAsySEM, upperAsySEM];

            if abs(abs(beta1) - beta1Cap) < 1e-3
                fprintf('INFO: P%s Std %.0f H %.3f hit beta1 cap (%.2f) -- informational only, not excluded\n', ...
                    pidStr, stdVal, hVal, beta1Cap);
            end

            % --- Individual plot (generated for everyone, including outliers) ---
            maxX = size(fitData,1); %#ok<NASGU>
            maxX = max(fitData(:,2));
            f = figure('Visible', 'off');
            plotTimeSeconds = (1:maxX) * frameRate;
            predMatrix = [ones(maxX, 1), (1:maxX)'];
            smoothedAcc = logist_valLU2(fits, predMatrix);
            avgAcc = squeeze(allRawAcc(condIdx, p, 1:maxX));

            plot(plotTimeSeconds, avgAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2);
            hold on;
            plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 4);

            range = upperAsy - lowerAsy;
            midpointFrames = -beta0 / beta1;
            midpointSeconds = midpointFrames * frameRate;
            maxSlopePerSec = ((beta1 * range) / 4) / frameRate;
            yMid = (lowerAsy + upperAsy) / 2;
            if midpointSeconds > 0 && midpointSeconds < 30
                tX = [midpointSeconds - 1, midpointSeconds + 1];
                tY = maxSlopePerSec * (tX - midpointSeconds) + yMid;
                plot(tX, tY, 'b-', 'LineWidth', 3);
                plot(midpointSeconds, yMid, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
            end

            ax = gca;
            ax.FontSize = 20; ax.LineWidth = 2; ax.TickLength = [0.02 0.02];
            xlim([0 30]); ylim([0 1]); grid on;
            xlabel('Seconds Since Source Switch', 'FontSize', 24, 'FontWeight', 'bold');
            ylabel('Accuracy', 'FontSize', 24, 'FontWeight', 'bold');

            titleStr = sprintf('P%s Std %.0f H %.3f: Logistic Fit', pidStr, stdVal, hVal);
            if isOutlier(condIdx, p)
                titleStr = [titleStr ' [OUTLIER -- excluded from group avg]']; %#ok<AGROW>
                text(15, 0.05, 'RAW-ACCURACY OUTLIER (excluded from group average)', ...
                    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 14, 'HorizontalAlignment', 'center');
            end
            title(titleStr, 'FontSize', 22, 'Interpreter', 'none');

            % saveas(f, fullfile(figurePath, sprintf('P%s_cond%d.png', pidStr, condIdx)));
            close(f);
        catch ME
            fprintf('Individual fit failed for P%s Std %.3f H %.3f: %s\n', pidStr, stdVal, hVal, ME.message);
        end
    end
end

%% --- Pass 3b: pooled group fits, excluding within-condition outliers ---
groupBeta0s = NaN(1, nConditions);
groupBeta1s = NaN(1, nConditions);
groupLowerAsys = NaN(1, nConditions);
groupUpperAsys = NaN(1, nConditions);
groupEffAccs = NaN(1, nConditions);
pooledOverallAccs = NaN(1, nConditions);

for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    keepP = find(~isOutlier(condIdx, :) & ~cellfun(@isempty, allTrialData(condIdx, :)));
    if isempty(keepP), continue; end

    combinedData = vertcat(allTrialData{condIdx, keepP});
    nPooledTrials = size(combinedData, 1);

    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, keepP, :), 2));
    lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
    groupRawAcc = groupRawAcc(1:lastValidIdx);
    numFrames = length(groupRawAcc);
    plotTimeSeconds = (1:numFrames) * frameRate;

    try
        maxXPooled = max(combinedData(:,2));
        tic;
        [groupFits, groupSems] = fitLogisticCapped(combinedData, 'lu2', beta1Cap, maxXPooled, midpointSlackLow, midpointSlackHigh);
        fprintf('Pooled fit for cond %d took %.1f sec\n', condIdx, toc);

        groupBeta0 = groupFits(1);
        groupBeta1 = groupFits(2);
        groupLowerAsy = groupFits(end-1);
        groupUpperAsy = 1 - groupFits(end);

        pooledOverallAcc = mean(combinedData(:,end));

        if groupBeta1 > 0
            groupEffAcc = groupUpperAsy;
        else
            groupEffAcc = groupLowerAsy;
        end

        groupBeta0s(condIdx) = groupBeta0;
        groupBeta1s(condIdx) = groupBeta1;
        groupLowerAsys(condIdx) = groupLowerAsy;
        groupUpperAsys(condIdx) = groupUpperAsy;
        groupEffAccs(condIdx) = groupEffAcc;
        pooledOverallAccs(condIdx) = pooledOverallAcc;

        predMatrix = [ones(numFrames, 1), (1:numFrames)'];
        smoothedAcc = logist_valLU2(groupFits, predMatrix);

        figure;
        plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
        hold on;
        plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);

        groupRange = groupEffAcc - groupLowerAsy;
        midpointSec = (-groupBeta0 / groupBeta1) * frameRate;
        maxSlopePerSec = ((groupBeta1 * groupRange) / 4) / frameRate;
        yMidGroup = (groupLowerAsy + groupEffAcc) / 2;

        yline(groupEffAcc, '--k', 'LineWidth', 2.5);
        text(28, groupEffAcc - 0.05, sprintf('Asy: %.2f', groupEffAcc), ...
            'FontWeight', 'bold', 'FontSize', 22, 'HorizontalAlignment', 'right');

        if midpointSec > 0 && midpointSec < 30
            tX = [midpointSec - 1.5, midpointSec + 1.5];
            tY = maxSlopePerSec * (tX - midpointSec) + yMidGroup;
            plot(tX, tY, 'b-', 'LineWidth', 4);
            plot(midpointSec, yMidGroup, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 12);
            text(midpointSec + 0.5, yMidGroup - 0.08, sprintf('Slope: %.2f acc/sec', maxSlopePerSec), ...
                'Color', 'b', 'FontWeight', 'bold', 'FontSize', 20);
        end

        ax = gca;
        ax.FontSize = 24; ax.LineWidth = 3; ax.TickLength = [0.02 0.02];
        xlim([0 30]); ylim([0 1]); grid on;
        xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
        nExcluded = length(participantIDs) - length(keepP);
        title(sprintf('Std %.3f, Hazard %.3f (n=%d trials pooled, %d/%d participants, %d excluded as outliers)', ...
            avgStdVal, avgHVal, nPooledTrials, length(keepP), length(participantIDs), nExcluded), 'FontSize', 24);

    catch ME
        fprintf('Pooled group fit failed for Std %.3f H %.3f: %s\n', avgStdVal, avgHVal, ME.message);
    end
end

%% --- Bar charts (mean marker computed excluding outliers; bars show everyone) ---
accMatrixRaw = allOverallAcc;
accMatrixFit = squeeze(allParams(:, :, 5));

condLabels = strings(nConditions, 1);
for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    condLabels(condIdx) = sprintf('Std %.0f, H %.3f', avgStdVal, avgHVal);
end
condLabelsCell = cellstr(condLabels);
legendLabels = arrayfun(@(pid) sprintf('P%03d', pid), participantIDs, 'UniformOutput', false);
cmap = lines(length(participantIDs));

meanRawExclOutliers = NaN(1, nConditions);
meanFitExclOutliers = NaN(1, nConditions);
for condIdx = 1:nConditions
    keep = ~isOutlier(condIdx, :);
    meanRawExclOutliers(condIdx) = nanmean(accMatrixRaw(condIdx, keep));
    meanFitExclOutliers(condIdx) = nanmean(accMatrixFit(condIdx, keep));
end

f = figure('Position', [100 100 1600 700]);

subplot(1,2,1);
bh1 = bar(accMatrixRaw, 'grouped');
for i = 1:length(bh1), bh1(i).FaceColor = cmap(i,:); end
hold on;
plot(1:nConditions, meanRawExclOutliers, 'ko', 'MarkerFaceColor','k','MarkerSize',10,'LineWidth',2);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ylim([0 1]); grid on;
ylabel('Raw Overall Accuracy', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Condition (Std / Hazard)', 'FontSize', 18, 'FontWeight', 'bold');
title('Raw Accuracy (black dot = mean excl. outliers)', 'FontSize', 18);

subplot(1,2,2);
bh2 = bar(accMatrixFit, 'grouped');
for i = 1:length(bh2), bh2(i).FaceColor = cmap(i,:); end
hold on;
plot(1:nConditions, meanFitExclOutliers, 'ko', 'MarkerFaceColor','k','MarkerSize',10,'LineWidth',2);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ylim([0 1]); grid on;
ylabel('Effective Accuracy (logistic asymptote)', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Condition (Std / Hazard)', 'FontSize', 18, 'FontWeight', 'bold');
title('Logistic Fit Asymptote (black dot = mean excl. outliers)', 'FontSize', 18);
legend(bh2, legendLabels, 'Location', 'eastoutside', 'FontSize', 11);

%% --- Heatmaps (full data shown; outlier cells marked, not blanked) ---
f2 = figure('Position', [100 100 1600 700]);

subplot(1,2,1);
imagesc(accMatrixRaw');
colormap(parula); cb = colorbar; cb.Label.String = 'Raw Accuracy'; clim([0 1]);
hold on;
[rOut, cOut] = find(isOutlier'); % note transpose to match plotted orientation
plot(cOut, rOut, 'rx', 'MarkerSize', 14, 'LineWidth', 3);
ax = gca; ax.FontSize = 12; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ax.YTick = 1:length(participantIDs); ax.YTickLabel = legendLabels;
xlabel('Condition', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Participant', 'FontSize', 16, 'FontWeight', 'bold');
title('Raw Overall Accuracy (red X = outlier, excluded from group avg)', 'FontSize', 16);

subplot(1,2,2);
imagesc(accMatrixFit');
colormap(parula); cb = colorbar; cb.Label.String = 'Effective Accuracy'; clim([0 1]);
hold on;
plot(cOut, rOut, 'rx', 'MarkerSize', 14, 'LineWidth', 3);
ax = gca; ax.FontSize = 12; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ax.YTick = 1:length(participantIDs); ax.YTickLabel = legendLabels;
xlabel('Condition', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Participant', 'FontSize', 16, 'FontWeight', 'bold');
title('Logistic Fit Asymptote (red X = outlier, excluded from group avg)', 'FontSize', 16);

%% --- Local Functions ---
function [fits_, sems_] = fitLogisticCapped(data, lumode, beta1Cap, maxX, midpointSlackLow, midpointSlackHigh)

[fun, data_, inits] = logist_setup(data, lumode);

inits(2,2) = -beta1Cap;
inits(2,3) =  beta1Cap;
inits(2,1) = min(max(inits(2,1), -beta1Cap), beta1Cap);

lowerMid = -midpointSlackLow * maxX;
upperMid =  midpointSlackHigh * maxX;

% Ensure the STARTING POINT is itself feasible w.r.t. the midpoint
% constraint, not just the final solution. logist_setup's OLS-based
% initial beta0 guess has no awareness of our added constraints and can
% imply a wildly infeasible starting midpoint in sparse/low-hazard
% conditions, which can cause 'active-set' to stall hunting for
% feasibility rather than actually optimizing.
initBeta1 = inits(2,1);
if initBeta1 == 0
    initBeta1 = 1e-9;
end
initMidpoint = -inits(1,1) / initBeta1;
initMidpointClipped = min(max(initMidpoint, lowerMid), upperMid);
inits(1,1) = -initMidpointClipped * initBeta1;

nonlconFun = @(x, varargin) midpointNonlcon(x, lowerMid, upperMid);

opts = optimset('LargeScale', 'off', 'Algorithm', 'active-set', ...
    'Display', 'off', 'Diagnostics', 'off', ...
    'MaxIter', 200, 'MaxFunEvals', 5000);

[fits_, ~, exitflag, ~, ~, ~, H] = fmincon('logist_err', inits(:,1), [], [], [], [], ...
    inits(:,2), inits(:,3), nonlconFun, opts, fun, data_);

if exitflag <= 0
    error('fitLogisticCapped:noConverge', 'fmincon did not converge (exitflag=%d)', exitflag);
end

sems_ = sqrt(diag(-((-H)^(-1))));
end


function [c, ceq] = midpointNonlcon(x, lowerMid, upperMid)
beta0 = x(1);
beta1 = x(2);
if beta1 == 0
    beta1 = 1e-9; % avoid literal div-by-zero; a true zero-slope point
                   % would make midpoint undefined/infinite anyway, which
                   % already violates the constraint below
end
midpoint = -beta0 / beta1;
c = [lowerMid - midpoint; midpoint - upperMid]; % both must be <= 0
ceq = [];
end
















