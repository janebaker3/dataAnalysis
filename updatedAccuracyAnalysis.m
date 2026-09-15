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

allBeta1SEMs = NaN(nConditions, length(participantIDs));
allEffAccSEMs = NaN(nConditions, length(participantIDs));

for p = 1:length(participantIDs)
    pid = participantIDs(p);
    pidStr = sprintf('%03d', pid);
    load(['/Users/janebaker/Desktop/triangles/taskData/finalData/p' pidStr '.mat']);

    blocks = unique([data.blockNumber]);
    blockNum = [data.blockNumber];

    minTrialsPerBin = 4;

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

        rawMaxX = max(trialFramesSince);
        counts = accumarray(trialFramesSince, 1, [rawMaxX, 1]);
        if (p == 1 && condIdx == 6) || (p == 10 && condIdx == 6)
            disp('COUNT: ')
            disp(counts)
        end
        reliableMaxX = find(counts >= minTrialsPerBin, 1, 'last');
        if isempty(reliableMaxX)
            reliableMaxX = rawMaxX; % degenerate edge case, shouldn't normally happen
        end

        if reliableMaxX < rawMaxX
            fprintf('INFO: P%s Std %.0f H %.3f -- trimmed tail from %d to %d frames (min %d trials/bin)\n', ...
                pidStr, stdVal, hVal, rawMaxX, reliableMaxX, minTrialsPerBin);
        end

        keepTrials = trialFramesSince <= reliableMaxX;
        trialFramesSince = trialFramesSince(keepTrials);
        trialIsCorrect = trialIsCorrect(keepTrials);
        nTrialsKept = sum(keepTrials);

        allOverallAcc(condIdx, p) = mean(trialIsCorrect);
        allNSwitches(condIdx, p) = sum(diff(thisBlockSource) ~= 0) + 1;
        if (p == 1 && condIdx == 6) || (p == 10 && condIdx == 6)
            disp('SOURCE SWITCHES: ')
            disp(allNSwitches(condIdx, p))
        end

        for n = 1:reliableMaxX
            allRawAcc(condIdx, p, n) = mean(trialIsCorrect(trialFramesSince == n));
        end

        fitData = [ones(nTrialsKept, 1), trialFramesSince, trialIsCorrect];
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
                effAccSEM = upperAsySEM;
            else
                effAcc = lowerAsy;
                effAccSEM = lowerAsySEM;
            end

            allParams(condIdx, p, :) = [beta0, beta1, lowerAsy, upperAsy, effAcc];
            allAsySEMs(condIdx, p, :) = [lowerAsySEM, upperAsySEM];
            allBeta1SEMs(condIdx, p) = sems(2);
            allEffAccSEMs(condIdx, p) = effAccSEM;

            allParams(condIdx, p, :) = [beta0, beta1, lowerAsy, upperAsy, effAcc];
            allAsySEMs(condIdx, p, :) = [lowerAsySEM, upperAsySEM];

            if abs(abs(beta1) - beta1Cap) < 1e-3
                fprintf('INFO: P%s Std %.0f H %.3f hit beta1 cap (%.2f) -- informational only, not excluded\n', ...
                    pidStr, stdVal, hVal, beta1Cap);
            end

            % --- Individual plot (generated for everyone, including outliers) ---
            maxX = size(fitData,1); %#ok<NASGU>
            maxX = max(fitData(:,2));
            f = figure('Visible', 'on');
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

%% --- Weighted group summary bars (individual fits, outlier + invalid-SEM excluded) ---
weightedMeanAcc = NaN(1, nConditions);
weightedSEMAcc  = NaN(1, nConditions);
weightedMeanBeta1 = NaN(1, nConditions);
weightedSEMBeta1  = NaN(1, nConditions);
nUsedAcc = NaN(1, nConditions);
nUsedBeta1 = NaN(1, nConditions);

for condIdx = 1:nConditions
    keep = ~isOutlier(condIdx, :);

    accVals = squeeze(allParams(condIdx, :, 5));
    accSEMs = allEffAccSEMs(condIdx, :);
    validAcc = keep & isfinite(accVals) & isfinite(accSEMs) & accSEMs > 0;

    if any(validAcc)
        w = 1 ./ (accSEMs(validAcc) .^ 2);
        weightedMeanAcc(condIdx) = sum(w .* accVals(validAcc)) / sum(w);
        weightedSEMAcc(condIdx) = sqrt(1 / sum(w));
        nUsedAcc(condIdx) = sum(validAcc);
    end

    betaVals = squeeze(allParams(condIdx, :, 2));
    betaSEMs = allBeta1SEMs(condIdx, :);
    validBeta = keep & isfinite(betaVals) & isfinite(betaSEMs) & betaSEMs > 0;

    if any(validBeta)
        w = 1 ./ (betaSEMs(validBeta) .^ 2);
        weightedMeanBeta1(condIdx) = sum(w .* betaVals(validBeta)) / sum(w);
        weightedSEMBeta1(condIdx) = sqrt(1 / sum(w));
        nUsedBeta1(condIdx) = sum(validBeta);
    end
end

fprintf('\n=== Weighted group summary (excludes outliers + non-finite/zero SEM fits) ===\n');
for condIdx = 1:nConditions
    fprintf('  %s: acc n=%d/%d, beta1 n=%d/%d\n', condLabelsCell{condIdx}, ...
        nUsedAcc(condIdx), length(participantIDs), nUsedBeta1(condIdx), length(participantIDs));
end

f3 = figure('Position', [100 100 1400 600]);

subplot(1,2,1);
bar(weightedMeanAcc, 'FaceColor', [0.4 0.6 0.8]);
hold on;
errorbar(1:nConditions, weightedMeanAcc, weightedSEMAcc, 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 12);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ylim([0 1]); grid on;
ylabel('Effective Accuracy (weighted mean)', 'FontSize', 16, 'FontWeight', 'bold');
title('Inverse-variance-weighted mean asymptote across participants', 'FontSize', 16);

subplot(1,2,2);
bar(weightedMeanBeta1, 'FaceColor', [0.8 0.5 0.4]);
hold on;
errorbar(1:nConditions, weightedMeanBeta1, weightedSEMBeta1, 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 12);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
grid on;
ylabel('\beta_1 (weighted mean)', 'FontSize', 16, 'FontWeight', 'bold');
title('Inverse-variance-weighted mean slope parameter', 'FontSize', 16);

%% --- Pass 3c: Hierarchical (mixed-effects) group fits, per condition ---
% Random effects on beta0/beta1 (midpoint/slope) only; asymptotes fixed.
% Fit on per-participant binned accuracy (allRawAcc), NOT pooled raw trials --
% avoids the trial-count-weighting artifact from Pass 3b, and matches the
% Gaussian-residual assumption of nlmefit far better than 0/1 trial outcomes.
% Fully unconstrained: no beta1 cap, no midpoint bound. Cross-subject shrinkage
% is the mechanism preventing runaway individual estimates, not hard bounds.

xCapSeconds = 15;
xCapFrames = round(xCapSeconds / frameRate);

nlmeFixed = NaN(nConditions, 4);      % beta0 beta1 lowerAsy upperLapse (fixed effects / population avg)
nlmeFixedSE = NaN(nConditions, 4);    % SEs on fixed effects
nlmeEffAcc = NaN(1, nConditions);
nlmeRandCov = cell(1, nConditions);   % PSI: random-effect covariance (beta0,beta1)
nlmeConverged = false(1, nConditions);

for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    keepP = find(~isOutlier(condIdx, :) & ~cellfun(@isempty, allTrialData(condIdx, :)));
    if length(keepP) < 3
        fprintf('Skipping nlme fit for cond %d: too few non-outlier participants (%d)\n', condIdx, length(keepP));
        continue;
    end

    % --- Build binned-proportion dataset across kept participants ---
    Xall = []; Yall = []; Gall = [];
    for pp = keepP
        accCurve = squeeze(allRawAcc(condIdx, pp, :));
        validBins = find(~isnan(accCurve));
        if isempty(validBins), continue; end
        Xall = [Xall; validBins]; %#ok<AGROW>
        Yall = [Yall; accCurve(validBins)]; %#ok<AGROW>
        Gall = [Gall; repmat(pp, length(validBins), 1)]; %#ok<AGROW>
    end

    if isempty(Xall)
        fprintf('Skipping nlme fit for cond %d: no binned data\n', condIdx);
        continue;
    end

    % --- Initial fixed-effect guess from mean of individual fits (Pass 3a) ---
    initBeta0 = nanmean(allParams(condIdx, keepP, 1));
    initBeta1 = nanmean(allParams(condIdx, keepP, 2));
    initLowerAsy = nanmean(allParams(condIdx, keepP, 3));
    initUpperAsy = nanmean(allParams(condIdx, keepP, 4));
    initUpperLapse = 1 - initUpperAsy;
    if any(isnan([initBeta0 initBeta1 initLowerAsy initUpperLapse]))
        fprintf('Skipping nlme fit for cond %d: NaN initial values\n', condIdx);
        continue;
    end
    beta0Init = [initBeta0; initBeta1; initLowerAsy; initUpperLapse];

    opts = statset('MaxIter', 200, 'TolFun', 1e-6, 'TolX', 1e-6); 

    lastwarn('');
    try
        [fixedFx, PSI, stats, ~] = nlmefit(Xall, Yall, Gall, [], @nlmeLogisticLU, beta0Init, ...
            'REParamsSelect', [1 2 3 4], 'Options', opts);

        [warnMsg, ~] = lastwarn();
        if ~isempty(warnMsg)
            fprintf('WARNING: cond %d nlme fit raised: %s\n', condIdx, warnMsg);
            nlmeConverged(condIdx) = false;
        else
            nlmeConverged(condIdx) = true;
        end

        nlmeFixed(condIdx, :) = fixedFx';
        seVec = sqrt(diag(stats.covb));
        nlmeFixedSE(condIdx, :) = seVec';
        nlmeRandCov{condIdx} = PSI;

        groupBeta0 = fixedFx(1);
        groupBeta1 = fixedFx(2);
        groupLowerAsy = fixedFx(3);
        groupUpperAsy = 1 - fixedFx(4);

        if groupBeta1 > 0
            groupEffAcc = groupUpperAsy;
        else
            groupEffAcc = groupLowerAsy;
        end
        nlmeEffAcc(condIdx) = groupEffAcc;

        fprintf('Cond %d (Std %.0f H %.3f): nlme fixed effects beta0=%.3f beta1=%.3f lowerAsy=%.3f upperAsy=%.3f\n', ...
            condIdx, avgStdVal, avgHVal, groupBeta0, groupBeta1, groupLowerAsy, groupUpperAsy);

        % --- Plot: population-average (fixed-effect) curve vs. across-subject mean raw accuracy ---
        groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, keepP, :), 2));
        lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
        maxFrame = min(lastValidIdx, xCapFrames);
        groupRawAcc = groupRawAcc(1:maxFrame);
        plotTimeSeconds = (1:maxFrame) * frameRate;

        predPHI = repmat(fixedFx', maxFrame, 1);
        smoothedAcc = nlmeLogisticLU(predPHI, (1:maxFrame)');

        figure;
        plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
        hold on;
        plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);

        midpointSec = (-groupBeta0 / groupBeta1) * frameRate;
        groupRange = groupEffAcc - groupLowerAsy;
        maxSlopePerSec = ((groupBeta1 * groupRange) / 4) / frameRate;
        yMidGroup = (groupLowerAsy + groupEffAcc) / 2;

        yline(groupEffAcc, '--k', 'LineWidth', 2.5);
        text(xCapSeconds - 1, groupEffAcc - 0.05, sprintf('Asy: %.2f', groupEffAcc), ...
            'FontWeight', 'bold', 'FontSize', 18, 'HorizontalAlignment', 'right');

        if midpointSec > 0 && midpointSec < xCapSeconds
            tX = [max(0, midpointSec - 1), min(xCapSeconds, midpointSec + 1)];
            tY = maxSlopePerSec * (tX - midpointSec) + yMidGroup;
            plot(tX, tY, 'b-', 'LineWidth', 4);
            plot(midpointSec, yMidGroup, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 12);
        end

        ax = gca;
        ax.FontSize = 20; ax.LineWidth = 2.5; ax.TickLength = [0.02 0.02];
        xlim([0 xCapSeconds]); ylim([0 1]); grid on;
        xlabel('Seconds Since Source Switch', 'FontSize', 24, 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontSize', 24, 'FontWeight', 'bold');
        title(sprintf('Std %.0f, H %.3f -- Mixed-effects group fit (n=%d participants, %s)', ...
            avgStdVal, avgHVal, length(keepP), ...
            ternary(nlmeConverged(condIdx), 'converged', 'CHECK CONVERGENCE')), ...
            'FontSize', 18, 'Interpreter', 'none');

    catch ME
        fprintf('nlme fit failed for cond %d (Std %.3f H %.3f): %s\n', condIdx, avgStdVal, avgHVal, ME.message);
    end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


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

function Y = nlmeLogisticLU(PHI, X)
% PHI columns: [beta0, beta1, lowerAsy, upperLapse], one row per observation
% (nlmefit broadcasts each subject's group-level PHI across their rows)
beta0 = PHI(:,1);
beta1 = PHI(:,2);
lowerAsy = PHI(:,3);
upperLapse = PHI(:,4);
upperAsy = 1 - upperLapse;

linpred = beta0 + beta1 .* X(:,1);
logistic = 1 ./ (1 + exp(-linpred));
Y = lowerAsy + (upperAsy - lowerAsy) .* logistic;
end