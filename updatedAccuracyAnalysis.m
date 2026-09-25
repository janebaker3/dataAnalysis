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

accDeltaThreshold = 0.03;    % max acceptable |delta effective accuracy| to still rescue an outlier
beta1DeltaThreshold = 0.02;  % max acceptable |delta beta1| to still rescue an outlier

%% --- Strategy Assignment ---
stratAssignment = cell(nConditions, 1);

% ---------------- enter manually ----------------------
% Condition 1
stratAssignment{1} = [1; 1; 2; 3; 3; 3; NaN; 3; 1; 2];

% Condition 2
stratAssignment{2} = [2; 2; 2; NaN; 3; 3; NaN; 2; 3; 2];

% Condition 3
stratAssignment{3} = [2; 3; 1; 3; 3; 3; NaN; 3; 3; 2];

% Condition 4
stratAssignment{4} = [2; 2; 2; 3; 3; 2; NaN; 2; 3; 2];

% Condition 5
stratAssignment{5} = [1; 1; 1; NaN; 3; 3; NaN; 3; 3; 2];

% Condition 6
stratAssignment{6} = [2; 3; 2; 3; 3; 3; NaN; 3; 1; 2];
% ------------------------------------------------------

% %% --- Global Strategy Assignment ---

% ---------------- enter manually ----------------------
% % Condition 1
% stratAssignment{1} = [1; 2; 3; 1; 1; 1; NaN; 2; 1; 2];
% 
% % Condition 2
% stratAssignment{2} = [2; 3; 2; NaN; 1; 2; NaN; 3; 2; 1];
% 
% % Condition 3
% stratAssignment{3} = [2; 3; 2; 1; 2; 3; NaN; 3; 2; 1];
% 
% % Condition 4
% stratAssignment{4} = [1; 3; 3; 1; 3; 3; NaN; 2; 1; 1];
% 
% % Condition 5
% stratAssignment{5} = [2; 3; 3; NaN; 2; 3; NaN; 3; 3; 1];
% 
% % Condition 6
% stratAssignment{6} = [1; 3; 2; 1; 2; 3; NaN; 2; 2; 1];
% ------------------------------------------------------

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
% NOTE: plotting for these fits now happens in Pass 5, after outlier
% impact has been assessed and a final inclusion decision made. This
% section only computes/stores the "all outliers excluded" baseline fit,
% which Pass 4 uses as its reference and Pass 5 reuses whenever no
% outlier ends up being rescued for that condition.
groupBeta0s = NaN(1, nConditions);
groupBeta1s = NaN(1, nConditions);
groupLowerAsys = NaN(1, nConditions);
groupUpperAsys = NaN(1, nConditions);
groupEffAccs = NaN(1, nConditions);
pooledOverallAccs = NaN(1, nConditions);
groupFitsFull = cell(1, nConditions);   % full fit param vector, for reuse
groupKeepPExcl = cell(1, nConditions);  % participant indices used (outliers excluded)

for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    keepP = find(~isOutlier(condIdx, :) & ~cellfun(@isempty, allTrialData(condIdx, :)));
    if isempty(keepP), continue; end
    groupKeepPExcl{condIdx} = keepP;

    combinedData = vertcat(allTrialData{condIdx, keepP});

    try
        maxXPooled = max(combinedData(:,2));
        tic;
        [groupFits, ~] = fitLogisticCapped(combinedData, 'lu2', beta1Cap, maxXPooled, midpointSlackLow, midpointSlackHigh);
        fprintf('Pooled fit for cond %d took %.1f sec\n', condIdx, toc);

        groupFitsFull{condIdx} = groupFits;

        groupBeta0 = groupFits(1);
        groupBeta1 = groupFits(2);
        groupLowerAsy = groupFits(end-1);
        groupUpperAsy = 1 - groupFits(end);

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
        pooledOverallAccs(condIdx) = mean(combinedData(:,end));

    catch ME
        fprintf('Pooled group fit failed for Std %.3f H %.3f: %s\n', avgStdVal, avgHVal, ME.message);
    end
end

%% --- Pass 4: outlier impact -- compare pooled fit with/without each flagged outlier ---
% For every condition with at least one flagged outlier, and for EACH such
% outlier individually, compare:
%   "Excluding" = pooled fit over all non-outlier participants (from Pass 3b)
%   "Including" = pooled fit over all non-outlier participants + this ONE outlier added back
%                 (other outliers in the same condition, if any, remain excluded)
% This isolates the distortion caused by each individual outlier.

fprintf('\n=== Outlier impact on pooled logistic fits ===\n');

maxInstances = nConditions * length(participantIDs);
impactCondIdx    = NaN(maxInstances, 1);
impactPIdx       = NaN(maxInstances, 1);
impactEffAccExcl = NaN(maxInstances, 1);
impactEffAccIncl = NaN(maxInstances, 1);
impactBeta1Excl  = NaN(maxInstances, 1);
impactBeta1Incl  = NaN(maxInstances, 1);
nImpactRows = 0;

for condIdx = 1:nConditions
    outlierPs = find(isOutlier(condIdx, :));
    if isempty(outlierPs) || isempty(groupFitsFull{condIdx})
        continue
    end

    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    fitsExcl = groupFitsFull{condIdx};
    keepExcl = groupKeepPExcl{condIdx};

    beta0Excl = fitsExcl(1);
    beta1Excl = fitsExcl(2);
    lowerAsyExcl = fitsExcl(end-1);
    upperAsyExcl = 1 - fitsExcl(end);
    effAccExcl = upperAsyExcl; if beta1Excl <= 0, effAccExcl = lowerAsyExcl; end

    combinedExclTmp = vertcat(allTrialData{condIdx, keepExcl});
    maxXExcl = max(combinedExclTmp(:,2));
    rawAccExcl = computeGroupRawAccTrace(allRawAcc, condIdx, keepExcl);

    for oi = 1:length(outlierPs)
        thisOutlierP = outlierPs(oi);
        pidStr = sprintf('%03d', participantIDs(thisOutlierP));
        if isempty(allTrialData{condIdx, thisOutlierP})
            continue
        end

        keepIncl = union(keepExcl, thisOutlierP);
        combinedIncl = vertcat(allTrialData{condIdx, keepIncl});

        try
            maxXIncl = max(combinedIncl(:,2));
            [fitsIncl, ~] = fitLogisticCapped(combinedIncl, 'lu2', beta1Cap, maxXIncl, midpointSlackLow, midpointSlackHigh);
        catch ME
            fprintf('Including-outlier fit failed for P%s Std %.3f H %.3f: %s\n', pidStr, avgStdVal, avgHVal, ME.message);
            continue
        end

        beta0Incl = fitsIncl(1);
        beta1Incl = fitsIncl(2);
        lowerAsyIncl = fitsIncl(end-1);
        upperAsyIncl = 1 - fitsIncl(end);
        effAccIncl = upperAsyIncl; if beta1Incl <= 0, effAccIncl = lowerAsyIncl; end

        rawAccIncl = computeGroupRawAccTrace(allRawAcc, condIdx, keepIncl);

        fprintf('  Std %.0f H %.3f -- P%s: effAcc excl=%.3f incl=%.3f (delta=%.3f) | beta1 excl=%.3f incl=%.3f (delta=%.3f)\n', ...
            avgStdVal, avgHVal, pidStr, effAccExcl, effAccIncl, effAccIncl - effAccExcl, ...
            beta1Excl, beta1Incl, beta1Incl - beta1Excl);

        nImpactRows = nImpactRows + 1;
        impactCondIdx(nImpactRows) = condIdx;
        impactPIdx(nImpactRows) = thisOutlierP;
        impactEffAccExcl(nImpactRows) = effAccExcl;
        impactEffAccIncl(nImpactRows) = effAccIncl;
        impactBeta1Excl(nImpactRows) = beta1Excl;
        impactBeta1Incl(nImpactRows) = beta1Incl;

        % --- Diagnostic leave-one-out plot ---
        maxXPlot = max(maxXExcl, maxXIncl);
        predMatrix = [ones(maxXPlot, 1), (1:maxXPlot)'];
        smoothedExcl = logist_valLU2(fitsExcl, predMatrix);
        smoothedIncl = logist_valLU2(fitsIncl, predMatrix);

        nExcl = length(rawAccExcl);
        nIncl = length(rawAccIncl);
        tExcl = (1:nExcl) * frameRate;
        tIncl = (1:nIncl) * frameRate;
        tFit = (1:maxXPlot) * frameRate;

        figure('Position', [100 100 900 650]);
        hold on;
        plot(tExcl, rawAccExcl, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.5);
        plot(tIncl, rawAccIncl, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
        plot(tFit, smoothedExcl, 'r-', 'LineWidth', 4);
        plot(tFit, smoothedIncl, 'b-', 'LineWidth', 4);

        ax = gca;
        ax.FontSize = 16; ax.LineWidth = 2; ax.TickLength = [0.02 0.02];
        xlim([0 30]); ylim([0 1]); grid on;
        xlabel('Seconds Since Source Switch', 'FontSize', 20, 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontSize', 20, 'FontWeight', 'bold');
        legend({'Raw (excl. outlier)', 'Raw (incl. outlier)', ...
                 sprintf('Fit excl. (acc=%.2f, \\beta_1=%.2f)', effAccExcl, beta1Excl), ...
                 sprintf('Fit incl. (acc=%.2f, \\beta_1=%.2f)', effAccIncl, beta1Incl)}, ...
                'Location', 'southoutside', 'FontSize', 12);
        title(sprintf('Std %.0f, H %.3f -- Outlier P%s impact (n excl=%d, n incl=%d)', ...
            avgStdVal, avgHVal, pidStr, length(keepExcl), length(keepIncl)), 'FontSize', 16);
    end
end

impactCondIdx    = impactCondIdx(1:nImpactRows);
impactPIdx       = impactPIdx(1:nImpactRows);
impactEffAccExcl = impactEffAccExcl(1:nImpactRows);
impactEffAccIncl = impactEffAccIncl(1:nImpactRows);
impactBeta1Excl  = impactBeta1Excl(1:nImpactRows);
impactBeta1Incl  = impactBeta1Incl(1:nImpactRows);
impactDeltaAcc   = impactEffAccIncl - impactEffAccExcl;
impactDeltaBeta1 = impactBeta1Incl - impactBeta1Excl;

%% --- Pass 4b: aggregate outlier-impact summary (across all conditions) ---
fprintf('\n=== Aggregate outlier impact summary ===\n');
impactLabels = cell(nImpactRows, 1);
for i = 1:nImpactRows
    pidStr = sprintf('%03d', participantIDs(impactPIdx(i)));
    stdVal = nanmean(allActualVals(impactCondIdx(i), :, 1));
    hVal = nanmean(allActualVals(impactCondIdx(i), :, 2));
    impactLabels{i} = sprintf('Std%.0f H%.3f P%s', stdVal, hVal, pidStr);
    fprintf('  %s: delta acc = %+.3f, delta beta1 = %+.3f\n', impactLabels{i}, impactDeltaAcc(i), impactDeltaBeta1(i));
end

if nImpactRows > 0
    [~, sortOrder] = sort(abs(impactDeltaAcc), 'descend');

    figure('Position', [100 100 1400 600]);

    subplot(1,2,1);
    barh(impactDeltaAcc(sortOrder));
    hold on;
    xline(accDeltaThreshold, 'r--', 'LineWidth', 2);
    xline(-accDeltaThreshold, 'r--', 'LineWidth', 2);
    ax = gca; ax.FontSize = 11; ax.YTick = 1:nImpactRows; ax.YTickLabel = impactLabels(sortOrder);
    xlabel('\Delta Effective Accuracy (incl. - excl.)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Outlier impact on pooled asymptote', 'FontSize', 15);
    grid on;

    subplot(1,2,2);
    barh(impactDeltaBeta1(sortOrder));
    hold on;
    xline(beta1DeltaThreshold, 'r--', 'LineWidth', 2);
    xline(-beta1DeltaThreshold, 'r--', 'LineWidth', 2);
    ax = gca; ax.FontSize = 11; ax.YTick = 1:nImpactRows; ax.YTickLabel = impactLabels(sortOrder);
    xlabel('\Delta \beta_1 (incl. - excl.)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Outlier impact on pooled slope', 'FontSize', 15);
    grid on;
else
    fprintf('  (no outlier instances to summarize)\n');
end

%% --- Pass 4c: decide final inclusion based on impact thresholds ---
finalOutlierMask = isOutlier; % start from the statistical flags

fprintf('\n=== Final inclusion decisions (threshold: |dAcc|<=%.3f AND |dBeta1|<=%.3f) ===\n', ...
    accDeltaThreshold, beta1DeltaThreshold);
for i = 1:nImpactRows
    condIdx = impactCondIdx(i);
    p = impactPIdx(i);
    pidStr = sprintf('%03d', participantIDs(p));
    stdVal = nanmean(allActualVals(condIdx, :, 1));
    hVal = nanmean(allActualVals(condIdx, :, 2));

    negligible = abs(impactDeltaAcc(i)) <= accDeltaThreshold && abs(impactDeltaBeta1(i)) <= beta1DeltaThreshold;
    if negligible
        finalOutlierMask(condIdx, p) = false;
        decisionStr = 'RESCUED (included in final pooled fit)';
    else
        decisionStr = 'EXCLUDED (impact too large)';
    end
    fprintf('  Std %.0f H %.3f P%s: dAcc=%+.3f dBeta1=%+.3f -> %s\n', ...
        stdVal, hVal, pidStr, impactDeltaAcc(i), impactDeltaBeta1(i), decisionStr);
end

%% --- Pass 5: final per-condition group plots (using finalOutlierMask) ---
for condIdx = 1:nConditions
    if isempty(groupFitsFull{condIdx})
        continue
    end

    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    keepExcl = groupKeepPExcl{condIdx};
    finalKeepP = find(~finalOutlierMask(condIdx, :) & ~cellfun(@isempty, allTrialData(condIdx, :)));

    outlierPs = find(isOutlier(condIdx, :));
    rescuedPs = outlierPs(~finalOutlierMask(condIdx, outlierPs));
    stillExcludedPs = outlierPs(finalOutlierMask(condIdx, outlierPs));

    if isequal(sort(finalKeepP), sort(keepExcl))
        fitsToPlot = groupFitsFull{condIdx};
        nUsed = length(keepExcl);
    else
        combinedFinal = vertcat(allTrialData{condIdx, finalKeepP});
        try
            maxXFinal = max(combinedFinal(:,2));
            [fitsToPlot, ~] = fitLogisticCapped(combinedFinal, 'lu2', beta1Cap, maxXFinal, midpointSlackLow, midpointSlackHigh);
        catch ME
            fprintf('Final refit failed for Std %.3f H %.3f: %s -- falling back to excluding-outliers fit\n', ...
                avgStdVal, avgHVal, ME.message);
            fitsToPlot = groupFitsFull{condIdx};
            finalKeepP = keepExcl;
        end
        nUsed = length(finalKeepP);
    end

    if isempty(outlierPs)
        statusStr = 'no outliers flagged';
    else
        parts = {};
        if ~isempty(rescuedPs)
            parts{end+1} = sprintf('rescued: %s', strjoin(arrayfun(@(pp) sprintf('P%03d', participantIDs(pp)), rescuedPs, 'UniformOutput', false), ', ')); %#ok<AGROW>
        end
        if ~isempty(stillExcludedPs)
            parts{end+1} = sprintf('excluded: %s', strjoin(arrayfun(@(pp) sprintf('P%03d', participantIDs(pp)), stillExcludedPs, 'UniformOutput', false), ', ')); %#ok<AGROW>
        end
        statusStr = strjoin(parts, ' | ');
    end

    titleStr = sprintf('Std %.0f, Hazard %.3f (n=%d/%d participants -- %s)', ...
        avgStdVal, avgHVal, nUsed, length(participantIDs), statusStr);

    plotPooledGroupFit(fitsToPlot, finalKeepP, condIdx, allRawAcc, frameRate, titleStr);
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
    keep = ~finalOutlierMask(condIdx, :);
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
[rOut, cOut] = find(finalOutlierMask'); % note transpose to match plotted orientation
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
    keep = ~finalOutlierMask(condIdx, :);

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

%% --- Strategy-Based Accuracy Analysis ---
% For each condition, splits participants by which strategy they used
% (per stratAssignment) and produces one plot per strategy (3 plots per
% condition, laid out as subplots in a single figure per condition).
% Participants assigned NaN for a given condition are excluded from all
% strategy groups. Statistical outliers (finalOutlierMask), if computed
% earlier in this script, are also excluded here for consistency with
% the rest of the pipeline -- comment that line out if you'd rather keep
% outliers in the strategy breakdown.

nStrategies = 3;
strategyColors = [0.85 0.33 0.10;   % strategy 1
                   0.00 0.45 0.74;   % strategy 2
                   0.47 0.67 0.19];  % strategy 3

for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal   = nanmean(allActualVals(condIdx, :, 2));

    if condIdx > length(stratAssignment) || isempty(stratAssignment{condIdx})
        continue
    end
    stratVec = stratAssignment{condIdx}; % column, one entry per participant

    figure('Position', [100 100 1800 500]);
    sgtitle(sprintf('Std %.0f, Hazard %.3f -- Accuracy by Strategy', avgStdVal, avgHVal), ...
        'FontSize', 20, 'FontWeight', 'bold');

    for strat = 1:nStrategies
        subplot(1, nStrategies, strat);

        keepMask = (stratVec' == strat) & ~cellfun(@isempty, allTrialData(condIdx, :));
        if exist('finalOutlierMask', 'var')
            keepMask = keepMask & ~finalOutlierMask(condIdx, :);
        end
        keepP = find(keepMask);

        if isempty(keepP)
            text(0.5, 0.5, sprintf('No participants\nusing Strategy %d', strat), ...
                'HorizontalAlignment', 'center', 'FontSize', 16);
            axis off;
            continue;
        end

        groupRawAcc = computeGroupRawAccTrace(allRawAcc, condIdx, keepP);
        numFrames = length(groupRawAcc);
        plotTimeSeconds = (1:numFrames) * frameRate;

        plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
        hold on;

        combinedData = vertcat(allTrialData{condIdx, keepP});

        try
            maxXStrat = max(combinedData(:,2));
            [fitsStrat, ~] = fitLogisticCapped(combinedData, 'lu2', beta1Cap, maxXStrat, ...
                midpointSlackLow, midpointSlackHigh);

            predMatrix = [ones(numFrames, 1), (1:numFrames)'];
            smoothedAcc = logist_valLU2(fitsStrat, predMatrix);
            plot(plotTimeSeconds, smoothedAcc, '-', 'Color', strategyColors(strat,:), 'LineWidth', 4);

            beta0 = fitsStrat(1); beta1 = fitsStrat(2);
            lowerAsy = fitsStrat(end-1); upperAsy = 1 - fitsStrat(end);
            effAcc = upperAsy; if beta1 <= 0, effAcc = lowerAsy; end

            yline(effAcc, '--k', 'LineWidth', 2);
            text(28, max(min(effAcc - 0.05, 0.95), 0.05), sprintf('Asy: %.2f', effAcc), ...
                'FontWeight', 'bold', 'FontSize', 14, 'HorizontalAlignment', 'right');
        catch ME
            fprintf('Strategy fit failed for cond %d strat %d: %s\n', condIdx, strat, ME.message);
        end

        ax = gca;
        ax.FontSize = 14; ax.LineWidth = 1.5; ax.TickLength = [0.02 0.02];
        xlim([0 30]); ylim([0 1]); grid on;
        xlabel('Seconds Since Switch', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontSize', 16, 'FontWeight', 'bold');

        pidLabels = strjoin(arrayfun(@(pp) sprintf('P%03d', participantIDs(pp)), ...
            keepP, 'UniformOutput', false), ', ');
        title(sprintf('Strategy %d (n=%d: %s)', strat, length(keepP), pidLabels), ...
            'FontSize', 13, 'Interpreter', 'none');
    end
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


function trace = computeGroupRawAccTrace(allRawAcc, condIdx, keepIdx)
raw = squeeze(nanmean(allRawAcc(condIdx, keepIdx, :), 2));
lastValidIdx = find(~isnan(raw), 1, 'last');
if isempty(lastValidIdx)
    trace = [];
else
    trace = raw(1:lastValidIdx);
end
end


function plotPooledGroupFit(fits, keepP, condIdx, allRawAcc, frameRate, titleStr)
groupRawAcc = computeGroupRawAccTrace(allRawAcc, condIdx, keepP);
numFrames = length(groupRawAcc);
plotTimeSeconds = (1:numFrames) * frameRate;
predMatrix = [ones(numFrames, 1), (1:numFrames)'];
smoothedAcc = logist_valLU2(fits, predMatrix);

beta0 = fits(1); beta1 = fits(2);
lowerAsy = fits(end-1); upperAsy = 1 - fits(end);
effAcc = upperAsy; if beta1 <= 0, effAcc = lowerAsy; end
range_ = effAcc - lowerAsy;
midpointSec = (-beta0 / beta1) * frameRate;
maxSlopePerSec = ((beta1 * range_) / 4) / frameRate;
yMidGroup = (lowerAsy + effAcc) / 2;

figure;
plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
hold on;
plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);
yline(effAcc, '--k', 'LineWidth', 2.5);
text(28, effAcc - 0.05, sprintf('Asy: %.2f', effAcc), ...
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
title(titleStr, 'FontSize', 22, 'Interpreter', 'none');
end