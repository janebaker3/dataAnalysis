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
midpointSlackLow  = 0.05; % midpoint allowed up to 0.5*maxX BEFORE t=0
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
            f = figure('Visible', 'off'); %%%%
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

%% --- Pass 3b: group curves from averaged individual parameters (ALL participants) ---
% Strategy change: instead of pooling every trial into one giant logistic fit,
% fit each participant separately (Pass 3a) and average the parameters.
%
% WHY THIS FIXES THE FLAT-LINE PROBLEM:
% A pooled fit is a single logistic fit to a *mixture* of participants with
% different midpoints and asymptotes. That mixture is not itself a logistic --
% it's a smeared ramp. In low-hazard conditions the smear is wide enough that a
% near-flat line fits it about as well as any sigmoid, so beta1 collapses to 0.
% Fitting within participant first (where the sigmoid IS coherent) and averaging
% afterwards removes that artifact.
%
% EVERY participant contributes. Negative and shallow slopes are kept; they are
% reported as diagnostics only. Two parameterization details make that safe:
%
%  1. beta0 is NEVER averaged. beta0 = -midpoint*beta1, so averaging it across
%     people with different slopes gives a midpoint describing nobody. We average
%     the midpoint, then rebuild beta0 = -mean(midpoint)*mean(beta1).
%
%  2. Asymptotes are averaged SIGN-AWARE. L/U mean "bottom/top of range", not
%     "start/end": beta1>0 runs L->U, beta1<0 runs U->L. So we convert each
%     participant to yEarly (t->0 asymptote) / yLate (t->inf asymptote), average
%     those, then map back. A negative-slope participant then just pulls the
%     group rise down slightly instead of corrupting the asymptote average.

excludeRawAccOutliers       = true;   % keeps your existing IQR rule; false = literally everyone
useMedianAcrossParticipants = false;  % true = median (still uses all participants, just robust)
minParticipantsForGroup     = 2;
flatRiseThreshold           = 0.02;   % DIAGNOSTIC ONLY -- nothing is removed
alsoFitPooledForComparison  = false;  % true = overlay old pooled fit in magenta

groupBeta0s       = NaN(1, nConditions);
groupBeta1s       = NaN(1, nConditions);
groupLowerAsys    = NaN(1, nConditions);
groupUpperAsys    = NaN(1, nConditions);
groupEffAccs      = NaN(1, nConditions);
pooledOverallAccs = NaN(1, nConditions);   % now: mean of per-participant accuracy
groupMidpointSecs = NaN(1, nConditions);
groupSlopePerSecs = NaN(1, nConditions);
groupParamSEMs    = NaN(nConditions, 4);   % [midpointFrames, beta1, yEarly, yLate]
nUsedForAvg       = zeros(1, nConditions);
nNegSlope         = zeros(1, nConditions);
nFlatish          = zeros(1, nConditions);

if useMedianAcrossParticipants
    centralTendency = @(v) median(v, 'omitnan');
    ctLabel = 'median';
else
    centralTendency = @(v) mean(v, 'omitnan');
    ctLabel = 'mean';
end

fprintf('\n=== Group curves from averaged individual parameters (%s, all participants) ===\n', ctLabel);

for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal   = nanmean(allActualVals(condIdx, :, 2));

    % Only requirement to contribute: the individual fit actually produced numbers.
    % (reshape, not squeeze' -- allParams(condIdx,:,2) is already 1x10, so the old
    %  transpose made it 10x1 and silently broadcast against the 1x10 outlier mask.)
    haveFit = ~isnan(reshape(allParams(condIdx, :, 2), 1, []));
    outRow  = reshape(isOutlier(condIdx, :), 1, []);

    if excludeRawAccOutliers
        candidates = find(haveFit & ~outRow);
    else
        candidates = find(haveFit);
    end
    candidates = reshape(candidates, 1, []);
    assert(all(candidates <= length(participantIDs)), ...
        'candidate index out of range -- shape bug in haveFit/isOutlier');

    pMid = []; pB1 = []; pEarly = []; pLate = []; usedP = []; pNeg = [];

    for p = candidates
        b0 = allParams(condIdx, p, 1);
        b1 = allParams(condIdx, p, 2);
        lo = allParams(condIdx, p, 3);
        up = allParams(condIdx, p, 4);

        thisMaxX = max(allTrialData{condIdx, p}(:, 2));
        lowerMid = -midpointSlackLow  * thisMaxX;
        upperMid =  midpointSlackHigh * thisMaxX;

        % --- midpoint, guarded against 0/0 and clamped to the SAME bounds the
        %     fit was already constrained by. No-op for well-behaved fits.
        b1safe = b1;
        if abs(b1safe) < 1e-9
            if b1safe == 0, b1safe = 1e-9; else, b1safe = sign(b1safe) * 1e-9; end
        end
        mid = -b0 / b1safe;
        if ~isfinite(mid), mid = 0; end
        mid = min(max(mid, lowerMid), upperMid);

        % --- sign-aware asymptotes: early (t->0) vs late (t->inf) ---
        if b1 >= 0
            yEarly = lo;  yLate = up;
        else
            yEarly = up;  yLate = lo;
        end

        % --- diagnostics only; this participant is kept regardless ---
        fitsVec  = [b0; b1; lo; 1 - up];
        yy       = logist_valLU2(fitsVec, [1, 1; 1, thisMaxX]);
        predRise = yy(2) - yy(1);
        isNeg    = (b1 < 0);
        isFlat   = (abs(predRise) < flatRiseThreshold);

        tol       = 1e-4 * max(1, thisMaxX);
        atLowBnd  = (mid - lowerMid) <  tol;
        atHighBnd = (upperMid - mid) <  tol;

        if isNeg, nNegSlope(condIdx) = nNegSlope(condIdx) + 1; end
        if isFlat, nFlatish(condIdx) = nFlatish(condIdx) + 1; end
        if isNeg || isFlat || atLowBnd || atHighBnd
            bndTag = '';
            if atLowBnd,  bndTag = ' [MIDPOINT PINNED AT LOWER BOUND]'; end
            if atHighBnd, bndTag = ' [MIDPOINT PINNED AT UPPER BOUND]'; end
            fprintf('    note: P%03d KEPT (beta1=%+.3f, mid=%+.2f s, rise=%+.3f over %d frames)%s%s%s\n', ...
                participantIDs(p), b1, mid*frameRate, predRise, thisMaxX, ...
                repmat(' [neg slope]',1,isNeg), repmat(' [flat]',1,isFlat), bndTag);
        end

        pMid(end+1)   = mid;    %#ok<SAGROW>
        pB1(end+1)    = b1;     %#ok<SAGROW>
        pEarly(end+1) = yEarly; %#ok<SAGROW>
        pLate(end+1)  = yLate;  %#ok<SAGROW>
        pNeg(end+1)   = isNeg;  %#ok<SAGROW>
        usedP(end+1)  = p;      %#ok<SAGROW>
    end

    nUsed = numel(usedP);
    nUsedForAvg(condIdx) = nUsed;
    if nUsed < minParticipantsForGroup
        fprintf('WARNING: Std %.0f H %.3f -- only %d individual fits available, skipping group curve.\n', ...
                avgStdVal, avgHVal, nUsed);
        continue
    end

    % --- average the interpretable parameters ---
    mMid   = median(pMid, 'omitnan');   % robust to boundary-pinned midpoints
    mB1    = centralTendency(pB1);
    mEarly = centralTendency(pEarly);
    mLate  = centralTendency(pLate);

    % --- map back to the L/U packing logist_valLU2 expects ---
    groupLowerAsy = min(mEarly, mLate);
    groupUpperAsy = max(mEarly, mLate);
    groupBeta0    = -mMid * mB1;          % <-- reconstructed, NOT averaged
    groupBeta1    = mB1;
    groupEffAcc   = mLate;                % accuracy at long times = what you want

    if sign(mB1) ~= sign(mLate - mEarly) && abs(mLate - mEarly) > 1e-6
        fprintf(['WARNING: Std %.0f H %.3f -- mean slope sign (%+.3f) disagrees with mean ' ...
                 'asymptote direction (%+.3f). Group signal is near-ambiguous here.\n'], ...
                 avgStdVal, avgHVal, mB1, mLate - mEarly);
    end

    groupParamSEMs(condIdx, :) = [std(pMid), std(pB1), std(pEarly), std(pLate)] ./ sqrt(nUsed);

    groupBeta0s(condIdx)       = groupBeta0;
    groupBeta1s(condIdx)       = groupBeta1;
    groupLowerAsys(condIdx)    = groupLowerAsy;
    groupUpperAsys(condIdx)    = groupUpperAsy;
    groupEffAccs(condIdx)      = groupEffAcc;
    pooledOverallAccs(condIdx) = nanmean(allOverallAcc(condIdx, usedP));

    groupFitsAvg = [groupBeta0; groupBeta1; groupLowerAsy; 1 - groupUpperAsy];

    groupRawAcc  = squeeze(nanmean(allRawAcc(condIdx, usedP, :), 2));
    lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
    groupRawAcc  = groupRawAcc(1:lastValidIdx);
    numFrames    = length(groupRawAcc);
    plotTimeSeconds = (1:numFrames) * frameRate;

    predMatrix  = [ones(numFrames, 1), (1:numFrames)'];
    smoothedAcc = logist_valLU2(groupFitsAvg, predMatrix);

    midpointSec    = mMid * frameRate;
    groupRange     = groupUpperAsy - groupLowerAsy;
    maxSlopePerSec = ((groupBeta1 * groupRange) / 4) / frameRate;
    yMidGroup      = (groupLowerAsy + groupUpperAsy) / 2;

    groupMidpointSecs(condIdx) = midpointSec;
    groupSlopePerSecs(condIdx) = maxSlopePerSec;

    avgCurveRise = smoothedAcc(end) - smoothedAcc(1);
    if abs(avgCurveRise) < flatRiseThreshold
        fprintf(['WARNING: Std %.0f H %.3f -- averaged curve is flat (rise=%+.3f) despite ' ...
                 'only %d/%d negative and %d/%d flat individual fits. Midpoint SD=%.1f frames.\n'], ...
                 avgStdVal, avgHVal, avgCurveRise, nNegSlope(condIdx), nUsed, ...
                 nFlatish(condIdx), nUsed, std(pMid));
    end

    fprintf(['Std %.0f H %.3f | n=%d (all kept; %d neg slope, %d flat) | midpoint %.2f s (SEM %.2f) | ' ...
             'slope %.3f acc/s | early %.3f | late %.3f | curve rise %+.3f\n'], ...
             avgStdVal, avgHVal, nUsed, nNegSlope(condIdx), nFlatish(condIdx), ...
             midpointSec, groupParamSEMs(condIdx,1)*frameRate, maxSlopePerSec, ...
             mEarly, mLate, avgCurveRise);

    % ----------------------------- plot -----------------------------
    figure;
    hold on;

    % every individual curve, over its own observed window
    % blue = positive slope, orange = negative slope (kept, just visible)
    for k = 1:nUsed
        p = usedP(k);
        thisMaxX = max(allTrialData{condIdx, p}(:, 2));
        indFits  = [allParams(condIdx,p,1); allParams(condIdx,p,2); ...
                    allParams(condIdx,p,3); 1 - allParams(condIdx,p,4)];
        indY = logist_valLU2(indFits, [ones(thisMaxX,1), (1:thisMaxX)']);
        if pNeg(k)
            indCol = [0.95 0.55 0.15 0.55];
        else
            indCol = [0.65 0.65 0.90 0.50];
        end
        plot((1:thisMaxX) * frameRate, indY, '-', 'Color', indCol, 'LineWidth', 1);
    end

    plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);

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

    if alsoFitPooledForComparison
        try
            combinedData = vertcat(allTrialData{condIdx, usedP});
            maxXPooled   = max(combinedData(:,2));
            pooledFits   = fitLogisticCapped(combinedData, 'lu2', beta1Cap, maxXPooled, ...
                                             midpointSlackLow, midpointSlackHigh);
            plot(plotTimeSeconds, logist_valLU2(pooledFits, predMatrix), 'm--', 'LineWidth', 3);
        catch ME
            fprintf('  (pooled comparison fit failed: %s)\n', ME.message);
        end
    end

    ax = gca;
    ax.FontSize = 24; ax.LineWidth = 3; ax.TickLength = [0.02 0.02];
    xlim([0 30]); ylim([0 1]); grid on;
    xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
    ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
    title(sprintf('Std %.3f, Hazard %.3f (%s of %d individual fits, all participants)', ...
        avgStdVal, avgHVal, ctLabel, nUsed), 'FontSize', 24);
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