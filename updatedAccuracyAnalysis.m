%% --- Environment Setup ---
% Define the participants to include in the analysis
participantIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

% Define paths for saving figures and locating necessary analysis toolboxes
figurePath = '/Users/janebaker/Desktop/triangles/taskData/figures';
addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
addpath(genpath('../../analysis/'))

% Task design parameters (3 std levels x 2 hazard levels = 6 conditions)
nStd = 3;
nHazard = 2;
nConditions = nStd * nHazard;

% FIX #2: Cap on beta1 (the logistic slope coefficient on frames-since-
% switch). Several individual AND pooled fits were hitting beta1 = 25-50,
% which is a quasi-complete-separation degenerate solution -- it means
% the optimizer found that accuracy jumps from one asymptote to the other
% in less than a single frame, which produced flat-looking fit lines that
% ignored the actual gradual rise visible in the raw data (see Std480/
% H0.003 condition). This is most common in low-hazard conditions where
% switches are rare, so at long times-since-switch there is essentially
% zero response variance (P(correct)=1 exactly), which drives ML slope
% estimates toward infinity.
% Well-constrained/well-behaved fits in this dataset show beta1 in the
% ~0.01-0.15 range, so 1.5 is a generous cap that only kicks in for
% genuinely degenerate fits, while still allowing considerably faster
% transitions than anything currently well-resolved.
beta1Cap = 1.5;

% Preallocate array to store logistic fit parameters for all participants
% Dimensions: [Condition x Participant x Parameter]
% Condition index = (stdIdx-1)*nHazard + hazardIdx, ranging 1-6
% Parameters: 1=beta0 (intercept), 2=beta1 (slope), 3=lowerAsy, 4=upperAsy, 5=EffectiveAccuracy
allParams = NaN(nConditions, length(participantIDs), 5);

% Preallocate array to store raw accuracy as a function of frames since switch
maxFramesPossible = 2000;
% Dimensions: [Condition x Participant x FrameIndex]
allRawAcc = NaN(nConditions, length(participantIDs), maxFramesPossible);

% Dimensions: [Condition x Participant x (stdVal, hazardVal)]
allActualVals = NaN(nConditions, length(participantIDs), 2);

% Dimensions: [Condition x Participant] - simple overall accuracy per block
allOverallAcc = NaN(nConditions, length(participantIDs));

% Dimensions: [Condition x Participant x (lowerAsySEM, upperAsySEM)]
% Stores the standard error of each asymptote estimate, which gives a
% direct, statistically principled measure of how well-constrained each
% fit is (rather than relying only on switch count as an indirect proxy)
allAsySEMs = NaN(nConditions, length(participantIDs), 2);

% Diagnostic log to flag suspicious/non-converged logistic fits
% Will be converted to a table at the end for easy sorting/inspection
diagLog = struct('participant', {}, 'condIdx', {}, 'stdVal', {}, 'hVal', {}, ...
    'nTrials', {}, 'nSwitches', {}, 'beta1', {}, 'lowerAsy', {}, 'upperAsy', {}, ...
    'lowerAsySEM', {}, 'upperAsySEM', {}, 'effAcc', {}, 'flagReason', {});

% Accumulate trial-level (frames-since-switch, correct/incorrect) data
% across ALL participants for each condition, to support a pooled group-level
% fit rather than averaging individual fit parameters (which breaks down when
% some individual fits are degenerate/quasi-separated -- see diagnostic flags)
pooledFitData = cell(nConditions, 1);

%% --- Individual Participant Loop ---
% Iterate over each specified participant to extract data and fit logistic models
for p = 1:length(participantIDs)
    pid = participantIDs(p);
    pidStr = sprintf('%03d', pid); % zero-padded participant ID (p001, p002, ...)
    
    % Load behavioral data for the current participant
    load(['/Users/janebaker/Desktop/triangles/taskData/finalData/p' pidStr '.mat']);
    
    % Extract unique block numbers and map them for filtering
    blocks = unique([data.blockNumber]);
    blockNum = [data.blockNumber];
    
    % Loop through blocks (starting at index 2, skipping training/practice block,
    % which is confirmed to always be the lowest blockNumber)
    for b = 2:length(blocks)
        % Filter task data, source states, and participant responses for this specific block
        thisBlockTask = data(blockNum == blocks(b));
        thisBlockSource = [thisBlockTask.sourceIsLeft];
        thisBlockBelief = [thisBlockTask.keyPressed];
        stdIdx = thisBlockTask(1).stdHrzIdx;
        hIdx = thisBlockTask(1).hazardIdx;
        
        % pull actual std/hazard values for labeling
        stdVal = thisBlockTask(1).stdHrz;
        hVal = thisBlockTask(1).H;
        
        % Compute condition index from std/hazard rather than relying on block number
        condIdx = (stdIdx - 1) * nHazard + hIdx;
        
        % Store actual values for this condition/participant
        allActualVals(condIdx, p, 1) = stdVal;
        allActualVals(condIdx, p, 2) = hVal;
        
        nTrials = length(thisBlockTask);
        
        % Preallocate arrays to track time since switch and trial accuracy
        trialFramesSince = zeros(nTrials, 1);
        trialIsCorrect = zeros(nTrials, 1);

        % Calculate raw accuracy and track how long it has been since the source changed
        countSince = 0;
        for t = 1:nTrials
            % Reset counter if it's the first trial or if the hidden source switched
            if t == 1 || thisBlockSource(t) ~= thisBlockSource(t-1)
                countSince = 1;
            else
                countSince = countSince + 1;
            end
            trialFramesSince(t) = countSince;
            
            % Check accuracy: True if source is Left (1) and key is 'F', 
            % OR if source is Right (0) and key is 'J'
            if (thisBlockSource(t) && thisBlockBelief(t) == 'F') || ...
               (~thisBlockSource(t) && thisBlockBelief(t) == 'J')
                trialIsCorrect(t) = 1;
            else
                trialIsCorrect(t) = 0;
            end
        end

        % Store simple overall accuracy for this block (computed AFTER
        % trialIsCorrect is actually populated by the loop above)
        allOverallAcc(condIdx, p) = mean(trialIsCorrect);
        
        % Calculate and store the mean raw accuracy for every frame interval
        maxX = max(trialFramesSince);
        for n = 1:maxX
            allRawAcc(condIdx, p, n) = mean(trialIsCorrect(trialFramesSince == n));
        end
        
        % Logistic Fit
        % Prepare matrix for logistic regression: [Intercept, Predictor(frames), Response(accuracy)]
        fitData = [ones(nTrials, 1), trialFramesSince, trialIsCorrect];

        % Accumulate this block's trial-level data into the pooled dataset
        % for this condition (used later for a single group-level fit)
        pooledFitData{condIdx} = [pooledFitData{condIdx}; fitData];

        try
            % FIX #2: Use the beta1-capped wrapper instead of calling
            % logist_fit directly, to prevent quasi-complete-separation
            % solutions (beta1 -> 25-50) seen in low-hazard conditions.
            [fits, sems] = fitLogisticCapped(fitData, 'lu2', beta1Cap);
            
            % Extract fitted parameters
            beta0 = fits(1);          % Intercept
            beta1 = fits(2);          % Slope coefficient
            lowerAsy = fits(end-1);   % Lower asymptote (chance level)
            upperAsy = 1 - fits(end); % Upper asymptote (maximum accuracy achieved)
            
            % SEMs are in the same order as fits: [betas..., gamma_sem, lambda_sem]
            lowerAsySEM = sems(end-1);
            upperAsySEM = sems(end);
            
            % --- Diagnostic Checks: Flag suspicious/non-converged fits ---
            nSwitches = sum(diff(thisBlockSource) ~= 0) + 1;
            
            flagReasons = {};
            if nTrials < 30
                flagReasons{end+1} = 'low nTrials';
            end
            if nSwitches < 5
                flagReasons{end+1} = 'low nSwitches';
            end
            if abs(beta1) < 1e-6
                flagReasons{end+1} = 'near-zero slope';
            end
            semThreshold = 0.1;
            if lowerAsySEM > semThreshold || upperAsySEM > semThreshold
                flagReasons{end+1} = sprintf('high asymptote uncertainty (lowerSEM=%.3f, upperSEM=%.3f)', ...
                    lowerAsySEM, upperAsySEM);
            end
            if abs(lowerAsy - 0.5) < 1e-6 || abs(upperAsy - 0.5) < 1e-6
                flagReasons{end+1} = 'hit logist_fit bound (0.5)';
            end
            % NEW: flag fits that hit the new beta1 cap -- indicates the
            % data wants an even steeper (likely degenerate) transition
            % than we're now allowing, so timing-related outputs
            % (midpoint, slope) for this fit should be treated as
            % low-confidence even though the asymptotes should now be
            % much better behaved than the uncapped version
            if abs(abs(beta1) - beta1Cap) < 1e-3
                flagReasons{end+1} = sprintf('hit beta1 cap (%.2f)', beta1Cap);
            end
            
            % FIX #1: Determine which asymptote represents genuine
            % "settled" performance using the SIGN of beta1, rather than
            % proximity to raw overall accuracy. The raw-accuracy-
            % proximity heuristic failed specifically in high-hazard
            % conditions, where a large fraction of ALL trials fall in
            % the pre-asymptote ramp period -- this pulls the overall
            % mean toward the "wrong" asymptote even when the fit itself
            % is well-behaved and clearly rising/falling in one direction.
            % Sign-of-beta1 avoids this because it looks at the fitted
            % dynamics directly rather than a mean that's contaminated by
            % how much of the block was spent still transitioning.
            if beta1 > 0
                effAcc = upperAsy;
            else
                effAcc = lowerAsy;
            end
            
            % Sanity check: does the sign of beta1 actually agree with
            % the raw data's direction (early vs. late accuracy after a
            % switch)? Flag if not, since this would mean sign-of-beta1
            % itself is unreliable for this particular fit (e.g., noisy/
            % non-monotonic block).
            earlyWindow = trialFramesSince <= 5;
            lateWindow  = trialFramesSince >= max(1, maxX - 20) & trialFramesSince <= maxX;
            earlyAcc = mean(trialIsCorrect(earlyWindow));
            lateAcc  = mean(trialIsCorrect(lateWindow));
            dataDirectionPositive = lateAcc > earlyAcc;
            signMismatch = (beta1 > 0) ~= dataDirectionPositive;
            if signMismatch
                flagReasons{end+1} = sprintf('beta1 sign disagrees with raw data trend (early=%.3f, late=%.3f)', ...
                    earlyAcc, lateAcc);
            end
            
            if ~isempty(flagReasons)
                reasonStr = strjoin(flagReasons, '; ');
                fprintf('FLAG: P%s Std %.0f H %.3f (nTrials=%d, nSwitches=%d, beta1=%.4f, lowerAsy=%.3f, upperAsy=%.3f, lowerSEM=%.3f, upperSEM=%.3f) -- %s\n', ...
                    pidStr, stdVal, hVal, nTrials, nSwitches, beta1, lowerAsy, upperAsy, lowerAsySEM, upperAsySEM, reasonStr);
                
                diagLog(end+1) = struct('participant', pidStr, 'condIdx', condIdx, ...
                    'stdVal', stdVal, 'hVal', hVal, 'nTrials', nTrials, 'nSwitches', nSwitches, ...
                    'beta1', beta1, 'lowerAsy', lowerAsy, 'upperAsy', upperAsy, ...
                    'lowerAsySEM', lowerAsySEM, 'upperAsySEM', upperAsySEM, ...
                    'effAcc', NaN, 'flagReason', reasonStr); %#ok<SAGROW>
            end
            
            % Backfill effAcc into the most recent diagnostic log entry, if one was just added
            if ~isempty(diagLog) && diagLog(end).condIdx == condIdx && strcmp(diagLog(end).participant, pidStr)
                diagLog(end).effAcc = effAcc;
            end
            
            % Store extracted parameters for Group Calculation
            allParams(condIdx, p, 1) = beta0;
            allParams(condIdx, p, 2) = beta1;
            allParams(condIdx, p, 3) = lowerAsy;
            allParams(condIdx, p, 4) = upperAsy;
            allParams(condIdx, p, 5) = effAcc;
            
            % Store SEMs for group-level use
            allAsySEMs(condIdx, p, 1) = lowerAsySEM;
            allAsySEMs(condIdx, p, 2) = upperAsySEM;

            % --- Plot Individual Figure ---
            f = figure('Visible', 'off'); 
            
            plotTimeSeconds = (1:maxX) * 0.05;
            
            predMatrix = [ones(maxX, 1), (1:maxX)'];
            smoothedAcc = logist_valLU2(fits, predMatrix);
            
            avgAcc = arrayfun(@(n) mean(trialIsCorrect(trialFramesSince == n)), 1:maxX);
            plot(plotTimeSeconds, avgAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2);
            hold on;
            
            plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 4);
            
            range = upperAsy - lowerAsy;
            midpointFrames = -beta0 / beta1;
            midpointSeconds = midpointFrames * 0.05;
            maxSlopePerSec = ((beta1 * range) / 4) / 0.05;
            yMid = (lowerAsy + upperAsy) / 2;
            
            if midpointSeconds > 0 && midpointSeconds < 30
                tX = [midpointSeconds - 1, midpointSeconds + 1];
                tY = maxSlopePerSec * (tX - midpointSeconds) + yMid;
                plot(tX, tY, 'b-', 'LineWidth', 3); 
                plot(midpointSeconds, yMid, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
            end

            ax = gca;
            ax.FontSize = 20;
            ax.LineWidth = 2;
            ax.TickLength = [0.02 0.02];
            xlim([0 30]);
            ylim([0 1]);
            grid on;
            
            xlabel('Seconds Since Source Switch', 'FontSize', 24, 'FontWeight', 'bold');
            ylabel('Accuracy', 'FontSize', 24, 'FontWeight', 'bold');
            title(sprintf('P%s Std %.3f H %.3f: Logistic Fit', pidStr, stdVal, hVal), 'FontSize', 30);
            
            %saveas(f, fullfile(figurePath, sprintf('p%s_std%d_h%d_logistic_fit.png', pidStr, stdIdx, hIdx)));
            close(f);
        catch ME
            fprintf('Fit failed for P%s Std %.3f H %.3f: %s\n', pidStr, stdVal, hVal, ME.message);
        end
    end
end

%% --- Summarize Diagnostic Flags ---
if ~isempty(diagLog)
    diagTable = struct2table(diagLog);
    fprintf('\n=== %d suspicious fit(s) flagged out of %d total fits ===\n', ...
        height(diagTable), nConditions * length(participantIDs));
    disp(diagTable);
    % writetable(diagTable, fullfile(figurePath, 'fit_diagnostics.csv'));
else
    fprintf('\n=== No suspicious fits flagged ===\n');
end

%% --- Generate Group Average Figures ---
groupBeta0s = zeros(1, nConditions);
groupBeta1s = zeros(1, nConditions);
groupLowerAsys = zeros(1, nConditions);
groupUpperAsys = zeros(1, nConditions);
pooledOverallAccs = zeros(1, nConditions);

for condIdx = 1:nConditions
    
    stdIdx = floor((condIdx - 1) / nHazard) + 1;
    hIdx = mod(condIdx - 1, nHazard) + 1;
    
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    if isempty(pooledFitData{condIdx}), continue; end
    
    combinedData = pooledFitData{condIdx};
    nPooledTrials = size(combinedData, 1);
    
    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, :, :), 2));
    lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
    groupRawAcc = groupRawAcc(1:lastValidIdx);
    numFrames = length(groupRawAcc);
    plotTimeSeconds = (1:numFrames) * 0.05;
    
    try
        % FIX #2: capped pooled fit
        [groupFits, groupSems] = fitLogisticCapped(combinedData, 'lu2', beta1Cap);
        
        groupBeta0 = groupFits(1);
        groupBeta1 = groupFits(2);
        groupLowerAsy = groupFits(end-1);
        groupUpperAsy = 1 - groupFits(end);
        
        % FIX #1: sign-of-beta1 instead of closest-to-raw-mean
        if groupBeta1 > 0
            groupEffAcc = groupUpperAsy;
        else
            groupEffAcc = groupLowerAsy;
        end
        
        pooledOverallAcc = mean(combinedData(:,end));
        
        % Sanity check against raw data direction (early vs late in the
        % pooled/averaged raw accuracy curve)
        earlyIdx = 1:min(5, numFrames);
        lateIdx  = max(1, numFrames-20):numFrames;
        earlyRaw = nanmean(groupRawAcc(earlyIdx));
        lateRaw  = nanmean(groupRawAcc(lateIdx));
        dataDirectionPositive = lateRaw > earlyRaw;
        if (groupBeta1 > 0) ~= dataDirectionPositive
            fprintf('WARNING: Group fit Std %.3f H %.3f -- beta1 sign disagrees with raw data trend (early=%.3f, late=%.3f)\n', ...
                avgStdVal, avgHVal, earlyRaw, lateRaw);
        end
        if abs(abs(groupBeta1) - beta1Cap) < 1e-3
            fprintf('WARNING: Group fit Std %.3f H %.3f -- hit beta1 cap (%.2f)\n', ...
                avgStdVal, avgHVal, beta1Cap);
        end

        groupBeta0s(1, condIdx) = groupBeta0;
        groupBeta1s(1, condIdx) = groupBeta1;
        groupLowerAsys(1, condIdx) = groupLowerAsy;
        groupUpperAsys(1, condIdx) = groupUpperAsy;
        pooledOverallAccs(1, condIdx) = pooledOverallAcc;

        predMatrix = [ones(numFrames, 1), (1:numFrames)'];
        smoothedAcc = logist_valLU2(groupFits, predMatrix);
        
        groupRange = groupEffAcc - groupLowerAsy;
        midpointSec = (-groupBeta0 / groupBeta1) * 0.05;
        maxSlopePerSec = ((groupBeta1 * groupRange) / 4) / 0.05;
        yMidGroup = (groupLowerAsy + groupEffAcc) / 2;
        
        f = figure;
        plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
        hold on;
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
        
        ax = gca;
        ax.FontSize = 24;
        ax.LineWidth = 3;
        ax.TickLength = [0.02 0.02];
        xlim([0 30]);
        ylim([0 1]);
        grid on;
        xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
        title(sprintf('Std %.3f, Hazard %.3f (n=%d trials pooled)', avgStdVal, avgHVal, nPooledTrials), 'FontSize', 30);
        
        % saveas(f, fullfile(figurePath, sprintf('POSTER_Average_Std%d_H%d.png', stdIdx, hIdx)));
        
    catch ME
        fprintf('Pooled group fit failed for Std %.3f H %.3f: %s\n', avgStdVal, avgHVal, ME.message);
    end
end

%% --- Individual Participant Accuracy Comparison (Outlier Check) ---
accMatrixFit = squeeze(allParams(:, :, 5));
accMatrixRaw = allOverallAcc;

condLabels = strings(nConditions, 1);
for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    stdSpread = nanstd(allActualVals(condIdx, :, 1));
    hSpread = nanstd(allActualVals(condIdx, :, 2));
    if stdSpread > 1e-6 || hSpread > 1e-6
        fprintf('WARNING: condIdx %d has inconsistent std/hazard values across participants (stdSpread=%.4f, hSpread=%.4f)\n', ...
            condIdx, stdSpread, hSpread);
    end
    
    condLabels(condIdx) = sprintf('Std %.0f, H %.3f', avgStdVal, avgHVal);
end
condLabelsCell = cellstr(condLabels);

legendLabels = arrayfun(@(pid) sprintf('P%03d', pid), participantIDs, 'UniformOutput', false);
cmap = lines(length(participantIDs));

%% --- Grouped Bar Charts: Raw vs Fit-based Accuracy, side by side ---
f = figure('Position', [100 100 1600 700]);

subplot(1,2,1);
bh1 = bar(accMatrixRaw, 'grouped');
for i = 1:length(bh1), bh1(i).FaceColor = cmap(i,:); end
hold on;
plot(1:nConditions, nanmean(accMatrixRaw,2), 'ko', 'MarkerFaceColor','k','MarkerSize',10,'LineWidth',2);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ylim([0 1]); grid on;
ylabel('Raw Overall Accuracy', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Condition (Std / Hazard)', 'FontSize', 18, 'FontWeight', 'bold');
title('Raw Accuracy (model-free)', 'FontSize', 20);

subplot(1,2,2);
bh2 = bar(accMatrixFit, 'grouped');
for i = 1:length(bh2), bh2(i).FaceColor = cmap(i,:); end
hold on;
plot(1:nConditions, nanmean(accMatrixFit,2), 'ko', 'MarkerFaceColor','k','MarkerSize',10,'LineWidth',2);
ax = gca; ax.FontSize = 14; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ylim([0 1]); grid on;
ylabel('Effective Accuracy (logistic asymptote)', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Condition (Std / Hazard)', 'FontSize', 18, 'FontWeight', 'bold');
title('Logistic Fit Asymptote', 'FontSize', 20);

legend(bh2, legendLabels, 'Location', 'eastoutside', 'FontSize', 11);

% saveas(f, fullfile(figurePath, 'individual_accuracy_raw_vs_fit_comparison.png'));

%% --- Heatmaps: Raw vs Fit-based Accuracy, side by side ---
f2 = figure('Position', [100 100 1600 700]);

subplot(1,2,1);
imagesc(accMatrixRaw');
colormap(parula); cb = colorbar; cb.Label.String = 'Raw Accuracy'; clim([0 1]);
ax = gca; ax.FontSize = 12; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ax.YTick = 1:length(participantIDs); ax.YTickLabel = legendLabels;
xlabel('Condition', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Participant', 'FontSize', 16, 'FontWeight', 'bold');
title('Raw Overall Accuracy', 'FontSize', 20);

subplot(1,2,2);
imagesc(accMatrixFit');
colormap(parula); cb = colorbar; cb.Label.String = 'Effective Accuracy'; clim([0 1]);
ax = gca; ax.FontSize = 12; ax.XTick = 1:nConditions; ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 30;
ax.YTick = 1:length(participantIDs); ax.YTickLabel = legendLabels;
xlabel('Condition', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Participant', 'FontSize', 16, 'FontWeight', 'bold');
title('Logistic Fit Asymptote', 'FontSize', 20);

% saveas(f2, fullfile(figurePath, 'heatmap_accuracy_raw_vs_fit_comparison.png'));

%% --- Local Functions ---
% (Must appear at the end of the script file, per MATLAB script-function rules)

function [fits_, sems_] = fitLogisticCapped(data, lumode, beta1Cap)
% Wrapper around logist_setup/fmincon that replicates logist_fit.m's
% fitting mechanics exactly, but caps the beta1 (slope-on-time)
% parameter to [-beta1Cap, beta1Cap] to prevent quasi-complete-
% separation solutions. logist_fit.m/logist_setup.m are left untouched
% since they're shared toolbox files used elsewhere.
%
% Assumes the standard single-predictor design used in this script:
% fitData = [ones(n,1), framesSinceSwitch, correct], so row 1 of the
% beta bounds (from logist_setup) is the intercept and row 2 is the
% slope coefficient on framesSinceSwitch -- only row 2 is capped here.

[fun, data_, inits] = logist_setup(data, lumode);

% Cap the slope coefficient (row 2) bounds
inits(2,2) = -beta1Cap;
inits(2,3) =  beta1Cap;

% Clip the initial guess into the new bounds too, in case logist_setup's
% OLS-based first guess already falls outside them
inits(2,1) = min(max(inits(2,1), -beta1Cap), beta1Cap);

[fits_, ~, ~, ~, ~, ~, H] = fmincon('logist_err', inits(:,1), [], [], [], [], ...
    inits(:,2), inits(:,3), [], ...
    optimset('LargeScale', 'off', 'Algorithm', 'active-set', ...
    'Display', 'off', 'Diagnostics', 'off'), ...
    fun, data_);

sems_ = sqrt(diag(-((-H)^(-1))));
end