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

% NEW: Dimensions: [Condition x Participant x (lowerAsySEM, upperAsySEM)]
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
            % Fit a 4-parameter logistic function with lower and upper bounds ('lu2')
            % Now requesting sems (standard errors) as second output, which
            % gives a direct measure of how well-constrained each asymptote
            % estimate is
            [fits, sems, ~, ~, ~] = logist_fit(fitData, 'lu2');
            
            % Extract fitted parameters
            beta0 = fits(1);          % Intercept
            beta1 = fits(2);          % Slope coefficient
            lowerAsy = fits(end-1);   % Lower asymptote (chance level)
            upperAsy = 1 - fits(end); % Upper asymptote (maximum accuracy achieved)
            
            % SEMs are in the same order as fits: [betas..., gamma_sem, lambda_sem]
            % gamma_sem = SEM of lowerAsy; lambda_sem = SEM of fits(end),
            % which is also the SEM of upperAsy (linear transform 1-x
            % doesn't change the magnitude of the SEM)
            lowerAsySEM = sems(end-1);
            upperAsySEM = sems(end);
            
            % --- Diagnostic Checks: Flag suspicious/non-converged fits ---
            % Count actual number of source switches in this block (used to
            % assess whether there was enough data to constrain the fit)
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
            % Flag based on actual estimation uncertainty (SEM) rather than
            % just switch count -- a direct measure of how well-constrained
            % this specific fit is
            semThreshold = 0.1;
            if lowerAsySEM > semThreshold || upperAsySEM > semThreshold
                flagReasons{end+1} = sprintf('high asymptote uncertainty (lowerSEM=%.3f, upperSEM=%.3f)', ...
                    lowerAsySEM, upperAsySEM);
            end
            % Updated boundary check: bounds widened from [0,0.2]/[0.8,1]
            % to [0,0.5]/[0.5,1] in logist_setup.m, so check against 0.5 now
            if abs(lowerAsy - 0.5) < 1e-6 || abs(upperAsy - 0.5) < 1e-6
                flagReasons{end+1} = 'hit logist_fit bound (0.5)';
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
            
            % Pick whichever asymptote is closer to the model-free raw
            % accuracy for this block, as a principled way to disambiguate
            % which one represents genuine "settled" performance rather
            % than relying on the (often noisy/near-zero) sign of beta1
            localOverallAcc = allOverallAcc(condIdx, p);
            if abs(lowerAsy - localOverallAcc) < abs(upperAsy - localOverallAcc)
                effAcc = lowerAsy;
            else
                effAcc = upperAsy;
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
            % Create an invisible figure to prevent pop-up spam during batch processing
            f = figure('Visible', 'off'); 
            
            % Convert frames to seconds (assuming 50ms / 20Hz frame rate)
            plotTimeSeconds = (1:maxX) * 0.05;
            
            % Generate predicted values for the smoothed logistic fit line
            predMatrix = [ones(maxX, 1), (1:maxX)'];
            smoothedAcc = logist_valLU2(fits, predMatrix);
            
            % Plot Raw Data (Thicker for visibility)
            % Compute the average accuracy per frame interval just for plotting
            avgAcc = arrayfun(@(n) mean(trialIsCorrect(trialFramesSince == n)), 1:maxX);
            plot(plotTimeSeconds, avgAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2);
            hold on;
            
            % Plot Fit Line (Very thick red line)
            plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 4);
            
            % Asymptote and Tangent Line Logic
            % Calculate mathematical properties of the curve for visualization
            range = upperAsy - lowerAsy;
            
            % Midpoint (inflection point) of the logistic curve
            midpointFrames = -beta0 / beta1;
            midpointSeconds = midpointFrames * 0.05;
            
            % Calculate the maximum slope (rate of learning) in accuracy per second
            maxSlopePerSec = ((beta1 * range) / 4) / 0.05;
            
            % Y-value at the inflection point
            yMid = (lowerAsy + upperAsy) / 2;
            
            % Plot the tangent line representing the maximum slope at the midpoint
            if midpointSeconds > 0 && midpointSeconds < 30
                tX = [midpointSeconds - 1, midpointSeconds + 1];
                tY = maxSlopePerSec * (tX - midpointSeconds) + yMid;
                plot(tX, tY, 'b-', 'LineWidth', 3); 
                plot(midpointSeconds, yMid, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
            end

            % --- Poster Formatting ---
            ax = gca;
            ax.FontSize = 20;           % Tick label size
            ax.LineWidth = 2;          % Axis line thickness
            ax.TickLength = [0.02 0.02]; % Larger ticks
            xlim([0 30]);              % Requested X limit (max 30 seconds)
            ylim([0 1]);               % Accuracy ranges from 0 to 1
            grid on;
            
            xlabel('Seconds Since Source Switch', 'FontSize', 24, 'FontWeight', 'bold');
            ylabel('Accuracy', 'FontSize', 24, 'FontWeight', 'bold');
            title(sprintf('P%s Std %.3f H %.3f: Logistic Fit', pidStr, stdVal, hVal), 'FontSize', 30);
            
            % Uncomment to save individual plots
            %saveas(f, fullfile(figurePath, sprintf('p%s_std%d_h%d_logistic_fit.png', pidStr, stdIdx, hIdx)));
            close(f);
        catch ME
            % Error handling to prevent one bad fit from stopping the entire script
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
    
    % Optional: save to CSV for easier review/sorting outside MATLAB
    % writetable(diagTable, fullfile(figurePath, 'fit_diagnostics.csv'));
else
    fprintf('\n=== No suspicious fits flagged ===\n');
end

%% --- Generate Group Average Figures ---
% Loop through conditions (std x hazard combinations) to create cross-participant
% aggregate visualisations. Loop is over condition index (1-6), not raw block
% number, since block order is randomized and no longer meaningful.
%
% IMPORTANT: The group curve is now generated by fitting ONE logistic model
% to the POOLED trial-level data from all participants in this condition,
% rather than averaging individually-fit parameters. Averaging parameters
% directly breaks down badly when some individual fits are degenerate
% (e.g., quasi-complete separation producing beta1~50), since mixing those
% with well-behaved fits in parameter space produces a curve that
% represents no one. Pooling trials first also directly fixes the
% low-switch-count instability, since pooling across 10 participants gives
% far more switches to constrain the fit than any single participant alone.

groupBeta0s = zeros(1, nConditions);
groupBeta1s = zeros(1, nConditions);
groupLowerAsys = zeros(1, nConditions);
groupUpperAsys = zeros(1, nConditions);
pooledOverallAccs = zeros(1, nConditions);

for condIdx = 1:nConditions
    
    % Recover the std/hazard indices that this condition corresponds to
    stdIdx = floor((condIdx - 1) / nHazard) + 1;
    hIdx = mod(condIdx - 1, nHazard) + 1;
    
    % Recover the actual std/hazard values (averaged across participants, should
    % be consistent within a condition, but averaged for safety)
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    % Skip if no data was accumulated for this condition
    if isempty(pooledFitData{condIdx}), continue; end
    
    combinedData = pooledFitData{condIdx};
    nPooledTrials = size(combinedData, 1);
    nPooledSwitches = sum(diff(combinedData(:,2)) < 0) + size(combinedData,1)*0; % placeholder, see note below
    
    % Calculate the group-average RAW accuracy across valid frames
    % (this part is unchanged -- it's a genuine data average, not a
    % parameter average, so it was never the problem)
    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, :, :), 2));
    lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
    groupRawAcc = groupRawAcc(1:lastValidIdx);
    numFrames = length(groupRawAcc);
    plotTimeSeconds = (1:numFrames) * 0.05;
    
    try
        % Fit ONE logistic model to the pooled trial-level data
        [groupFits, groupSems] = logist_fit(combinedData, 'lu2');
        
        groupBeta0 = groupFits(1);
        groupBeta1 = groupFits(2);
        groupLowerAsy = groupFits(end-1);
        groupUpperAsy = 1 - groupFits(end);
        
        % Pick whichever asymptote is closer to the overall pooled raw
        % accuracy, same principled logic as the individual-level fits
        pooledOverallAcc = mean(combinedData(:,end));
        if abs(groupLowerAsy - pooledOverallAcc) < abs(groupUpperAsy - pooledOverallAcc)
            groupEffAcc = groupLowerAsy;
        else
            groupEffAcc = groupUpperAsy;
        end

        groupBeta0s(1, condIdx) = groupBeta0;
        groupBeta1s(1, condIdx) = groupBeta1;
        groupLowerAsys(1, condIdx) = groupLowerAsy;
        groupUpperAsys(1, condIdx) = groupUpperAsy;
        pooledOverallAccs(1, condIdx) = pooledOverallAcc;

        % Generate predicted values for the pooled fit line
        predMatrix = [ones(numFrames, 1), (1:numFrames)'];
        smoothedAcc = logist_valLU2(groupFits, predMatrix);
        
        % Compute inflection point, max slope, dynamic range for the pooled curve
        groupRange = groupEffAcc - groupLowerAsy;
        midpointSec = (-groupBeta0 / groupBeta1) * 0.05;
        maxSlopePerSec = ((groupBeta1 * groupRange) / 4) / 0.05;
        yMidGroup = (groupLowerAsy + groupEffAcc) / 2;
        
        f = figure;
        % Plot Group Raw Data background line (unchanged -- genuine data average)
        plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
        hold on;
        % Plot Pooled Fit Line
        plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);
        
        % Plot horizontal line for the upper asymptote (effective maximum accuracy)
        yline(groupEffAcc, '--k', 'LineWidth', 2.5);
        text(28, groupEffAcc - 0.05, sprintf('Asy: %.2f', groupEffAcc), ...
            'FontWeight', 'bold', 'FontSize', 22, 'HorizontalAlignment', 'right');
        
        % Midpoint / Slope representation
        if midpointSec > 0 && midpointSec < 30
            tX = [midpointSec - 1.5, midpointSec + 1.5];
            tY = maxSlopePerSec * (tX - midpointSec) + yMidGroup;
            plot(tX, tY, 'b-', 'LineWidth', 4);
            plot(midpointSec, yMidGroup, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 12);
            text(midpointSec + 0.5, yMidGroup - 0.08, sprintf('Slope: %.2f acc/sec', maxSlopePerSec), ...
                'Color', 'b', 'FontWeight', 'bold', 'FontSize', 20);
        end
        
        % --- Poster Formatting (Maximum Legibility) ---
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
        
        % Save as PDF (Commented out by default)
        % saveas(f, fullfile(figurePath, sprintf('POSTER_Average_Std%d_H%d.png', stdIdx, hIdx)));
        
    catch ME
        fprintf('Pooled group fit failed for Std %.3f H %.3f: %s\n', avgStdVal, avgHVal, ME.message);
    end
end

%% --- Individual Participant Accuracy Comparison (Outlier Check) ---
% Compare raw overall accuracy (model-free, robust) against logistic
% effAcc (captures detection-speed-independent ceiling performance) to
% distinguish genuine behavioral outliers from fitting artifacts.

accMatrixFit = squeeze(allParams(:, :, 5));  % [nConditions x nParticipants] - logistic effAcc
accMatrixRaw = allOverallAcc;                 % [nConditions x nParticipants] - raw overall accuracy

% Build condition labels using the actual std/hazard values.
% Using a single-line label (no embedded \n) to avoid a MATLAB
% tick-rendering issue where multi-line labels for 6 closely-spaced ticks
% were being torn apart and displayed as if std/hazard were separate,
% unpaired entries.
condLabels = strings(nConditions, 1);
for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    % Sanity check: confirm this condition's std/hazard values are
    % consistent across all participants (i.e., condIdx reliably maps to
    % the same real condition for everyone). Flag if not.
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

% Save figure (commented out by default)
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

% Save figure (commented out by default)
% saveas(f2, fullfile(figurePath, 'heatmap_accuracy_raw_vs_fit_comparison.png'));