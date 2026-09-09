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

% Diagnostic log to flag suspicious/non-converged logistic fits
% Will be converted to a table at the end for easy sorting/inspection
diagLog = struct('participant', {}, 'condIdx', {}, 'stdVal', {}, 'hVal', {}, ...
    'nTrials', {}, 'nSwitches', {}, 'beta1', {}, 'lowerAsy', {}, 'upperAsy', {}, ...
    'effAcc', {}, 'flagReason', {});

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
        
        % NEW: pull actual std/hazard values for labeling
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

        allOverallAcc(condIdx, p) = mean(trialIsCorrect);
        
        % Calculate and store the mean raw accuracy for every frame interval
        maxX = max(trialFramesSince);
        for n = 1:maxX
            allRawAcc(condIdx, p, n) = mean(trialIsCorrect(trialFramesSince == n));
        end
        
        % Logistic Fit
        % Prepare matrix for logistic regression: [Intercept, Predictor(frames), Response(accuracy)]
        fitData = [ones(nTrials, 1), trialFramesSince, trialIsCorrect];
        try
            % Fit a 4-parameter logistic function with lower and upper bounds ('lu2')
            [fits, ~, ~, ~, ~] = logist_fit(fitData, 'lu2');
            
            % Extract fitted parameters
            beta0 = fits(1);          % Intercept
            beta1 = fits(2);          % Slope coefficient
            lowerAsy = fits(end-1);   % Lower asymptote (chance level)
            upperAsy = 1 - fits(end); % Upper asymptote (maximum accuracy achieved)

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
            if lowerAsy < 1e-6
                flagReasons{end+1} = 'lowerAsy == 0';
            end
            if upperAsy > 0.999
                flagReasons{end+1} = 'upperAsy == 1';
            end
            % Flag suspicious "boundary" values seen repeatedly (e.g., 0.8),
            % which suggest the optimizer returned a default/init value
            % rather than a genuine converged fit
            if abs(lowerAsy - 0.8) < 1e-6 || abs(upperAsy - 0.8) < 1e-6
                flagReasons{end+1} = 'suspected boundary value (0.8)';
            end

            if ~isempty(flagReasons)
                reasonStr = strjoin(flagReasons, '; ');
                fprintf('FLAG: P%s Std %.0f H %.3f (nTrials=%d, nSwitches=%d, beta1=%.4f, lowerAsy=%.3f, upperAsy=%.3f) -- %s\n', ...
                    pidStr, stdVal, hVal, nTrials, nSwitches, beta1, lowerAsy, upperAsy, reasonStr);

                diagLog(end+1) = struct('participant', pidStr, 'condIdx', condIdx, ...
                    'stdVal', stdVal, 'hVal', hVal, 'nTrials', nTrials, 'nSwitches', nSwitches, ...
                    'beta1', beta1, 'lowerAsy', lowerAsy, 'upperAsy', upperAsy, ...
                    'effAcc', NaN, 'flagReason', reasonStr); %#ok<SAGROW>
            end

            % General rule: if the fitted slope is negative, accuracy is decreasing
            % over time since switch (inverted relationship), so the effective
            % "ceiling" accuracy is actually the lower asymptote in that case.
            if beta1 < 0
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
for condIdx = 1:nConditions
    
    % Recover the std/hazard indices that this condition corresponds to
    stdIdx = floor((condIdx - 1) / nHazard) + 1;
    hIdx = mod(condIdx - 1, nHazard) + 1;
    
    % Recover the actual std/hazard values (averaged across participants, should
    % be consistent within a condition, but averaged for safety)
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    % Calculate group means for each parameter, ignoring NaN values (failed fits)
    avgBeta0    = nanmean(allParams(condIdx, :, 1));
    avgBeta1    = nanmean(allParams(condIdx, :, 2));
    avgLowerAsy = nanmean(allParams(condIdx, :, 3));
    avgEffAcc   = nanmean(allParams(condIdx, :, 5)); 
    
    % Skip if no valid parameters were generated for this condition
    if isnan(avgBeta0), continue; end
    
    % Reconstruct the average fits array for generating the group logistic line
    avgFits = [avgBeta0; avgBeta1; avgLowerAsy; (1 - avgEffAcc)];
    
    % Calculate the group-average raw accuracy across valid frames
    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, :, :), 2));
    lastValidIdx = find(~isnan(groupRawAcc), 1, 'last');
    groupRawAcc = groupRawAcc(1:lastValidIdx);
    
    % Generate coordinates for plotting the averaged model line
    numFrames = length(groupRawAcc);
    plotTimeSeconds = (1:numFrames) * 0.05;
    predMatrix = [ones(numFrames, 1), (1:numFrames)'];
    smoothedAcc = logist_valLU2(avgFits, predMatrix);
    
    % Compute inflection point, max slope, and dynamic range for the group average curve
    avgRange = avgEffAcc - avgLowerAsy;
    midpointSec = (-avgBeta0 / avgBeta1) * 0.05;
    maxSlopePerSec = ((avgBeta1 * avgRange) / 4) / 0.05;
    yMidGroup = (avgLowerAsy + avgEffAcc) / 2;

    f = figure;
    % Plot Group Raw Data background line
    plot(plotTimeSeconds, groupRawAcc, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    hold on;
    % Plot Group Fit Line
    plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);
    
    % Plot horizontal line for the upper asymptote (effective maximum accuracy)
    yline(avgEffAcc, '--k', 'LineWidth', 2.5);
    text(28, avgEffAcc - 0.05, sprintf('Asy: %.2f', avgEffAcc), ...
        'FontWeight', 'bold', 'FontSize', 22, 'HorizontalAlignment', 'right');
    
    % Midpoint / Slope representation
    if midpointSec > 0 && midpointSec < 30
        % Draw the tangent line at the inflection point to show maximum learning rate
        tX = [midpointSec - 1.5, midpointSec + 1.5];
        tY = maxSlopePerSec * (tX - midpointSec) + yMidGroup;
        plot(tX, tY, 'b-', 'LineWidth', 4);
        
        % Mark the midpoint and display the slope value
        plot(midpointSec, yMidGroup, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 12);
        text(midpointSec + 0.5, yMidGroup - 0.08, sprintf('Slope: %.2f acc/sec', maxSlopePerSec), ...
            'Color', 'b', 'FontWeight', 'bold', 'FontSize', 20);
    end

    % --- Poster Formatting (Maximum Legibility) ---
    ax = gca;
    ax.FontSize = 24;          % Larger ticks for group plot
    ax.LineWidth = 3;          % Bold axis lines
    ax.TickLength = [0.02 0.02];
    
    xlim([0 30]);              % Requested X limit
    ylim([0 1]);               % Constrain Y-axis to 0-100% accuracy range
    grid on;
    
    xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
    ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
    
    % Title using actual std/hazard values
    title(sprintf('Std %.3f, Hazard %.3f', avgStdVal, avgHVal), 'FontSize', 36)
    
    % Save as PDF (Commented out by default)
    % saveas(f, fullfile(figurePath, sprintf('POSTER_Average_Std%d_H%d.png', stdIdx, hIdx)));
end

close all

%% --- Individual Participant Accuracy Comparison (Outlier Check) ---
% Compare raw overall accuracy (model-free, robust) against logistic
% effAcc (captures detection-speed-independent ceiling performance) to
% distinguish genuine behavioral outliers from fitting artifacts.

accMatrixFit = squeeze(allParams(:, :, 5));  % [nConditions x nParticipants] - logistic effAcc
accMatrixRaw = allOverallAcc;                 % [nConditions x nParticipants] - raw overall accuracy

% Build condition labels using the actual std/hazard values.
% NOTE: Using a single-line label (no embedded \n) to avoid a MATLAB
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