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

% NEW: Store actual std/hazard values (not just indices) for labeling plots later
% Dimensions: [Condition x Participant x (stdVal, hazardVal)]
allActualVals = NaN(nConditions, length(participantIDs), 2);

%% --- Individual Participant Loop ---
% Minimum number of data points (beliefs/epochs) required at a given
% "frames since switch" value for that value to be used in fitting
minPoints = 4;

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
        
        % Determine the last "frames since switch" value that still has at least
        % minPoints data points (i.e. at least minPoints beliefs lasted that long).
        % Counts can only decrease as frames-since-switch increases, so everything
        % beyond this point has fewer than minPoints data points and is cut.
        framesCounts = accumarray(trialFramesSince, 1);
        maxX = find(framesCounts >= minPoints, 1, 'last');
        if isempty(maxX)
            fprintf('Skipping P%s Std %.3f H %.3f: no time point has >= %d data points\n', ...
                pidStr, stdVal, hVal, minPoints);
            continue;
        end
        
        % Calculate and store the mean raw accuracy for every valid frame interval
        for n = 1:maxX
            allRawAcc(condIdx, p, n) = mean(trialIsCorrect(trialFramesSince == n));
        end
        
        % Logistic Fit
        % Only use trials at or before the cutoff time point
        keepIdx = trialFramesSince <= maxX;
        % Prepare matrix for logistic regression: [Intercept, Predictor(frames), Response(accuracy)]
        fitData = [ones(sum(keepIdx), 1), trialFramesSince(keepIdx), trialIsCorrect(keepIdx)];
        try
            % Fit a 4-parameter logistic function with lower and upper bounds ('lu2')
            [fits, ~, ~, ~, ~] = logist_fit(fitData, 'lu2');
            
            % Extract fitted parameters
            beta0 = fits(1);          % Intercept
            beta1 = fits(2);          % Slope coefficient
            lowerAsy = fits(end-1);   % Lower asymptote (chance level)
            upperAsy = 1 - fits(end); % Upper asymptote (maximum accuracy achieved)
            
            % General rule: if the fitted slope is negative, accuracy is decreasing
            % over time since switch (inverted relationship), so the effective
            % "ceiling" accuracy is actually the lower asymptote in that case.
            if beta1 < 0
                effAcc = lowerAsy;
            else
                effAcc = upperAsy;
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

%% --- DIAGNOSTIC: are fits hitting bounds / degenerate? ---
fprintf('\n%-6s %-6s %8s %8s %9s %8s %8s %6s\n', ...
    'Cond','PID','beta0','beta1','t50(s)','lower','upper','FLAG');
for c = 1:nConditions
    for p = 1:length(participantIDs)
        b0 = allParams(c,p,1); b1 = allParams(c,p,2);
        lo = allParams(c,p,3); up = allParams(c,p,4);
        if isnan(b0), continue; end
        t50 = (-b0/b1)*0.05;
        flag = '';
        if abs(b1) > 5,            flag = [flag 'STEEP ']; end
        if b1 < 0,                 flag = [flag 'NEG ']; end
        if up > 0.999,             flag = [flag 'UP@1 ']; end
        if lo < 0.001,             flag = [flag 'LO@0 ']; end
        if lo > 0.399,             flag = [flag 'LO@BND ']; end
        if t50 < 0 || t50 > 15,    flag = [flag 'MIDOFF ']; end
        fprintf('%-6d %-6d %8.3f %8.3f %9.2f %8.3f %8.3f  %s\n', ...
            c, participantIDs(p), b0, b1, t50, lo, up, flag);
    end
    fprintf('  -> MEAN b0=%.3f b1=%.3f  => group t50 = %.2f s  (median indiv t50 = %.2f s)\n\n', ...
        nanmean(allParams(c,:,1)), nanmean(allParams(c,:,2)), ...
        (-nanmean(allParams(c,:,1))/nanmean(allParams(c,:,2)))*0.05, ...
        nanmedian((-allParams(c,:,1)./allParams(c,:,2))*0.05));
end

%% --- Generate Group Average Figures ---
% Loop through conditions (std x hazard combinations) to create cross-participant
% aggregate visualisations. Loop is over condition index (1-6), not raw block
% number, since block order is randomized and no longer meaningful.

% Criteria for excluding degenerate fits from the group average
minSlope = 0.01;     % exclude fits with beta1 <= this (negative or ~flat)
maxMidSec = 15;      % exclude fits whose midpoint (t50) is outside 0 to this many seconds

for condIdx = 1:nConditions
    
    % Recover the std/hazard indices that this condition corresponds to
    stdIdx = floor((condIdx - 1) / nHazard) + 1;
    hIdx = mod(condIdx - 1, nHazard) + 1;
    
    % Recover the actual std/hazard values (averaged across participants, should
    % be consistent within a condition, but averaged for safety)
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    
    % Identify non-degenerate fits for this condition (logical row vector over participants)
    thisB0 = allParams(condIdx, :, 1);
    thisB1 = allParams(condIdx, :, 2);
    thisT50 = (-thisB0 ./ thisB1) * 0.05;
    validFit = ~isnan(thisB0) & (thisB1 > minSlope) & (thisT50 >= 0) & (thisT50 <= maxMidSec);
    fprintf('Cond %d: using %d of %d fits in group average\n', ...
        condIdx, sum(validFit), sum(~isnan(thisB0)));
    
    % Calculate group means for each parameter using only valid fits
    avgBeta0    = nanmean(allParams(condIdx, validFit, 1));
    avgBeta1    = nanmean(allParams(condIdx, validFit, 2));
    avgLowerAsy = nanmean(allParams(condIdx, validFit, 3));
    avgEffAcc   = nanmean(allParams(condIdx, validFit, 5)); 
    
    % Skip if no valid parameters were generated for this condition
    if isnan(avgBeta0), continue; end
    
    % Reconstruct the average fits array for generating the group logistic line
    avgFits = [avgBeta0; avgBeta1; avgLowerAsy; (1 - avgEffAcc)];
    
    % Calculate the group-average raw accuracy across valid frames (valid fits only)
    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, validFit, :), 2));
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
    hold on;
    % Plot each individual participant's logistic fit faintly in the background
    % (all the same color) to show variability. All fits are shown here,
    % including the ones excluded from the average.
    for p = 1:length(participantIDs)
        if isnan(allParams(condIdx, p, 1)), continue; end
        indivFits = [allParams(condIdx, p, 1); ...
                     allParams(condIdx, p, 2); ...
                     allParams(condIdx, p, 3); ...
                     1 - allParams(condIdx, p, 4)];
        indivSmoothedAcc = logist_valLU2(indivFits, predMatrix);
        plot(plotTimeSeconds, indivSmoothedAcc, 'Color', [0.7 0.5 0.5], 'LineWidth', 0.5);
    end
    % Plot Group Raw Data background line
    plot(plotTimeSeconds, groupRawAcc, 'Color', [0 0 0], 'LineWidth', 1.5);
    % Plot Group Fit Line
    plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);
    
    % Plot horizontal line for the upper asymptote (effective maximum accuracy)
    yline(avgEffAcc, '--k', 'LineWidth', 2.5);
    text(14.5, avgEffAcc - 0.05, sprintf('Asy: %.2f', avgEffAcc), ...
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
    
    xlim([0 15]);              % X limit (max 15 seconds)
    ylim([0 1]);               % Constrain Y-axis to 0-100% accuracy range
    grid on;
    
    xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
    ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
    
    % Title using actual std/hazard values
    title(sprintf('Std %.3f, Hazard %.3f', avgStdVal, avgHVal), 'FontSize', 36)
    
    % Save as PDF (Commented out by default)
    % saveas(f, fullfile(figurePath, sprintf('POSTER_Average_Std%d_H%d.png', stdIdx, hIdx)));
end

%% --- Individual Participant Accuracy Comparison (Outlier Check) ---
% Visualize each participant's effective accuracy across all std/hazard
% conditions to help identify any outlier participants or conditions.

% Extract effective accuracy for all participants and conditions
% Rows = conditions, Columns = participants
accMatrix = squeeze(allParams(:, :, 5)); % [nConditions x nParticipants]

% Build condition labels using the actual std/hazard values (averaged
% across participants, should be consistent within a condition)
condLabels = strings(nConditions, 1);
for condIdx = 1:nConditions
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));
    condLabels(condIdx) = sprintf('Std %.3f\nH %.3f', avgStdVal, avgHVal);
end
condLabelsCell = cellstr(condLabels);

% Build legend labels for each participant (P001, P002, ...)
legendLabels = arrayfun(@(pid) sprintf('P%03d', pid), participantIDs, 'UniformOutput', false);

%% --- Option 1: Grouped Bar Chart (one group per condition, one bar per participant) ---
f = figure;
bh = bar(accMatrix, 'grouped');

% Assign distinct colors per participant for easy tracking across conditions
cmap = lines(length(participantIDs));
for i = 1:length(bh)
    bh(i).FaceColor = cmap(i, :);
end

hold on;

% Overlay the group mean per condition as a black marker for quick reference
groupMeans = nanmean(accMatrix, 2);
meanHandle = plot(1:nConditions, groupMeans, 'ko', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 10, 'LineWidth', 2);

% --- Formatting ---
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 2;
ax.XTick = 1:nConditions;
ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 0;
ylim([0 1]);
ylabel('Effective Accuracy', 'FontSize', 20, 'FontWeight', 'bold');
xlabel('Condition (Std / Hazard)', 'FontSize', 20, 'FontWeight', 'bold');
title('Individual Participant Accuracy by Condition', 'FontSize', 24);
grid on;

legend([bh, meanHandle], [legendLabels, {'Group Mean'}], ...
    'Location', 'eastoutside', 'FontSize', 12);

% Save figure (commented out by default)
% saveas(f, fullfile(figurePath, 'individual_accuracy_by_condition_bar.png'));

%% --- Option 2: Heatmap (Participant x Condition) ---
% Often better for outlier detection: a participant row that's a
% consistently different color stands out, and a condition column that's
% uniformly off stands out separately.
f2 = figure;
imagesc(accMatrix'); % transpose so rows = participants, columns = conditions
colormap(parula);
cb = colorbar;
cb.Label.String = 'Effective Accuracy';
cb.Label.FontSize = 16;
cb.Label.FontWeight = 'bold';
clim([0 1]);

ax = gca;
ax.FontSize = 14;
ax.XTick = 1:nConditions;
ax.XTickLabel = condLabelsCell;
ax.XTickLabelRotation = 0;
ax.YTick = 1:length(participantIDs);
ax.YTickLabel = legendLabels;

xlabel('Condition (Std / Hazard)', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Participant', 'FontSize', 20, 'FontWeight', 'bold');
title('Heatmap of Effective Accuracy: Participant x Condition', 'FontSize', 24);

% Save figure (commented out by default)
% saveas(f2, fullfile(figurePath, 'heatmap_accuracy_participant_by_condition.png'));