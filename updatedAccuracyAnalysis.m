%% --- Environment Setup ---
% Define the participants to include in the analysis
participantIDs = 1:31;%[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

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
% Fixed maximum time (seconds since switch) used for fitting and plotting
maxSeconds = 10;
maxFrames = round(maxSeconds / 0.05);   % 200 frames at 20 Hz

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
        
        % Cutoff is fixed at maxFrames (10 s) regardless of the data
        maxX = maxFrames;
        
        % Calculate and store the mean raw accuracy for every frame interval up to the cutoff
        % (frames with no data are NaN)
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
for condIdx = 1:nConditions

    % Recover the std/hazard indices that this condition corresponds to
    stdIdx = floor((condIdx - 1) / nHazard) + 1;
    hIdx = mod(condIdx - 1, nHazard) + 1;

    % Recover the actual std/hazard values (averaged across participants, should
    % be consistent within a condition, but averaged for safety)
    avgStdVal = nanmean(allActualVals(condIdx, :, 1));
    avgHVal = nanmean(allActualVals(condIdx, :, 2));

    % Calculate group medians for each parameter, ignoring NaN values (failed fits)
    avgBeta0    = nanmedian(allParams(condIdx, :, 1));
    avgBeta1    = nanmedian(allParams(condIdx, :, 2));
    avgLowerAsy = nanmedian(allParams(condIdx, :, 3));
    avgEffAcc   = nanmedian(allParams(condIdx, :, 5)); 

    % Skip if no valid parameters were generated for this condition
    if isnan(avgBeta0), continue; end

    % Reconstruct the average fits array for generating the group logistic line
    avgFits = [avgBeta0; avgBeta1; avgLowerAsy; (1 - avgEffAcc)];

    % Group-average raw accuracy over the fixed window (always maxFrames long = 10 s)
    groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, :, 1:maxFrames), 2));

    % Generate coordinates for plotting the averaged model line
    numFrames = maxFrames;
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
    % (all the same color) to show variability
    for p = 1:length(participantIDs)
        if isnan(allParams(condIdx, p, 1)), continue; end
        indivFits = [allParams(condIdx, p, 1); ...
                     allParams(condIdx, p, 2); ...
                     allParams(condIdx, p, 3); ...
                     1 - allParams(condIdx, p, 4)];
        indivSmoothedAcc = logist_valLU2(indivFits, predMatrix);
        plot(plotTimeSeconds, indivSmoothedAcc, 'Color', [0.9 0.5 0.5], 'LineWidth', 0.5);
    end
    % Plot Group Raw Data background line
    plot(plotTimeSeconds, groupRawAcc, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
    % Plot Group Fit Line
    plot(plotTimeSeconds, smoothedAcc, 'r-', 'LineWidth', 5);

    % Plot horizontal line for the upper asymptote (effective maximum accuracy)
    yline(avgEffAcc, '--k', 'LineWidth', 2.5);
    text(9.8, avgEffAcc - 0.05, sprintf('Asy: %.2f', avgEffAcc), ...
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

    xlim([0 10]);              % X limit (max 15 seconds)
    ylim([0 1]);               % Constrain Y-axis to 0-100% accuracy range
    grid on;

    xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
    ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');

    % Title using actual std/hazard values
    title(sprintf('Std %.3f, Hazard %.3f', avgStdVal, avgHVal), 'FontSize', 36)

    % Save as PNG (Commented out by default)
    saveas(f, fullfile(figurePath, sprintf('SFN_accuracy_condition_Std%d_H%d.png', stdIdx, hIdx)));
end

% --- Individual Participant Accuracy Comparison (Outlier Check) ---
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

%% --- Group Fits by Global Strategy (per condition) ---
% % Choose which global strategy classification to use
% strategyType = 'opposite';  % 'central' or 'opposite'
% 
% switch strategyType
%     case 'central'
%         strategyFile = 'centralFixationStrategies.mat';
%     case 'opposite'
%         strategyFile = 'oppositeFixationStrategies.mat';
%     otherwise
%         error('Unknown strategy type: %s', strategyType);
% end
% 
% S = load(fullfile('/Users/janebaker/Desktop/triangles/taskData', strategyFile));
% 
% strategyMatrix = S.strategyMatrix;   % [nConditions x nParticipants], 1=Low, 2=Moderate, 3=High, NaN=excluded
% assert(isequal(S.strategyParticipantIDs(:), participantIDs(:)), ...
%     'Participant order differs between strategy and accuracy scripts');
% assert(size(strategyMatrix,1) == nConditions, 'strategyMatrix rows must equal nConditions');
% 
% nStrategies = 3;
% stratShort  = {'Low', 'Moderate', 'High'};
% stratColors = [0.20 0.55 0.85; 0.90 0.60 0.10; 0.75 0.10 0.15];  % same as heatmap
% 
% showIndiv  = true;    % faint individual participant fits
% showRaw    = false;   % faint raw accuracy per strategy (noisy)
% maxTangentRise = 0.15; % tangent never rises/falls more than this (accuracy units)
% 
% plotTimeSeconds = (1:maxFrames) * 0.05;
% predMatrix = [ones(maxFrames, 1), (1:maxFrames)'];
% 
% for condIdx = 1:nConditions
% 
%     avgStdVal = nanmean(allActualVals(condIdx, :, 1));
%     avgHVal   = nanmean(allActualVals(condIdx, :, 2));
% 
%     % ---- Compute group parameters for each strategy ----
%     idxS = cell(nStrategies, 1);
%     P    = NaN(nStrategies, 4);   % [beta0 beta1 lowerAsy effAcc]
%     for s = 1:nStrategies
%         idxS{s} = find(strategyMatrix(condIdx, :) == s & ~isnan(allParams(condIdx, :, 1)));
%         if isempty(idxS{s}), continue; end
%         P(s,:) = [median(allParams(condIdx, idxS{s}, 1), 'omitnan'), ...
%                   median(allParams(condIdx, idxS{s}, 2), 'omitnan'), ...
%                   median(allParams(condIdx, idxS{s}, 3), 'omitnan'), ...
%                   median(allParams(condIdx, idxS{s}, 5), 'omitnan')]; %%%%
%     end
% 
%     f = figure('Position', [100 100 1400 800]);
%     hold on;
% 
%     % ---- Pass 1: faint background lines (so bold fits always sit on top) ----
%     for s = 1:nStrategies
%         if isempty(idxS{s}), continue; end
%         colFade = 0.5 * stratColors(s,:) + 0.5;
%         if showIndiv
%             for p = idxS{s}
%                 indivFits = [allParams(condIdx, p, 1); allParams(condIdx, p, 2); ...
%                              allParams(condIdx, p, 3); 1 - allParams(condIdx, p, 4)];
%                 plot(plotTimeSeconds, logist_valLU2(indivFits, predMatrix), ...
%                     'Color', colFade, 'LineWidth', 0.5);
%             end
%         end
%         if showRaw
%             groupRawAcc = squeeze(nanmean(allRawAcc(condIdx, idxS{s}, 1:maxFrames), 2));
%             plot(plotTimeSeconds, groupRawAcc, 'Color', colFade, 'LineWidth', 1);
%         end
%     end
% 
%     % ---- Pass 2: asymptotes, group fits, tangents ----
%     legH = []; legL = {};
%     for s = 1:nStrategies
%         if isempty(idxS{s}), continue; end
%         col = stratColors(s,:);
%         avgBeta0 = P(s,1); avgBeta1 = P(s,2); avgLowerAsy = P(s,3); avgEffAcc = P(s,4);
% 
%         avgFits     = [avgBeta0; avgBeta1; avgLowerAsy; (1 - avgEffAcc)];
%         smoothedAcc = logist_valLU2(avgFits, predMatrix);
% 
%         avgRange       = avgEffAcc - avgLowerAsy;
%         midpointSec    = (-avgBeta0 / avgBeta1) * 0.05;
%         maxSlopePerSec = ((avgBeta1 * avgRange) / 4) / 0.05;
%         yMidGroup      = (avgLowerAsy + avgEffAcc) / 2;
% 
%         yline(avgEffAcc, '--', 'Color', col, 'LineWidth', 1.5, 'HandleVisibility', 'off');
% 
%         h = plot(plotTimeSeconds, smoothedAcc, '-', 'Color', col, 'LineWidth', 5);
% 
%         if midpointSec > 0 && midpointSec < 30
%             % Cap tangent length so steep slopes don't dominate the plot
%             dx = min(1.5, maxTangentRise / max(abs(maxSlopePerSec), eps));
%             tX = [midpointSec - dx, midpointSec + dx];
%             tY = maxSlopePerSec * (tX - midpointSec) + yMidGroup;
%             plot(tX, tY, '-', 'Color', col * 0.6, 'LineWidth', 3, 'HandleVisibility', 'off');
%             plot(midpointSec, yMidGroup, 'o', 'MarkerFaceColor', col * 0.6, ...
%                 'MarkerEdgeColor', 'w', 'MarkerSize', 12, 'HandleVisibility', 'off');
%         end
% 
%         legH(end+1) = h; %#ok<SAGROW>
%         legL{end+1} = sprintf('%s (n=%d)\nAsy: %.2f\nSlope: %.2f acc/s', ...
%             stratShort{s}, numel(idxS{s}), avgEffAcc, maxSlopePerSec); %#ok<SAGROW>
%     end
% 
%     % --- Poster Formatting ---
%     ax = gca;
%     ax.FontSize = 24;
%     ax.LineWidth = 3;
%     ax.TickLength = [0.02 0.02];
%     xlim([0 10]);
%     ylim([0 1]);
%     grid on;
%     xlabel('Seconds Since Source Switch', 'FontSize', 30, 'FontWeight', 'bold');
%     ylabel('Accuracy', 'FontSize', 30, 'FontWeight', 'bold');
%     title({sprintf('Std %g, Hazard %g', avgStdVal, avgHVal), 'By Global Strategy'}, 'FontSize', 28);
%     if ~isempty(legH)
%         legend(legH, legL, 'Location', 'eastoutside', 'FontSize', 20, 'Box', 'off');
%     end
% 
%     % saveas(f, fullfile(figurePath, sprintf('Strategy_Fits_Cond%d.png', condIdx)));
% end

%% --- Option 1: Grouped Bar Chart (one group per condition, one bar per participant) ---
% f = figure;
% bh = bar(accMatrix, 'grouped');
% 
% % Assign distinct colors per participant for easy tracking across conditions
% cmap = lines(length(participantIDs));
% for i = 1:length(bh)
%     bh(i).FaceColor = cmap(i, :);
% end
% 
% hold on;
% 
% % Overlay the group mean per condition as a black marker for quick reference
% groupMeans = nanmean(accMatrix, 2);
% meanHandle = plot(1:nConditions, groupMeans, 'ko', 'MarkerFaceColor', 'k', ...
%     'MarkerSize', 10, 'LineWidth', 2);
% 
% % --- Formatting ---
% ax = gca;
% ax.FontSize = 16;
% ax.LineWidth = 2;
% ax.XTick = 1:nConditions;
% ax.XTickLabel = condLabelsCell;
% ax.XTickLabelRotation = 0;
% ylim([0 1]);
% ylabel('Effective Accuracy', 'FontSize', 20, 'FontWeight', 'bold');
% xlabel('Condition (Std / Hazard)', 'FontSize', 20, 'FontWeight', 'bold');
% title('Individual Participant Accuracy by Condition', 'FontSize', 24);
% grid on;
% 
% legend([bh, meanHandle], [legendLabels, {'Group Mean'}], ...
%     'Location', 'eastoutside', 'FontSize', 12);
% 
% % Save figure (commented out by default)
% % saveas(f, fullfile(figurePath, 'individual_accuracy_by_condition_bar.png'));
% 
% %% --- Option 2: Heatmap (Participant x Condition) ---
% % Often better for outlier detection: a participant row that's a
% % consistently different color stands out, and a condition column that's
% % uniformly off stands out separately.
% f2 = figure;
% imagesc(accMatrix'); % transpose so rows = participants, columns = conditions
% colormap(parula);
% cb = colorbar;
% cb.Label.String = 'Effective Accuracy';
% cb.Label.FontSize = 16;
% cb.Label.FontWeight = 'bold';
% clim([0 1]);
% 
% ax = gca;
% ax.FontSize = 14;
% ax.XTick = 1:nConditions;
% ax.XTickLabel = condLabelsCell;
% ax.XTickLabelRotation = 0;
% ax.YTick = 1:length(participantIDs);
% ax.YTickLabel = legendLabels;
% 
% xlabel('Condition (Std / Hazard)', 'FontSize', 20, 'FontWeight', 'bold');
% ylabel('Participant', 'FontSize', 20, 'FontWeight', 'bold');
% title('Heatmap of Effective Accuracy: Participant x Condition', 'FontSize', 24);
% 
% % Save figure (commented out by default)
% % saveas(f2, fullfile(figurePath, 'heatmap_accuracy_participant_by_condition.png'));