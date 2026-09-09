%% --- Environment Setup ---
% Define the participants to include in the analysis
participantIDs = [001, 002, 003, 004, 005, 006, 007, 008, 009, 010];

% Define paths for saving figures and locating necessary analysis toolboxes
figurePath = '/Users/janebaker/Desktop/triangles/taskData/figures';
addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
addpath(genpath('../../analysis/'))

% Preallocate array to store logistic fit parameters for all participants
% Dimensions: [Block x Participant x Parameter] 
% Parameters: 1=beta0 (intercept), 2=beta1 (slope), 3=lowerAsy, 4=upperAsy, 5=EffectiveAccuracy
allParams = NaN(5, length(participantIDs), 5); 

% Preallocate array to store raw accuracy as a function of frames since switch
maxFramesPossible = 2000; 
% Dimensions: [Block x Participant x FrameIndex]
allRawAcc = NaN(5, length(participantIDs), maxFramesPossible);

%% --- Individual Participant Loop ---
% Iterate over each specified participant to extract data and fit logistic models
for p = 1:length(participantIDs)
    pid = participantIDs(p);
    
    % Load behavioral data for the current participant
    load(['/Users/janebaker/Desktop/triangles/taskData/finalData/p' num2str(pid) '.mat']);
    
    % Extract unique block numbers and map them for filtering
    blocks = unique([data.blockNumber]);
    blockNum = [data.blockNumber];
    
    % Loop through blocks (starting at index 2, typically skipping training/practice)
    for b = 2:length(blocks)
        % Filter task data, source states, and participant responses for this specific block
        thisBlockTask = data(blockNum == blocks(b));
        thisBlockSource = [thisBlockTask.sourceIsLeft];
        thisBlockBelief = [thisBlockTask.keyPressed];
        stdIdx = thisBlockTask(1).stdHrzIdx;
        hIdx = thisBlockTask(1).hazardIdx;
        
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
        
        % Calculate and store the mean raw accuracy for every frame interval
        maxX = max(trialFramesSince);
        for n = 1:maxX
            allRawAcc(b, p, n) = mean(trialIsCorrect(trialFramesSince == n));
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
            
            % Logic: Only use bottom asymptote for P10 in Block 3 (likely due to inverted mapping/performance issue)
            if pid == 10 && b == 3
                effAcc = lowerAsy;
            else
                effAcc = upperAsy;
            end
            
            % Store extracted parameters for Group Calculation
            allParams(b, p, 1) = beta0;
            allParams(b, p, 2) = beta1;
            allParams(b, p, 3) = lowerAsy;
            allParams(b, p, 4) = upperAsy;
            allParams(b, p, 5) = effAcc;

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
            title(sprintf('P%d Std %d H %d: Logistic Fit', pid, stdIdx, hIdx), 'FontSize', 30);
            
            % Uncomment to save individual plots
            %saveas(f, fullfile(figurePath, sprintf('p%d_b%d_logistic_fit.png', pid, b)));
            close(f);
        catch ME
            % Error handling to prevent one bad fit from stopping the entire script
            fprintf('Fit failed for P %d Std %d H %d: Logistic Fit', pid, stdIdx, hIdx, ME.message);
        end
    end
end

%% --- Generate Group Average Figures ---
% Loop through blocks to create cross-participant aggregate visualisations
for b = 2:length(blocks)
    
    % Calculate group means for each parameter, ignoring NaN values (failed fits)
    avgBeta0    = nanmean(allParams(b, :, 1));
    avgBeta1    = nanmean(allParams(b, :, 2));
    avgLowerAsy = nanmean(allParams(b, :, 3));
    avgEffAcc   = nanmean(allParams(b, :, 5)); 
    
    % Skip if no valid parameters were generated for this block
    if isnan(avgBeta0), continue; end
    
    % Reconstruct the average fits array for generating the group logistic line
    avgFits = [avgBeta0; avgBeta1; avgLowerAsy; (1 - avgEffAcc)];
    
    % Calculate the group-average raw accuracy across valid frames
    groupRawAcc = squeeze(nanmean(allRawAcc(b, :, :), 2));
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
    
    % Map block index to condition title for presentation
    titles = ["Low Uncertainty", "High Uncertainty"];

    % Since blocks start at 2, b-1 correctly selects index 1 and 2 from titles array
    title(titles(b-1), 'FontSize', 36)
    
    % Save as PDF (Commented out by default)
    % saveas(f, fullfile(figurePath, sprintf('POSTER_low_Average_Block%d.png', b)));
end