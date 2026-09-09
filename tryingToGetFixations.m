%% --- 1. INITIALIZATION & STRATEGY MAPPING ---
% Define participant pool
participantIDs = 6:13; 

% Define strategy groups for the Local Analysis (using actual PIDs)
% These groups categorize participants based on their distinct gaze behaviors
strat1_pids = [6, 7];           
strat2_pids = [13];             
strat3_pids = [8, 9, 10, 11, 12]; 

% Define temporal window for local event-related analysis (in frames/samples)
timeBefore = 200;
timeAfter = 200;

% Define variables for potential spatial filtering/splitting (not strictly active in this snippet)
isSplit = 0;
topCut = 1536;
botCut = 384;

% Output directory for saving figures
figurePath = '/Users/janebaker/Desktop/triangles/taskData/figures';

% Storage for Local Strategies (Time-Series)
% Dimensions: [2 Conditions (Low/High) x 3 Strategy Groups]
stratDataStorage = cell(2, 3); 
indivDataStorage = cell(2, length(participantIDs));


% Storage for Global Strategies (Proportions)
% Dimensions: [2 Conditions x Participants]
oppProportion = zeros(2, length(participantIDs));
sameProportion = zeros(2, length(participantIDs));

%% --- 2. MAIN PROCESSING LOOP (DATA COLLECTION) ---
% Loop through each participant to extract and process eye-tracking and task data
for p = 1:length(participantIDs)
    pid = participantIDs(p);
    
    % Determine strategy bucket index (s_idx) for time-series storage
    if ismember(pid, strat1_pids), s_idx = 1;
    elseif ismember(pid, strat2_pids), s_idx = 2;
    elseif ismember(pid, strat3_pids), s_idx = 3;
    else, s_idx = []; % Fallback if PID isn't mapped
    end

    % Set up sampling parameters (e.g., downsampling factor to convert frames to ms)
    numFrames = 6000;
    dsFactor = 50;
    numMs = numFrames * dsFactor;
    
    % Add necessary toolboxes and analysis scripts to path
    addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
    addpath(genpath('../../analysis/'))
    dataDir = '/Users/janebaker/Desktop/triangles/taskData/real data/';
    
    % Load Eye & Task Data for the current participant
    load(fullfile(dataDir, ['p' num2str(pid) '.mat'])); 
    et = parseeyelink([dir 'p' num2str(pid) '.asc'], [dir 'p' num2str(pid) '_eyeData.mat']);
    et = load(fullfile(dataDir, ['p' num2str(pid) '_eyeData.mat']));
    
    % Parse eye-tracker messages to find true block start and end timestamps
    msgs = string({et.othermessages.msg});
    start_mask = contains(msgs, "BLOCK_START");
    end_mask   = contains(msgs, "BLOCK_END");
    
    % Extract the numeric timestamps from the matching messages
    start_cell = regexp(msgs(start_mask), '\d+', 'match', 'once');
    end_cell   = regexp(msgs(end_mask), '\d+', 'match', 'once');
    
    blockStarts = str2double(start_cell);
    blockEnds   = str2double(end_cell);
    
    % Isolate timestamps specifically for Blocks 2 (Low) and 3 (High)
    % (Indices 6 and 7 represent blocks 2 and 3 in the overarching sequence)
    b2s = blockStarts(6); b2e = blockEnds(6);
    b3s = blockStarts(7); b3e = blockEnds(7);
    
    eyeData = et.data;
    % Check if data is binocular (more than 5 columns implies left and right eye data present)
    twoEye = size(eyeData, 2) > 5;
    
    % Extract raw X coordinates for both blocks using the timestamp masks
    b2mask_raw = (eyeData(:, 1) >= b2s & eyeData(:, 1) <= b2e);
    b3mask_raw = (eyeData(:, 1) >= b3s & eyeData(:, 1) <= b3e);

    if ~twoEye
        % Handle Monocular Data
        xRaw2 = eyeData(b2mask_raw, 2);
        xRaw3 = eyeData(b3mask_raw, 2);
        yRaw2 = eyeData(b2mask_raw, 3);
        yRaw3 = eyeData(b3mask_raw, 3);
    else
        % Handle Binocular Data: Extract X coordinates for both eyes
        X2 = eyeData(b2mask_raw, [2, 5]);
        X3 = eyeData(b3mask_raw, [2, 5]);
        
        % Clean zero values (lost tracking in one eye) by carrying over the valid eye's data
        X2(X2(:,1) == 0 & X2(:,2) ~= 0, 1) = X2(X2(:,1) == 0 & X2(:,2) ~= 0, 2);
        X2(X2(:,2) == 0 & X2(:,1) ~= 0, 2) = X2(X2(:,2) == 0 & X2(:,1) ~= 0, 1);
        X3(X3(:,1) == 0 & X3(:,2) ~= 0, 1) = X3(X3(:,1) == 0 & X3(:,2) ~= 0, 2);
        X3(X3(:,2) == 0 & X3(:,1) ~= 0, 2) = X3(X3(:,2) == 0 & X3(:,1) ~= 0, 1);
        
        % Average the two eyes to get a single X coordinate stream; zero out Y coordinates
        xRaw2 = mean(X2, 2); xRaw3 = mean(X3, 2);
        yRaw2 = zeros(size(xRaw2)); yRaw3 = zeros(size(xRaw3)); 
    end
    
    % Calculate spatial references based on target locations
    leftMean = data(1).centers;
    rightMean = data(2).centers;
    middle = mean([leftMean, rightMean]);
    halfRange = (rightMean - leftMean) / 2;
    
    % Loop through Block 2 (Low Uncertainty) and Block 3 (High Uncertainty)
    for b = 2:3
        cond_idx = b - 1; % Maps blocks 2,3 to condition indices 1,2
        
        % Select the appropriate eye data and task data for the current block
        if b == 2
            ed_full = eyeData(b2mask_raw, :);
            td = data([data.blockNumber] == 2);
            xR = xRaw2; yR = yRaw2;
        else
            ed_full = eyeData(b3mask_raw, :);
            td = data([data.blockNumber] == 3);
            xR = xRaw3; yR = yRaw3;
        end
        % Downsample the full eye data based on the predetermined dsFactor
        ed_ds = ed_full(1:50:numMs, :);
        
        % --- FIX: POPULATE beliefShift FIELD ---
        % Identify trials where the participant changed their response key
        keys = [td.keyPressed];
        shifts = [false, diff(keys) ~= 0];
        for i = 1:length(td)
            td(i).beliefShift = shifts(i);
        end
        
        % Blink/Saccade Filter Logic
        % Identify blinks (pupil size/status == 0)
        if ~twoEye, isBlink = ed_ds(:, 4) == 0; 
        else, isBlink = ed_ds(:, 4) == 0 | ed_ds(:, 7) == 0; end
        
        % Detect saccades using velocity/acceleration thresholds
        SACs = findSaccadesWithPlots(xR, yR, 'vi_min', 3.0, 'vp_min', 8.0, 'd_min', 50, 'a_min', 1.0, 'pad', 50);
        isSacRaw = false(length(xR), 1);
        
        % Create a boolean mask of all raw samples that fall within a saccade
        for i = 1:size(SACs, 1)
            on = SACs(i, 1); off = min(on + SACs(i, 2) - 1, length(isSacRaw));
            isSacRaw(on:off) = true;
        end
        
        % Apply downsampling to the raw X and Saccade mask
        xP = xR(1:50:numMs); isSac = isSacRaw(1:50:numMs);
        
        % Remove blinks and saccades by setting them to NaN
        filtered = xP; filtered(isSac | isBlink) = nan;
        
        % Normalize gaze so that -1 is the Left target and +1 is the Right target
        nFiltered = (filtered - middle) / halfRange;
        
        % Store for Local Analysis (Event-related alignment around belief shifts)
        if ~isempty(s_idx)
            % analyzeRawData aligns gaze time-series to specific trial events
            allTrialPos = analyzeRawData(nFiltered, td, leftMean, rightMean);
            % Append the mean of the aligned trials to the group storage array
            stratDataStorage{cond_idx, s_idx} = [stratDataStorage{cond_idx, s_idx}; mean(allTrialPos, 1, 'omitnan')];
        end

        if ~isempty(allTrialPos)
            validPos = allTrialPos(~all(isnan(allTrialPos), 2), :);
            indivDataStorage{cond_idx, p} = [indivDataStorage{cond_idx, p}; validPos];
        end

        % Store for Global Analysis (Proportion of time looking at each source)
        lookCount = 0; oppCount = 0; sameCount = 0;
        
        for f_idx = 1:length(td)
            if ~isnan(nFiltered(f_idx))
                lookCount = lookCount + 1;
                
                % Determine if participant selected Left ('F') or Right ('J')
                isLeftK = td(f_idx).keyPressed == 'F'; 
                
                % Check if gaze is strongly biased to the Right (>= 0.80) or Left (<= -0.80)
                lookR = nFiltered(f_idx) >= 0.80;
                isRightK = td(f_idx).keyPressed == 'J'; 
                lookL = nFiltered(f_idx) <= -0.80;
                
                % Tally whether they are looking at the opposite source they selected, 
                % or the same source they selected
                if (isLeftK && lookR) || (isRightK && lookL), oppCount = oppCount + 1;
                elseif (isLeftK && lookL) || (isRightK && lookR), sameCount = sameCount + 1; end
            end
        end
        % Calculate raw proportions relative to total valid gaze frames
        oppProportion(cond_idx, p) = oppCount / lookCount;
        sameProportion(cond_idx, p) = sameCount / lookCount;
    end
end
% 
% %% --- 3. LOCAL STRATEGIES PLOT ---
% % Define styling and labels for the Time-Series plot
condNames = {'Low Uncertainty', 'High Uncertainty'};
stratNames = {'Strategy 1', 'Strategy 2', 'Strategy 3'};
colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];

% Loop through Low (c=1) and High (c=2) conditions to generate separate figures
for c = 1:2
    figL = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;
    legH = [];

    % Optional: Add a subtle dashed line at the center of the screen (gaze midpoint)
    yline(0.5, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'HandleVisibility', 'off');

    % Loop through the 3 strategy groups to plot their respective lines
    for s = 1:3
        dataMat = stratDataStorage{c, s}; 
        if isempty(dataMat), continue; end

        % --- TRANSFORMATION ---
        % Converts [0 = New, 2 = Old] to [1 = New, 0 = Old]
        % This makes the Y-axis conceptually intuitive: High = New Source, Low = Old Source
        dataMat = 1 - (dataMat / 2);

        % Compute the cross-participant average for this strategy group
        avg = mean(dataMat, 1, 'omitnan');

        % Construct the time axis in seconds centered around the shift (0s)
        timeAxis = ((1:length(avg)) - (timeBefore + 1)) * (dsFactor / 1000);

        % If there's more than one participant in the group, plot shaded standard error (SEM)
        if size(dataMat, 1) > 1
            sem = std(dataMat, 0, 1, 'omitnan') ./ sqrt(size(dataMat, 1));
            fill([timeAxis, fliplr(timeAxis)], [avg+sem, fliplr(avg-sem)], colors(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end

        % Plot the thick mean line and store handle for the legend
        h = plot(timeAxis, avg, 'Color', colors(s,:), 'LineWidth', 5); legH = [legH, h];
    end

    % Apply formatting for poster readability
    set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out');
    xlabel('Time (Seconds)', 'FontSize', 36, 'FontWeight', 'bold');

    % --- NEW INTUITIVE Y-AXIS ---
    ylabel('Gaze Position', 'FontSize', 36, 'FontWeight', 'bold');
    ylim([0 1]);
    yticks([0, 0.5, 1]);
    yticklabels({'Old Belief Source', 'Center', 'New Belief Source'});

    title(condNames{c}, 'FontSize', 44, 'FontWeight', 'bold');
    legend(legH, stratNames, 'FontSize', 26, 'Location', 'best'); 
    xlim([-10 10]); grid on;

    % Uncomment to save the figure
    % saveas(figL, fullfile(figurePath, [condNames{c} '_Local_Strategies.png']));
end

%% --- 4. GLOBAL STRATEGIES PLOT (Annotated P1-P8) ---
% Generate distinct colors for each participant scatter point
participantColors = lines(length(participantIDs));

% Loop through Low and High uncertainty conditions
for c = 1:2
    figG = figure('Units', 'inches', 'Position', [1, 1, 14, 12], 'Color', 'w'); hold on;

    % Draw a diagonal reference line (x = y) indicating equal looking behavior
    plot([0 1], [0 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2.5, 'HandleVisibility', 'off');

    % Plot each participant's aggregate viewing proportion
    for p = 1:length(participantIDs)
        % xVal = Look Same, yVal = Look Opposite. 
        % midVal represents time spent looking at neither (the center/elsewhere)
        xVal = sameProportion(c, p); yVal = oppProportion(c, p); midVal = 1 - xVal - yVal;

        % Plotting markers for the individual participant
        plot(xVal, yVal, 'o', 'MarkerSize', 18, 'MarkerFaceColor', participantColors(p,:), 'MarkerEdgeColor', 'k', 'LineWidth', 2);

        % Annotating the data point with the synthetic label "P1-P8" and their mid-viewing proportion
        labelStr = sprintf(' P%d (mid: %.2f)', p, midVal); 
        text(xVal + 0.015, yVal, labelStr, 'FontSize', 24, 'FontWeight', 'bold', 'Color', participantColors(p,:) * 0.7);
    end

    % Format axes for presentation
    set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out', 'Box', 'on');
    xlabel('Fixation Proportion: Same Source', 'FontSize', 36, 'FontWeight', 'bold');
    ylabel('Fixation Proportion: Opposite Source', 'FontSize', 36, 'FontWeight', 'bold');
    title([condNames{c} ': Global Viewing Strategy'], 'FontSize', 44, 'FontWeight', 'bold');
    xlim([0 1]); ylim([0 1]); grid on;

    % Uncomment to save the figure
    % saveas(figG, fullfile(figurePath, [condNames{c} '_Global_Strategy_Scatter.png']));
end

%% --- 5. LOCAL STRATEGIES PLOT (INDIVIDUAL PARTICIPANTS) ---
% Define distinct colors for the two conditions
% condColors = [0.00 0.45 0.74;  % Blue for Low Uncertainty (Block 1)
%               0.85 0.33 0.10]; % Orange for High Uncertainty (Block 2)
% 
% % Loop through INDIVIDUAL participants to create a separate figure for each
% for p = 1:length(participantIDs)
%     pid = participantIDs(p);
% 
%     figI = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;
%     legH = [];
%     legNames = {};
% 
%     % Add a subtle dashed line at the center of the screen (gaze midpoint)
%     yline(0.5, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'HandleVisibility', 'off');
% 
%     % Plot Low (c=1) and High (c=2) on the SAME graph for this participant
%     for c = 1:2
%         dataMat = indivDataStorage{c, p};
%         if isempty(dataMat), continue; end
% 
%         % --- TRANSFORMATION ---
%         % Convert [0 = New, 2 = Old] to [1 = New, 0 = Old]
%         dataMat = 1 - (dataMat / 2);
% 
%         % Compute the cross-shift average for this condition
%         avg = mean(dataMat, 1, 'omitnan');
% 
%         % Construct the time axis in seconds centered around the shift (0s)
%         timeAxis = ((1:length(avg)) - (timeBefore + 1)) * (dsFactor / 1000);
% 
%         % --- SHADED ERROR BOUNDS ---
%         % Calculate standard error across the individual's shifts/trials
%         if size(dataMat, 1) > 1
%             sem = std(dataMat, 0, 1, 'omitnan') ./ sqrt(size(dataMat, 1));
% 
%             % Plot standard error shading (HandleVisibility off so it doesn't show in legend)
%             fill([timeAxis, fliplr(timeAxis)], [avg+sem, fliplr(avg-sem)], ...
%                 condColors(c,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
%         end
% 
%         % Plot the solid mean line on top of the shading
%         h = plot(timeAxis, avg, 'Color', condColors(c,:), 'LineWidth', 4); 
%         legH = [legH, h];
%         legNames{end+1} = condNames{c};
%     end
% 
%     % Apply formatting for readability
%     set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out');
%     xlabel('Time (Seconds)', 'FontSize', 36, 'FontWeight', 'bold');
% 
%     ylabel('Gaze Position', 'FontSize', 36, 'FontWeight', 'bold');
%     ylim([0 1]);
%     yticks([0, 0.5, 1]);
%     yticklabels({'Old Belief Source', 'Center', 'New Belief Source'});
% 
%     title(sprintf('Local Gaze Shifts: Participant %d', p), 'FontSize', 44, 'FontWeight', 'bold');
% 
%     % Only create legend if data was successfully plotted
%     if ~isempty(legH)
%         legend(legH, legNames, 'FontSize', 24, 'Location', 'best');
%     end
% 
%     xlim([-10 10]); grid on;
% 
%     % Uncomment to save the figure per participant
%     % saveas(figI, fullfile(figurePath, sprintf('P%d_Local_Strategy_Old.png', pid)));
% end