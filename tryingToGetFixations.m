warning('off','all')

%% --- SELECT TESTS TO RUN ---
validTests = {
    'Raw Data'                  % shows fixations of selected participants for each condition
    'Global Strats Scatter'     % scatter plot comparing each participant's same vs opposite fixation proportion
    'Global Strats Bar'         % bar graph of how the same vs opposite fixation proportions differ by condition
    'Global Strats Individual'  % bar graph of same vs opposite fixation proportions for individual participants
    'Central Fixation'          % heatmap grouping participants into central fixation categories
    'Opposite Fixation'         % heatmap grouping participants into opposite fixation categories
    'Opposite Fixation Means'   % compares average opposite fixation proportion across conditions
    'Local Strats Individual'   % plots individual fixations surrounding a belief switch
    'Grouped Local Strats'      % plots manually categorized local strategies
    };

% Select any combination of tests
testsToRun = {'Raw Data', 'Global Strats Scatter', 'Opposite Fixation Means'};

% Check for invalid test names
unknownTests = setdiff(testsToRun, validTests);
if ~isempty(unknownTests)
    error('Unknown test(s): %s', strjoin(unknownTests, ', '));
end

% Determine which shared calculations are needed
runLocal = any(ismember(testsToRun, ...
    {'Local Strats Individual', 'Grouped Local Strats'}));

runGlobal = any(ismember(testsToRun, ...
    {'Global Strats Scatter', 'Global Strats Bar', ...
    'Global Strats Individual', 'Central Fixation', ...
    'Opposite Fixation', 'Opposite Fixation Means'}));

if isempty(testsToRun)
    disp('No tests selected.');
    return;
end

%% --- 1. INITIALIZATION & CONDITION / STRATEGY MAPPING ---
% Define participant pool (p001 - p010)
participantIDs = 1:34;
numParticipants = length(participantIDs);

% --- Condition mapping ---
% 6 conditions = all combinations of stdHrzIdx (1-3) x hazardIdx (1-2)
% condIdx = (stdHrzIdx - 1)*2 + hazardIdx
%   1: std1,h1   2: std1,h2   3: std2,h1   4: std2,h2   5: std3,h1   6: std3,h2
numConditions = 6;
condLabels   = repmat({'Cond (unfilled)'}, numConditions, 1); % filled in during main loop
condStdHrz   = nan(numConditions, 1); % actual SD value for each condIdx
condHazard   = nan(numConditions, 1); % actual H value for each condIdx

% --- Strategy assignment for LOCAL analysis ---
% Because participants can have good data in one block/condition and bad
% data in another, strategy membership is tracked PER CONDITION.
% stratAssignment{condIdx}(p) = 1, 2, 3, or NaN (unassigned / excluded)
% Currently unknown -> initialize as all NaN. Fill in manually once
% individual local strategies have been inspected, e.g.:
%   stratAssignment{1}(ismember(participantIDs,[1 2])) = 1;
%   stratAssignment{1}(ismember(participantIDs,[5]))   = 2;


% 1 = Peak at belief switch
% 2 = Decreasing at belief switch
% 3 = Mixed/balanced
% NaN = No usable trace

% Rows: p001-p031
% Columns: conditions 1-6

% Define temporal window for local event-related analysis (in frames/samples)
timeBefore = 200;
timeAfter = 200;

% Define variables for potential spatial filtering/splitting (not strictly active in this snippet)
isSplit = 0;
topCut = 1536;
botCut = 384;

% Output directory for saving figures
figurePath = '/Users/janebaker/Desktop/triangles/taskData/figures';

% --- Eye-data validity safeguard ---
% Minimum % of non-NaN (non-blink/non-saccade) samples required within a
% block for that participant/condition to be included in analysis.
minValidPct = 50;
validEyeData = false(numConditions, numParticipants);
pctValidEyeData = nan(numConditions, numParticipants);

% --- Storage for LOCAL analysis (time-series), collected per participant ---
% indivDataStorage{condIdx, p} = matrix of aligned gaze trials for that
% participant/condition. Used both for individual plots AND (later, once
% stratAssignment is populated) for group strategy plots.
% Local analysis storage
indivDataStorage = cell(numConditions, numParticipants);

% Normalized gaze and corresponding response keys
nFilteredStorage = cell(numConditions, numParticipants);
keyPressedStorage = cell(numConditions, numParticipants);

% --- Storage for GLOBAL analysis (proportions) ---
% NaN indicates missing/excluded data
oppProportion  = nan(numConditions, numParticipants);
sameProportion = nan(numConditions, numParticipants);

% --- Data-driven fixation boundary settings ---
nSDBoundary = 1;                 % # of SDs from each peak defining "source" zone; tune after inspecting plots
showFixationHistograms = true;   % set true to display GMM/histogram diagnostic per condition

% Add necessary toolboxes and analysis scripts to path
addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
addpath(genpath('../../analysis/'))
dataDir = '/Users/janebaker/Desktop/triangles/taskData/finalData/';

%% --- 2. MAIN PROCESSING LOOP (DATA COLLECTION) ---
for p = 1:numParticipants
    pid = participantIDs(p);
    pidStr = sprintf('p%03d', pid);

    % Downsampling factor (ms per downsampled sample). Eyetracker samples at 1000Hz (1ms/sample).
    if pid <= 10
        sr = 1000;
    else
        sr = 2000;
    end

    dsFactor = sr / 20;
    scale = 1000 / sr;

    % Load Task Data for the current participant
    load(fullfile(dataDir, [pidStr '.mat']));

    % Load/parse Eye Data for the current participant
    eyeFile = fullfile(dataDir, [pidStr '_eyeData.mat']);
    ascFile = fullfile(dataDir, [pidStr '.asc']);

    if ~isfile(eyeFile)
        if ~isfile(ascFile)
            fprintf('No eye data found for %s. Skipping.\n', pidStr);
            continue;  % Move to next participant in for loop
        end
        et = parseeyelink(ascFile, eyeFile);
    else
        et = load(eyeFile);
    end

    % Parse eye-tracker messages to find true block start and end timestamps
    msgs = string({et.othermessages.msg});
    start_mask = contains(msgs, "BLOCK_START");
    end_mask   = contains(msgs, "BLOCK_END");

    % Extract the numeric timestamps from the matching messages
    start_cell = regexp(msgs(start_mask), '\d+', 'match', 'once');
    end_cell   = regexp(msgs(end_mask), '\d+', 'match', 'once');

    blockStarts = str2double(start_cell);
    blockEnds   = str2double(end_cell);
    % There are 11 BLOCK_START/END pairs total: the first 5 belong to the
    % training block (ignored here). The 6 experimental blocks correspond
    % to message indices 6-11, i.e. eyeMsgIdx = blockNumber + 4 for
    % data.blockNumber = 2:7.

    eyeData = et.data;
    % Check if data is binocular (more than 5 columns implies left and right eye data present)
    twoEye = size(eyeData, 2) > 5;

    % Calculate spatial references based on target locations
    leftMean = data(1).centers;
    rightMean = data(2).centers;
    middle = mean([leftMean, rightMean]);
    halfRange = (rightMean - leftMean) / 2;

    % Loop through the 6 experimental blocks (data.blockNumber = 2:7)
    for blockNumber = 2:7
        eyeMsgIdx = blockNumber + 4; % maps blockNumber 2-7 -> eye message index 6-11

        if eyeMsgIdx > length(blockStarts) || eyeMsgIdx > length(blockEnds)
            warning('%s: missing BLOCK_START/END for blockNumber %d. Skipping.', pidStr, blockNumber);
            continue;
        end

        bs = blockStarts(eyeMsgIdx);
        be = blockEnds(eyeMsgIdx);

        td = data([data.blockNumber] == blockNumber);
        if isempty(td)
            warning('%s: no task trials found for blockNumber %d. Skipping.', pidStr, blockNumber);
            continue;
        end

        % --- Determine condition index from std/hazard indices (constant within block) ---
        stdHrzIdx = td(1).stdHrzIdx;
        hazardIdx = td(1).hazardIdx;
        stdHrzVal = td(1).stdHrz;
        Hval      = td(1).H;
        condIdx = (stdHrzIdx - 1) * 2 + hazardIdx;

        condLabels{condIdx} = sprintf('SD = %.3g, H = %.3g', stdHrzVal, Hval);
        condStdHrz(condIdx) = stdHrzVal;
        condHazard(condIdx) = Hval;

        % Expected number of frames/samples for this block, based on actual trial count
        numFramesBlock = length(td);

        % Mask for this block's raw eye samples
        bmask_raw = (eyeData(:, 1) >= bs & eyeData(:, 1) <= be);

        if ~twoEye
            % Handle Monocular Data
            xR = eyeData(bmask_raw, 2);
            yR = eyeData(bmask_raw, 3);
        else
            % Handle Binocular Data: Extract X coordinates for both eyes
            X = eyeData(bmask_raw, [2, 5]);

            % Clean zero values (lost tracking in one eye) by carrying over the valid eye's data
            X(X(:,1) == 0 & X(:,2) ~= 0, 1) = X(X(:,1) == 0 & X(:,2) ~= 0, 2);
            X(X(:,2) == 0 & X(:,1) ~= 0, 2) = X(X(:,2) == 0 & X(:,1) ~= 0, 1);

            % Average the two eyes to get a single X coordinate stream; zero out Y coordinates
            xR = mean(X, 2);
            yR = zeros(size(xR));
        end

        % Full raw eye data for this block (no fixed-length assumption)
        ed_full = eyeData(bmask_raw, :);

        % --- FIX: POPULATE beliefShift FIELD ---
        % Identify trials where the participant changed their response key
        keys = [td.keyPressed];
        shifts = [false, diff(keys) ~= 0];
        for i = 1:length(td)
            td(i).beliefShift = shifts(i);
        end

        % Detect saccades using velocity/acceleration thresholds (on raw, full-rate data)
        SACs = findSaccadesWithPlots(xR, yR, ...
            'vi_min', 3.0 * scale, ...
            'vp_min', 8.0 * scale, ...
            'd_min', 50, ...
            'a_min', 1.0 * scale^2, ...
            'pad', round(sr * 0.05), ...
            'sr', sr);
        isSacRaw = false(length(xR), 1);
        for i = 1:size(SACs, 1)
            on = SACs(i, 1); off = min(on + SACs(i, 2) - 1, length(isSacRaw));
            isSacRaw(on:off) = true;
        end

        % --- Downsample based on ACTUAL available samples (no fixed-length assumption) ---
        ed_ds_full  = ed_full(1:dsFactor:end, :);
        xP_full     = xR(1:dsFactor:end);
        isSac_full  = isSacRaw(1:dsFactor:end);

        nAvail = min([size(ed_ds_full, 1), length(xP_full), length(isSac_full)]);

        if nAvail ~= numFramesBlock
            warning('%s block %d (%s): %d downsampled samples available, expected %d frames (trial-count based). Aligning by truncation/padding.', ...
                pidStr, blockNumber, condLabels{condIdx}, nAvail, numFramesBlock);
        end

        % Align to expected frame count (length(td)) so trial-index mapping
        % (nFiltered(f_idx) <-> td(f_idx)) stays valid regardless of small
        % timing discrepancies in the recorded block duration.
        ed_ds = nan(numFramesBlock, size(ed_full, 2));
        xP    = nan(numFramesBlock, 1);
        isSac = false(numFramesBlock, 1);

        nCopy = min(nAvail, numFramesBlock);
        ed_ds(1:nCopy, :) = ed_ds_full(1:nCopy, :);
        xP(1:nCopy)       = xP_full(1:nCopy);
        isSac(1:nCopy)    = isSac_full(1:nCopy);

        % Blink Filter Logic (NaN rows from padding naturally evaluate to
        % false here, which is fine since xP is already NaN for those rows)
        if ~twoEye, isBlink = ed_ds(:, 4) == 0;
        else, isBlink = ed_ds(:, 4) == 0 | ed_ds(:, 7) == 0; end

        % Remove blinks and saccades by setting them to NaN
        filtered = xP; filtered(isSac | isBlink) = nan;

        % Normalize gaze so that -1 is the Left target and +1 is the Right target
        nFiltered = (filtered - middle) / halfRange;

        % --- SAFEGUARD: check valid eye data for this participant/condition ---
        pctValid = 100 * sum(~isnan(nFiltered)) / length(nFiltered);
        pctValidEyeData(condIdx, p) = pctValid;
        validEyeData(condIdx, p) = pctValid >= minValidPct;

        if ~validEyeData(condIdx, p)
            fprintf('SKIPPING %s, condition %d (%s): only %.1f%% valid eye data (< %d%%)\n', ...
                pidStr, condIdx, condLabels{condIdx}, pctValid, minValidPct);
            continue;
        end

        % Store for Local Analysis (Event-related alignment around belief shifts)
        if runLocal
            allTrialPos = analyzeRawData(nFiltered, td, leftMean, rightMean);

            if ~isempty(allTrialPos)
                validPos = allTrialPos(~all(isnan(allTrialPos), 2), :);
                indivDataStorage{condIdx, p} = ...
                    [indivDataStorage{condIdx, p}; validPos];
            end
        end

        % % Store for Global Analysis (Proportion of time looking at each source)
        % lookCount = 0; oppCount = 0; sameCount = 0;
        %
        % for f_idx = 1:length(td)
        %     if f_idx > length(nFiltered), break; end
        %     if ~isnan(nFiltered(f_idx))
        %         lookCount = lookCount + 1;
        %
        %         % Determine if participant selected Left ('F') or Right ('J')
        %         isLeftK = td(f_idx).keyPressed == 'F';
        %
        %         % Check if gaze is strongly biased to the Right (>= 0.80) or Left (<= -0.80)
        %         lookR = nFiltered(f_idx) >= 0.80;
        %         isRightK = td(f_idx).keyPressed == 'J';
        %         lookL = nFiltered(f_idx) <= -0.80;
        %
        %         % Tally whether they are looking at the opposite source they selected,
        %         % or the same source they selected
        %         if (isLeftK && lookR) || (isRightK && lookL), oppCount = oppCount + 1;
        %         elseif (isLeftK && lookL) || (isRightK && lookR), sameCount = sameCount + 1; end
        %     end
        % end
        % % Calculate raw proportions relative to total valid gaze frames
        % if lookCount > 0
        %     oppProportion(condIdx, p) = oppCount / lookCount;
        %     sameProportion(condIdx, p) = sameCount / lookCount;
        % end

        % Store raw per-trial normalized gaze + keys for later boundary-based proportion calc
        nFilteredStorage{condIdx, p}  = nFiltered;
        keyPressedStorage{condIdx, p} = [td.keyPressed];
    end
end

%% --- VALIDITY SUMMARY ---
fprintf('\n--- Eye Data Validity Summary (threshold = %d%%) ---\n', minValidPct);
for c = 1:numConditions
    nValid = sum(validEyeData(c, :));
    fprintf('Condition %d (%s): %d/%d participants valid\n', c, condLabels{c}, nValid, numParticipants);
end

%% --- PLOT RAW DATA (FIXATIONS, TARGET, BELIEF, SOURCE) ---
if ismember('Raw Data', testsToRun)
    plotPIDs = [11, 32];  % Change to specific participants
    % only care about p11 c1 and p32 c3

    for p = 1:numParticipants
        pid = participantIDs(p);
        if ~ismember(pid, plotPIDs), continue; end

        pidStr = sprintf('p%03d', pid);

        % Reload task data for plotting
        taskFile = load(fullfile(dataDir, [pidStr '.mat']));
        taskData = taskFile.data;

        for blockNumber = 2:7
            td = taskData([taskData.blockNumber] == blockNumber);
            if isempty(td), continue; end

            % Determine condition
            stdHrzIdx = td(1).stdHrzIdx;
            hazardIdx = td(1).hazardIdx;
            condIdx = (stdHrzIdx - 1) * 2 + hazardIdx;

            % Skip conditions without valid eye data
            if ~validEyeData(condIdx, p) || isempty(nFilteredStorage{condIdx, p})
                continue;
            end

            % Extract data
            fixX = nFilteredStorage{condIdx, p}(:)';
            normX = [td.targetX];
            thisBlockIsLeft = [td.sourceIsLeft];
            thisBlockBelief = [td.keyPressed];

            nFrames = length(td);

            % Normalize target and fixation positions (z-scores)
            nxz = (normX - mean(normX, 'omitnan')) / std(normX, 'omitnan');
            normFixX = (fixX - mean(fixX, 'omitnan')) / std(fixX, 'omitnan');

            % Map source and belief to vertical plot positions
            yl = [-3 3];

            sourceVal = zeros(1, nFrames);
            beliefVal = nan(1, nFrames);

            for t = 1:nFrames
                if thisBlockIsLeft(t)
                    sourceVal(t) = yl(1) + 0.5;
                else
                    sourceVal(t) = yl(2) - 0.5;
                end

                if thisBlockBelief(t) == 'F'
                    beliefVal(t) = yl(1) + 0.7;
                elseif thisBlockBelief(t) == 'J'
                    beliefVal(t) = yl(2) - 0.7;
                end
            end

            % --- PLOTTING ---
            figure;

            lb = [0.6 0.8 1];

            h1 = plot(1:nFrames, nxz, 'LineWidth', 0.1, 'Color', lb);
            hold on

            yline(0, '--', 'Color', 'k', 'LineWidth', 1);

            h3 = plot(1:nFrames, sourceVal, 'o', 'MarkerSize', 3);
            h4 = plot(1:nFrames, beliefVal, 'o', 'MarkerSize', 3);
            h6 = plot(1:nFrames, normFixX, 'LineWidth', 3, 'Color', 'b');

            xlabel('Frame');
            ylabel('Normalized Horizontal Position (- Left, + Right)');

            title(sprintf('Participant %d - %s', pid, condLabels{condIdx}));

            ylim(yl);
            xlim([1 nFrames]);

            legend([h1 h3 h4 h6], ...
                {'Target Position', 'True Source', ...
                'Participant Belief', 'Gaze Fixation'}, ...
                'Location', 'best');
        end
    end
end

%% --- DERIVE SINGLE DATA-DRIVEN FIXATION BOUNDARY (pooled across all conditions & participants) ---
if runGlobal
    pooledGaze = [];
    for c = 1:numConditions
        for p = 1:numParticipants
            if validEyeData(c, p) && ~isempty(nFilteredStorage{c, p})
                pooledGaze = [pooledGaze; nFilteredStorage{c, p}(:)];
            end
        end
    end

    [sourceBoundaryPos, sourceBoundaryNeg, boundaryFitInfo] = ...
        computeFixationBoundary(pooledGaze, nSDBoundary, showFixationHistograms, 'All Conditions Pooled');

    fprintf('\n--- Empirically-Derived Source Fixation Boundary (N = %d SD, pooled across all conditions) ---\n', nSDBoundary);
    fprintf('Boundary = +/- %.3f\n', sourceBoundaryPos);

    %% --- COMPUTE GLOBAL PROPORTIONS USING SINGLE POOLED BOUNDARY ---
    posThr = sourceBoundaryPos;
    negThr = sourceBoundaryNeg;

    for c = 1:numConditions
        for p = 1:numParticipants
            if ~validEyeData(c, p) || isempty(nFilteredStorage{c, p}), continue; end

            nF   = nFilteredStorage{c, p};
            keys = keyPressedStorage{c, p};

            lookCount = 0; oppCount = 0; sameCount = 0;
            for f_idx = 1:length(nF)
                if ~isnan(nF(f_idx))
                    lookCount = lookCount + 1;
                    isLeftK  = keys(f_idx) == 'F';
                    isRightK = keys(f_idx) == 'J';
                    lookR = nF(f_idx) >= posThr;
                    lookL = nF(f_idx) <= negThr;

                    if (isLeftK && lookR) || (isRightK && lookL)
                        oppCount = oppCount + 1;
                    elseif (isLeftK && lookL) || (isRightK && lookR)
                        sameCount = sameCount + 1;
                    end
                end
            end

            if lookCount > 0
                oppProportion(c, p)  = oppCount / lookCount;
                sameProportion(c, p) = sameCount / lookCount;
            end
        end
    end
end

%% --- GLOBAL STRATEGIES PLOT (Per condition) ---
% --- Central fixation ("middle") proportion, derived from same/opposite ---
midProportion = nan(numConditions, numParticipants);

if ismember('Global Strats Scatter', testsToRun)
    participantColors = lines(numParticipants);

    % --- Output coordinates (same, opposite, middle) per participant x condition ---
    % coordTable{p, c} = [sameProp, oppProp, midProp], each rounded to 2 decimals
    coordTable = cell(numParticipants, numConditions);
    for p = 1:numParticipants
        for c = 1:numConditions
            if ~validEyeData(c, p) || isnan(oppProportion(c, p)) || isnan(sameProportion(c, p))
                coordTable{p, c} = [NaN, NaN, NaN];
            else
                xVal   = round(sameProportion(c, p), 2);
                yVal   = round(oppProportion(c, p), 2);
                midVal = round(1 - sameProportion(c, p) - oppProportion(c, p), 2);
                coordTable{p, c} = [xVal, yVal, midVal];
            end
        end
    end

    % Build a human-readable string version: "(same, opposite, middle)"
    coordStrTable = cell(numParticipants, numConditions);
    for p = 1:numParticipants
        for c = 1:numConditions
            v = coordTable{p, c};
            if any(isnan(v))
                coordStrTable{p, c} = 'NA';
            else
                coordStrTable{p, c} = sprintf('(%.2f, %.2f, %.2f)', v(1), v(2), v(3));
            end
        end
    end

    % Display as a table with participant IDs as rows, conditions as columns
    rowNames = arrayfun(@(pid) sprintf('p%03d', pid), participantIDs, 'UniformOutput', false);
    colNames = matlab.lang.makeUniqueStrings( ...
        matlab.lang.makeValidName(condLabels));
    coordDisplay = cell2table(coordStrTable, 'RowNames', rowNames, 'VariableNames', colNames);

    fprintf('\n--- Participant x Condition Coordinates (Same, Opposite, Middle) ---\n');
    disp(coordDisplay);

    for c = 1:numConditions
        figG = figure('Units', 'inches', 'Position', [1, 1, 14, 12], 'Color', 'w'); hold on;

        plot([0 1], [0 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2.5, 'HandleVisibility', 'off');

        for p = 1:numParticipants
            if ~validEyeData(c, p) || isnan(oppProportion(c, p)) || isnan(sameProportion(c, p))
                continue; % exclude participants without enough valid eye data
            end

            xVal = sameProportion(c, p); yVal = oppProportion(c, p); midVal = 1 - xVal - yVal;

            plot(xVal, yVal, 'o', 'MarkerSize', 18, 'MarkerFaceColor', participantColors(p,:), 'MarkerEdgeColor', 'k', 'LineWidth', 2);

            %%%%% remove annotations since theres a bunch of people now :p
            % labelStr = sprintf(' P%03d (mid: %.2f)', participantIDs(p), midVal);
            % text(xVal + 0.015, yVal, labelStr, 'FontSize', 24, 'FontWeight', 'bold', 'Color', participantColors(p,:) * 0.7);
        end

        set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out', 'Box', 'on');
        xlabel('Fixation Proportion: Same Source', 'FontSize', 36, 'FontWeight', 'bold');
        ylabel('Fixation Proportion: Opposite Source', 'FontSize', 36, 'FontWeight', 'bold');
        title([condLabels{c} ': Global Viewing Strategy'], 'FontSize', 44, 'FontWeight', 'bold');
        xlim([0 1]); ylim([0 1]); grid on;

        saveas(figG, fullfile(figurePath, sprintf('SFN_GLOBAL_%.3g_%.3g.png', condStdHrz(c), condHazard(c))));
    end
end

%% --- GLOBAL STRATEGIES BAR CHART (Group Average Across Participants, per Condition) ---
% One figure. X-axis = condition, with 3 grouped bars per condition
% representing the AVERAGE Same / Opposite / Middle fixation proportions
% across all valid participants, with SEM error bars.
if ismember('Global Strats Bar', testsToRun)
    groupBarData = nan(numConditions, 3); % columns: Same, Opposite, Middle (mean)
    groupErrData = nan(numConditions, 3); % columns: Same, Opposite, Middle (SEM)

    for c = 1:numConditions
        validMask = validEyeData(c, :) & ~isnan(oppProportion(c, :)) & ~isnan(sameProportion(c, :));

        sVals = sameProportion(c, validMask);
        oVals = oppProportion(c, validMask);
        mVals = 1 - sVals - oVals;

        nValidC = sum(validMask);
        if nValidC > 0
            groupBarData(c, :) = [mean(sVals), mean(oVals), mean(mVals)];
            if nValidC > 1
                groupErrData(c, :) = [std(sVals)/sqrt(nValidC), std(oVals)/sqrt(nValidC), std(mVals)/sqrt(nValidC)];
            else
                groupErrData(c, :) = [0, 0, 0];
            end
        end
    end

    figAvg = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;

    b = bar(groupBarData, 'grouped');
    b(1).FaceColor = [0.20 0.55 0.85]; % Same
    b(2).FaceColor = [0.85 0.30 0.25]; % Opposite
    b(3).FaceColor = [0.60 0.60 0.60]; % Middle
    b(1).EdgeColor = 'k'; b(2).EdgeColor = 'k'; b(3).EdgeColor = 'k';
    b(1).LineWidth = 1.5; b(2).LineWidth = 1.5; b(3).LineWidth = 1.5;

    % Add error bars: compute x-positions of each bar within each group
    numBarsPerGroup = size(groupBarData, 2);
    groupWidth = min(0.8, numBarsPerGroup / (numBarsPerGroup + 1.5));
    for i = 1:numBarsPerGroup
        xPos = (1:numConditions) - groupWidth/2 + (2*i-1) * groupWidth / (2*numBarsPerGroup);
        errorbar(xPos, groupBarData(:, i), groupErrData(:, i), 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 12);
    end

    set(gca, 'FontSize', 24, 'LineWidth', 2, 'TickDir', 'out');
    set(gca, 'XTick', 1:numConditions, 'XTickLabel', condLabels, 'XTickLabelRotation', 20);
    ylabel('Fixation Proportion', 'FontSize', 32, 'FontWeight', 'bold');
    xlabel('Condition', 'FontSize', 32, 'FontWeight', 'bold');
    ylim([0 1]);
    title('Group Average Global Viewing Strategy (±SEM)', 'FontSize', 40, 'FontWeight', 'bold');
    legend(b, {'Same', 'Opposite', 'Middle'}, 'FontSize', 22, 'Location', 'bestoutside');
    grid on;

    % saveas(figAvg, fullfile(figurePath, 'GroupAvg_Global_Strategy_Bar.png'));
end

%% --- GLOBAL STRATEGIES BAR CHART (Per-Participant: Same/Opposite/Middle across Conditions) ---
% One figure per participant. X-axis = condition, with 3 grouped bars per
% condition representing Same / Opposite / Middle fixation proportions.
if ismember('Global Strats Individual', testsToRun)
    for p = 1:numParticipants
        pid = participantIDs(p);

        barData = nan(numConditions, 3); % columns: Same, Opposite, Middle
        for c = 1:numConditions
            if ~validEyeData(c, p) || isnan(oppProportion(c, p)) || isnan(sameProportion(c, p))
                continue; % leave as NaN (will show as gap in bar chart)
            end
            sVal = sameProportion(c, p);
            oVal = oppProportion(c, p);
            mVal = 1 - sVal - oVal;
            barData(c, :) = [sVal, oVal, mVal];
        end

        figB = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;

        b = bar(barData, 'grouped');
        b(1).FaceColor = [0.20 0.55 0.85]; % Same
        b(2).FaceColor = [0.85 0.30 0.25]; % Opposite
        b(3).FaceColor = [0.60 0.60 0.60]; % Middle
        b(1).EdgeColor = 'k'; b(2).EdgeColor = 'k'; b(3).EdgeColor = 'k';
        b(1).LineWidth = 1.5; b(2).LineWidth = 1.5; b(3).LineWidth = 1.5;

        set(gca, 'FontSize', 24, 'LineWidth', 2, 'TickDir', 'out');
        set(gca, 'XTick', 1:numConditions, 'XTickLabel', condLabels, 'XTickLabelRotation', 20);
        ylabel('Fixation Proportion', 'FontSize', 32, 'FontWeight', 'bold');
        xlabel('Condition', 'FontSize', 32, 'FontWeight', 'bold');
        ylim([0 1]);
        title(sprintf('Global Viewing Strategy: Participant p%03d', pid), 'FontSize', 40, 'FontWeight', 'bold');
        legend({'Same', 'Opposite', 'Middle'}, 'FontSize', 22, 'Location', 'bestoutside');
        grid on;

        % saveas(figB, fullfile(figurePath, sprintf('p%03d_Global_Strategy_Bar.png', pid)));
    end
end

%% --- COMPUTE CENTRAL FIXATION ("MIDDLE") PROPORTION ---
for c = 1:numConditions
    for p = 1:numParticipants
        if validEyeData(c, p) && ~isnan(oppProportion(c, p)) && ~isnan(sameProportion(c, p))
            midProportion(c, p) = 1 - sameProportion(c, p) - oppProportion(c, p);
        end
    end
end

%% --- DIAGNOSTIC: POOLED CENTRAL FIXATION DISTRIBUTION ---
if ismember('Central Fixation', testsToRun)
    pooledMid = midProportion(:);
    pooledMid = pooledMid(~isnan(pooledMid));

    medianSplit  = median(pooledMid);
    tertileSplit = quantile(pooledMid, [1/3, 2/3]);

    figMid = figure('Units', 'inches', 'Position', [1 1 10 7], 'Color', 'w'); hold on;

    histogram(pooledMid, 20, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'k');

    yl = ylim;
    h1 = plot([medianSplit medianSplit], yl, 'r-', 'LineWidth', 2.5);
    h2 = plot([tertileSplit(1) tertileSplit(1)], yl, 'b--', 'LineWidth', 2.5);
    plot([tertileSplit(2) tertileSplit(2)], yl, 'b--', 'LineWidth', 2.5);

    set(gca, 'FontSize', 18, 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Central Fixation Proportion (mid)', 'FontSize', 20);
    ylabel('Count', 'FontSize', 20);
    legend([h1, h2], ...
        {sprintf('Median split (%.2f)', medianSplit), ...
        sprintf('Tertile splits (%.2f, %.2f)', tertileSplit(1), tertileSplit(2))}, ...
        'FontSize', 14, 'Location', 'best');
    title(sprintf('Pooled Central Fixation Distribution (N=%d obs)', numel(pooledMid)), 'FontSize', 22);
    grid on;

    fprintf('\n--- Pooled Central Fixation (mid) Diagnostics ---\n');
    fprintf('N = %d observations (participant x condition)\n', numel(pooledMid));
    fprintf('Median split point: %.3f\n', medianSplit);
    fprintf('Tertile split points: %.3f, %.3f\n', tertileSplit(1), tertileSplit(2));
    fprintf('Min: %.3f | Max: %.3f | Mean: %.3f | SD: %.3f\n', ...
        min(pooledMid), max(pooledMid), mean(pooledMid), std(pooledMid));

    %% --- CLASSIFY LOW / MODERATE / HIGH CENTRAL FIXATION STRATEGY ---
    % Using pooled tertile cutoffs derived in Section 2f
    lowCut  = tertileSplit(1); % 0.06
    highCut = tertileSplit(2); % 0.15

    % centralFixCategory(c,p) = 1 (Low), 2 (Moderate), 3 (High), or NaN (invalid)
    centralFixCategory = nan(numConditions, numParticipants);

    for c = 1:numConditions
        for p = 1:numParticipants
            m = midProportion(c, p);
            if isnan(m), continue; end

            if m <= lowCut
                centralFixCategory(c, p) = 1; % Low
            elseif m <= highCut
                centralFixCategory(c, p) = 2; % Moderate
            else
                centralFixCategory(c, p) = 3; % High
            end
        end
    end

    centralFixLabels = {'Low Central Fixation', 'Moderate Central Fixation', 'High Central Fixation'};
    centralFixColors = [0.20 0.55 0.85; 0.90 0.60 0.10; 0.75 0.10 0.15];

    fprintf('\n--- Central Fixation Category Counts (Low <= %.2f < Moderate <= %.2f < High) ---\n', lowCut, highCut);
    for c = 1:numConditions
        nLow = sum(centralFixCategory(c,:) == 1);
        nMod = sum(centralFixCategory(c,:) == 2);
        nHigh = sum(centralFixCategory(c,:) == 3);
        fprintf('Condition %d (%s): Low=%d, Moderate=%d, High=%d\n', c, condLabels{c}, nLow, nMod, nHigh);
    end

    %% --- DIAGNOSTIC: WITHIN- VS BETWEEN-SUBJECT VARIANCE IN CENTRAL FIXATION ---

    % --- (1) Heatmap of centralFixCategory: participant x condition ---
    figHeat = figure('Units', 'inches', 'Position', [1 1 12 8], 'Color', 'w');
    catMapDisplay = centralFixCategory'; % participants as rows, conditions as columns

    imagesc(catMapDisplay, 'AlphaData', ~isnan(catMapDisplay));
    set(gca, 'Color', [0.85 0.85 0.85]); % NaN cells show as gray background

    colormap([centralFixColors]);
    caxis([1 3]);
    cb = colorbar('Ticks', [1 2 3], 'TickLabels', {'Low', 'Moderate', 'High'});
    cb.FontSize = 16;

    set(gca, 'FontSize', 16, 'XTick', 1:numConditions, 'XTickLabel', condLabels, ...
        'XTickLabelRotation', 20, 'YTick', 1:numParticipants, ...
        'YTickLabel', arrayfun(@(pid) sprintf('p%03d', pid), participantIDs, 'UniformOutput', false));
    xlabel('Condition', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('Participant', 'FontSize', 20, 'FontWeight', 'bold');
    title('Central Fixation Category by Participant x Condition', 'FontSize', 24, 'FontWeight', 'bold');

    %% --- EXPORT GLOBAL STRATEGY LABELS FOR ACCURACY SCRIPT ---
    % strategyMatrix(c,p) = 1 (Low), 2 (Moderate), 3 (High) central fixation, NaN = excluded/invalid
    % Rows = condIdx = (stdHrzIdx-1)*2 + hazardIdx ; Columns = participantIDs order
    strategyMatrix         = centralFixCategory;
    strategyParticipantIDs = participantIDs;
    strategyCondLabels     = condLabels;
    strategyNames          = centralFixLabels;

    strategySavePath = '/Users/janebaker/Desktop/triangles/taskData/centralFixationStrategies.mat';
    save(strategySavePath, 'strategyMatrix', 'strategyParticipantIDs', ...
        'strategyCondLabels', 'strategyNames');
    fprintf('\nSaved global strategy labels to %s\n', strategySavePath);
end

%% --- DIAGNOSTIC: POOLED OPPOSITE FIXATION DISTRIBUTION ---
if ismember('Opposite Fixation', testsToRun)
    pooledOpp = oppProportion(:);
    pooledOpp = pooledOpp(~isnan(pooledOpp));

    medianSplit  = median(pooledOpp);
    tertileSplit = quantile(pooledOpp, [1/3, 2/3]);

    figOpp = figure('Units', 'inches', 'Position', [1 1 10 7], 'Color', 'w'); hold on;

    histogram(pooledOpp, 20, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'k');

    yl = ylim;
    h1 = plot([medianSplit medianSplit], yl, 'r-', 'LineWidth', 2.5);
    h2 = plot([tertileSplit(1) tertileSplit(1)], yl, 'b--', 'LineWidth', 2.5);
    plot([tertileSplit(2) tertileSplit(2)], yl, 'b--', 'LineWidth', 2.5);

    set(gca, 'FontSize', 18, 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Opposite Fixation Proportion', 'FontSize', 20);
    ylabel('Count', 'FontSize', 20);
    legend([h1, h2], ...
        {sprintf('Median split (%.2f)', medianSplit), ...
        sprintf('Tertile splits (%.2f, %.2f)', tertileSplit(1), tertileSplit(2))}, ...
        'FontSize', 14, 'Location', 'best');
    title(sprintf('Pooled Opposite Fixation Distribution (N=%d obs)', numel(pooledOpp)), 'FontSize', 22);
    grid on;

    fprintf('\n--- Pooled Opposite Fixation Diagnostics ---\n');
    fprintf('N = %d observations (participant x condition)\n', numel(pooledOpp));
    fprintf('Median split point: %.3f\n', medianSplit);
    fprintf('Tertile split points: %.3f, %.3f\n', tertileSplit(1), tertileSplit(2));
    fprintf('Min: %.3f | Max: %.3f | Mean: %.3f | SD: %.3f\n', ...
        min(pooledOpp), max(pooledOpp), mean(pooledOpp), std(pooledOpp));

    %% --- CLASSIFY LOW / MODERATE / HIGH OPPOSITE FIXATION STRATEGY ---
    % Using pooled tertile cutoffs
    lowCut  = tertileSplit(1);
    highCut = tertileSplit(2);

    % oppositeFixCategory(c,p) = 1 (Low), 2 (Moderate), 3 (High), or NaN (invalid)
    oppositeFixCategory = nan(numConditions, numParticipants);

    for c = 1:numConditions
        for p = 1:numParticipants
            m = oppProportion(c, p);
            if isnan(m), continue; end

            if m <= lowCut
                oppositeFixCategory(c, p) = 1; % Low
            elseif m <= highCut
                oppositeFixCategory(c, p) = 2; % Moderate
            else
                oppositeFixCategory(c, p) = 3; % High
            end
        end
    end

    oppositeFixLabels = {'Low Opposite Fixation', 'Moderate Opposite Fixation', 'High Opposite Fixation'};
    oppositeFixColors = [0.20 0.55 0.85; 0.90 0.60 0.10; 0.75 0.10 0.15];

    fprintf('\n--- Opposite Fixation Category Counts (Low <= %.2f < Moderate <= %.2f < High) ---\n', lowCut, highCut);
    for c = 1:numConditions
        nLow = sum(oppositeFixCategory(c,:) == 1);
        nMod = sum(oppositeFixCategory(c,:) == 2);
        nHigh = sum(oppositeFixCategory(c,:) == 3);
        fprintf('Condition %d (%s): Low=%d, Moderate=%d, High=%d\n', c, condLabels{c}, nLow, nMod, nHigh);
    end

    %% --- DIAGNOSTIC: WITHIN- VS BETWEEN-SUBJECT VARIANCE IN OPPOSITE FIXATION ---

    % --- (1) Heatmap of oppositeFixCategory: participant x condition ---
    figHeat = figure('Units', 'inches', 'Position', [1 1 12 8], 'Color', 'w');
    catMapDisplay = oppositeFixCategory'; % participants as rows, conditions as columns

    imagesc(catMapDisplay, 'AlphaData', ~isnan(catMapDisplay));
    set(gca, 'Color', [0.85 0.85 0.85]); % NaN cells show as gray background

    colormap([oppositeFixColors]);
    caxis([1 3]);
    cb = colorbar('Ticks', [1 2 3], 'TickLabels', {'Low', 'Moderate', 'High'});
    cb.FontSize = 16;

    set(gca, 'FontSize', 16, 'XTick', 1:numConditions, 'XTickLabel', condLabels, ...
        'XTickLabelRotation', 20, 'YTick', 1:numParticipants, ...
        'YTickLabel', arrayfun(@(pid) sprintf('p%03d', pid), participantIDs, 'UniformOutput', false));
    xlabel('Condition', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('Participant', 'FontSize', 20, 'FontWeight', 'bold');
    title('Opposite Fixation Category by Participant x Condition', 'FontSize', 24, 'FontWeight', 'bold');

    %% --- EXPORT OPPOSITE FIXATION STRATEGY LABELS FOR ACCURACY SCRIPT ---
    % strategyMatrix(c,p) = 1 (Low), 2 (Moderate), 3 (High) opposite fixation, NaN = excluded/invalid
    % Rows = condIdx = (stdHrzIdx-1)*2 + hazardIdx ; Columns = participantIDs order
    strategyMatrix         = oppositeFixCategory;
    strategyParticipantIDs = participantIDs;
    strategyCondLabels     = condLabels;
    strategyNames          = oppositeFixLabels;

    strategySavePath = '/Users/janebaker/Desktop/triangles/taskData/oppositeFixationStrategies.mat';
    save(strategySavePath, 'strategyMatrix', 'strategyParticipantIDs', ...
        'strategyCondLabels', 'strategyNames');
    fprintf('\nSaved opposite fixation strategy labels to %s\n', strategySavePath);
end

%% --- MEAN OPPOSITE-SOURCE FIXATION ACROSS CONDITIONS ---
if ismember('Opposite Fixation Means', testsToRun)
    % Mean and SEM across valid participants
    meanOpp = mean(oppProportion, 2, 'omitnan');
    nOpp = sum(~isnan(oppProportion), 2);
    semOpp = std(oppProportion, 0, 2, 'omitnan') ./ sqrt(nOpp);

    % Use participants with data in all 6 conditions for repeated-measures tests
    completeOpp = oppProportion(:, all(~isnan(oppProportion), 1));

    % Overall Friedman test
    pFriedman = friedman(completeOpp', 1, 'off');

    % Pairwise Wilcoxon signed-rank tests
    pairs = nchoosek(1:numConditions, 2);
    rawP = nan(size(pairs, 1), 1);

    for k = 1:size(pairs, 1)
        rawP(k) = signrank(completeOpp(pairs(k,1), :), ...
            completeOpp(pairs(k,2), :), ...
            'method', 'exact');
    end

    % Holm correction for multiple comparisons
    [sortedP, sortIdx] = sort(rawP);
    m = length(rawP);
    adjustedSorted = min(1, cummax(sortedP .* (m:-1:1)'));
    adjustedP = nan(size(rawP));
    adjustedP(sortIdx) = adjustedSorted;

    % Identify significant comparisons
    sigIdx = find(adjustedP < 0.05);
    sigPairs = pairs(sigIdx, :);
    sigP = adjustedP(sigIdx);

    % --- Plot ---
    figOpp = figure('Units', 'inches', ...
        'Position', [1 1 14 10], 'Color', 'w');
    hold on;

    bar(1:numConditions, meanOpp, 0.7, ...
        'FaceColor', [0.20 0.55 0.85], ...
        'EdgeColor', 'k', 'LineWidth', 1.5);

    errorbar(1:numConditions, meanOpp, semOpp, ...
        'k', 'LineStyle', 'none', ...
        'LineWidth', 2, 'CapSize', 12);

    % Add significant pairwise comparisons
    % Use normalized coordinates so labels remain visible when ylim changes
    sigLabels = cell(length(sigP) + 1, 1);
    sigLabels{1} = 'Significant comparisons (Holm-corrected):';

    for k = 1:length(sigP)
        sigLabels{k+1} = sprintf('Cond %d vs %d: p = %.4f', ...
            sigPairs(k,1), sigPairs(k,2), sigP(k));
    end

    if isempty(sigP)
        sigLabels{2} = 'None (p < 0.05)';
    end

    text(0.98, 0.98, sigLabels, ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 14, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', 'w', ...
        'EdgeColor', 'k', ...
        'Margin', 8);


    set(gca, 'FontSize', 18, 'LineWidth', 1.5, 'TickDir', 'out');
    set(gca, 'XTick', 1:numConditions, ...
        'XTickLabel', condLabels, 'XTickLabelRotation', 20);

    xlabel('Condition', 'FontSize', 22, 'FontWeight', 'bold');
    ylabel('Mean Opposite-Source Fixation Proportion', ...
        'FontSize', 22, 'FontWeight', 'bold');

    title(sprintf('Opposite-Source Fixation Across Conditions (Friedman p = %.4g)', ...
        pFriedman), 'FontSize', 23, 'FontWeight', 'bold');

    subtitle('Mean ± SEM; pairwise Wilcoxon tests with Holm correction', ...
        'FontSize', 15);

    ylim([0.5 0.7]);%([0 max(1.05, yStart + length(sigP)*yStep + 0.03)]);
    xlim([0.5 numConditions+0.5]);
    grid on;

    % Print results
    fprintf('\n--- Opposite-Source Fixation Across Conditions ---\n');
    for c = 1:numConditions
        fprintf('Condition %d: Mean = %.4f, SEM = %.4f, N = %d\n', ...
            c, meanOpp(c), semOpp(c), nOpp(c));
    end

    fprintf('\nFriedman test: p = %.6f\n', pFriedman);
    fprintf('Significant pairwise comparisons (Holm-corrected):\n');

    for k = 1:length(sigP)
        fprintf('Condition %d vs %d: p = %.6f\n', ...
            sigPairs(k,1), sigPairs(k,2), sigP(k));
    end

    saveas(figOpp, fullfile(figurePath, 'SFN_Opp_Fix_Means.png'));
end

%% --- LOCAL STRATEGIES PLOT (INDIVIDUAL PARTICIPANTS) ---
if ismember('Local Strats Individual', testsToRun)
    condColors = lines(numConditions);

    for p = 1:numParticipants
        pid = participantIDs(p);

        figI = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;
        legH = [];
        legNames = {};

        yline(0.5, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'HandleVisibility', 'off');

        for c = 1:numConditions
            if ~validEyeData(c, p), continue; end

            dataMat = indivDataStorage{c, p};
            if isempty(dataMat), continue; end

            % --- TRANSFORMATION ---
            dataMat = 1 - (dataMat / 2);
            avg = mean(dataMat, 1, 'omitnan');
            timeAxis =  ((1:length(avg)) - (timeBefore + 1)) * 0.05;

            if size(dataMat, 1) > 1
                sem = std(dataMat, 0, 1, 'omitnan') ./ sqrt(size(dataMat, 1));
                fill([timeAxis, fliplr(timeAxis)], [avg+sem, fliplr(avg-sem)], ...
                    condColors(c,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end

            h = plot(timeAxis, avg, 'Color', condColors(c,:), 'LineWidth', 4);
            legH = [legH, h];
            legNames{end+1} = condLabels{c};
        end

        set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out');
        xlabel('Time (Seconds)', 'FontSize', 36, 'FontWeight', 'bold');
        ylabel('Gaze Position', 'FontSize', 36, 'FontWeight', 'bold');
        ylim([0 1]);
        yticks([0, 0.5, 1]);
        yticklabels({'Old Belief Source', 'Center', 'New Belief Source'});

        title(sprintf('Local Gaze Shifts: Participant p%03d', pid), 'FontSize', 44, 'FontWeight', 'bold');

        if ~isempty(legH)
            legend(legH, legNames, 'FontSize', 22, 'Location', 'best');
        end

        xlim([-10 10]); grid on;

        % saveas(figI, fullfile(figurePath, sprintf('IndLocStrat_p%03d.png', pid)));
    end
end

%% --- LOCAL STRATEGIES PLOT (GROUPED BY STRATEGY) ---
if ismember('Grouped Local Strats', testsToRun)
    localStrategyMatrix = [
        2   2   2   1   1   1;  % p001
        2   1   1   1   2   1;  % p002
        1   2   2   1   2   2;  % p003
        3 NaN   3   3 NaN   3;  % p004
        1   1   1   1   1   1;  % p005
        1   1   1   1   1   1;  % p006
        NaN NaN NaN NaN NaN NaN;  % p007
        1   2   1   1   1   1;  % p008
        2   3   1   3   3   3;  % p009
        2   2   2   2   2   2;  % p010
        3   2   2   2   2   2;  % p011
        2   1   1   1   3   3;  % p012
        NaN NaN NaN NaN NaN NaN;  % p013
        1   1   1   3   1   3;  % p014
        2   1   1   1   1   1;  % p015
        3   1   3   1   3   3;  % p016
        3   1   1   1   1   1;  % p017
        2   1   1   3   1   1;  % p018
        1   2   3   1   2   1;  % p019
        3   3   3   3   3   3;  % p020
        2   1   2   3   1   2;  % p021
        3   1   2   1   1   1;  % p022
        2   1   2   2   2   3;  % p023
        2   2   2   2   2   2;  % p024
        3   3   3   3   3   3;  % p025
        2   1   1   3   3   2;  % p026
        3   1   1   3 NaN   1;  % p027
        NaN NaN NaN NaN NaN NaN;  % p028
        2   2   1   1   3   1;  % p029
        2   1   1   1   1   1;  % p030
        1   2   1   1   1   1;  % p031
        ];

    stratAssignment = cell(numConditions, 1);

    for c = 1:numConditions
        stratAssignment{c} = localStrategyMatrix(participantIDs, c);
    end

    stratNames = {
        'Peak at Switch'
        'Decreasing at Switch'
        'Mixed/Balanced'
        };

    stratColors = [
        0.00 0.45 0.74;
        0.85 0.33 0.10;
        0.47 0.67 0.19
        ];

    for c = 1:numConditions
        figL = figure('Units', 'inches', 'Position', [1, 1, 14, 10], 'Color', 'w'); hold on;
        legH = [];
        legNames = {};

        yline(0.5, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'HandleVisibility', 'off');

        for s = 1:numel(stratNames)
            % Determine which participants belong to this strategy AND have
            % valid eye data for this condition
            inStrat = (stratAssignment{c} == s) & validEyeData(c, :)';
            pidsInStrat = participantIDs(inStrat);
            nStrat = numel(pidsInStrat);

            if nStrat == 0, continue; end

            % Gather each valid participant's mean trace for this condition
            dataMat = [];
            for pp = find(inStrat)'
                thisData = indivDataStorage{c, pp};
                if isempty(thisData), continue; end
                dataMat = [dataMat; mean(thisData, 1, 'omitnan')];
            end
            if isempty(dataMat), continue; end
            nStrat = size(dataMat, 1);

            % --- TRANSFORMATION ---
            dataMat = 1 - (dataMat / 2);
            avg = mean(dataMat, 1, 'omitnan');
            timeAxis = ((1:length(avg)) - (timeBefore + 1)) * 0.05;

            if size(dataMat, 1) > 1
                sem = std(dataMat, 0, 1, 'omitnan') ./ sqrt(size(dataMat, 1));
                fill([timeAxis, fliplr(timeAxis)], [avg+sem, fliplr(avg-sem)], stratColors(s,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end

            h = plot(timeAxis, avg, 'Color', stratColors(s,:), 'LineWidth', 5);
            legH = [legH, h];
            legNames{end+1} = sprintf('%s (N=%d)', stratNames{s}, nStrat);
        end

        set(gca, 'FontSize', 28, 'LineWidth', 3, 'TickDir', 'out');
        xlabel('Time (Seconds)', 'FontSize', 36, 'FontWeight', 'bold');
        ylabel('Gaze Position', 'FontSize', 36, 'FontWeight', 'bold');
        ylim([0 1]);
        yticks([0, 0.5, 1]);
        yticklabels({'Old Belief Source', 'Center', 'New Belief Source'});

        title(condLabels{c}, 'FontSize', 44, 'FontWeight', 'bold');
        if ~isempty(legH)
            legend(legH, legNames, 'FontSize', 26, 'Location', 'best');
        end
        xlim([-10 10]); grid on;

        % saveas(figL, fullfile(figurePath, [strrep(condLabels{c}, ' ', '') '_Local_Strategies.png']));
    end
end