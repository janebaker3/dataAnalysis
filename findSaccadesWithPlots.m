function SACs = findSaccadesWithPlots(x, y, varargin)
% findSaccadesWithPlots  Detect saccades and review each one interactively.
%
% Drop-in replacement for findSaccadesA.  Detects saccades using velocity
% and acceleration thresholds plus a minimum distance criterion, then shows
% a 4-panel figure for every detected saccade and waits for a keypress
% before advancing to the next one.
%
% Usage
%   SACs = findSaccadesWithPlots(x, y)
%   SACs = findSaccadesWithPlots(x, y, 'param', value, ...)
%
% Required inputs
%   x, y     Column vectors of horizontal / vertical gaze position (samples)
%
% Optional name-value parameters (defaults tuned to p9 data, position in pixels at 1000 Hz)
%   'd_min'    Minimum saccade displacement in px.            Default 50.
%   'vp_min'   Minimum peak velocity (px/sample).             Default 8.0.
%              (Natural break between fixation and saccade in this dataset is
%               ~2-3 px/sample; 8 px/sample ≈ 200-300 deg/s peak is firmly
%               in saccade territory.)
%   'vi_min'   Velocity threshold for onset/offset (px/sample). Default 3.0.
%              (Counts drop from hundreds to tens above this value, indicating
%               the fixation-to-saccade boundary in the velocity distribution.)
%   'a_min'    Minimum peak |acceleration| (px/sample^2).    Default 1.0.
%              (p99 of fixation acceleration ≈ 1.7; threshold of 1.0 rejects
%               most noise while passing the rising/falling edge of saccades.)
%   'sr'       Sample rate in Hz — used only for time-axis labels.  Default 1000.
%   'pad'      Samples before/after each saccade shown in plots.   Default 50.
%   'show_fig' true/false — display interactive figure.  Default true.
%
% Output  SACs  —  N x 8 matrix, one row per saccade
%   col 1  latency       start sample index
%   col 2  duration      duration in samples
%   col 3  max speed     peak velocity during saccade
%   col 4  avg speed     mean velocity during saccade
%   col 5  end x         x position at saccade offset
%   col 6  end y         y position at saccade offset
%   col 7  dist raw      cumulative sample-by-sample distance
%   col 8  dist vect     straight-line displacement (start → end)

% -------------------------------------------------------------------------
% Parse inputs
% -------------------------------------------------------------------------
p = struct('d_min', 50, 'vp_min', 8.0, 'vi_min', 3.0, 'a_min', 1.0, ...
           'sr', 1000, 'pad', 50, 'show_fig', false);

for k = 1:2:length(varargin)
    p.(varargin{k}) = varargin{k+1};
end

x = x(:);
y = y(:);
N = length(x);

% -------------------------------------------------------------------------
% Compute velocity and acceleration (both length N, zero-padded at start)
% -------------------------------------------------------------------------
step = sqrt(diff(x).^2 + diff(y).^2);   % N-1 inter-sample distances
vel  = [0; step];                         % N,  vel(i) = dist from i-1 to i
acc  = [0; abs(diff(vel))];               % N,  |Δvel| per sample

% -------------------------------------------------------------------------
% Detect saccade candidates: contiguous runs above velocity threshold
% -------------------------------------------------------------------------
above  = vel >= p.vi_min;
onsets  = find(diff([0; above]) ==  1);
offsets = find(diff([above; 0]) == -1);

SACs = zeros(0, 8);

for k = 1:length(onsets)
    on  = onsets(k);
    off = offsets(k);
    idx = on:off;

    peak_vel  = max(vel(idx));
    peak_acc  = max(acc(idx));
    raw_dist  = sum(step(on : min(off, N-1)));
    vect_dist = sqrt((x(off) - x(on))^2 + (y(off) - y(on))^2);

    if peak_vel  < p.vp_min,  continue; end
    if peak_acc  < p.a_min,   continue; end
    if vect_dist < p.d_min,   continue; end

    SACs(end+1, :) = [on, length(idx), peak_vel, mean(vel(idx)), ...
                      x(off), y(off), raw_dist, vect_dist];      %#ok<AGROW>
end

fprintf('findSaccadesWithPlots: found %d saccades\n', size(SACs, 1));

if isempty(SACs) || ~p.show_fig
    return
end

% -------------------------------------------------------------------------
% Pre-compute fixed axis limits from all data across all saccade windows
% -------------------------------------------------------------------------
nSacs = size(SACs, 1);

% Collect every sample index that will appear in any window
all_idx = [];
max_win_ms = 0;
for k = 1:nSacs
    on      = SACs(k, 1);
    off     = on + SACs(k, 2) - 1;
    win_on  = max(1, on  - p.pad);
    win_off = min(N, off + p.pad);
    all_idx = union(all_idx, win_on:win_off);
    max_win_ms = max(max_win_ms, (win_off - win_on) / p.sr * 1000);
end

xlims_xy  = [min(x(all_idx), [], 'omitnan'), max(x(all_idx), [], 'omitnan')];
ylims_xy  = [min(y(all_idx), [], 'omitnan'), max(y(all_idx), [], 'omitnan')];
ylims_pos = [min([x(all_idx); y(all_idx)], [], 'omitnan'), ...
             max([x(all_idx); y(all_idx)], [], 'omitnan')];
xlims_t   = [0, max_win_ms];
ylims_vel = [0, prctile(vel(all_idx), 99) * 1.05];
ylims_acc = [0, prctile(acc(all_idx), 99) * 1.05];

% -------------------------------------------------------------------------
% Interactive review — one saccade at a time, keypress to advance
% -------------------------------------------------------------------------
fig = figure('Name', 'Saccade Inspector', 'NumberTitle', 'off', ...
             'KeyPressFcn', @(~,~) uiresume(gcbf));

for k = 1:nSacs
    on  = SACs(k, 1);
    off = on + SACs(k, 2) - 1;

    % Window around the saccade
    win_on  = max(1, on  - p.pad);
    win_off = min(N, off + p.pad);
    win     = win_on : win_off;

    % Time axis in ms, zeroed at window start
    t_ms        = (win - win_on) / p.sr * 1000;
    sac_on_ms   = (on  - win_on) / p.sr * 1000;
    sac_off_ms  = (off - win_on) / p.sr * 1000;

    % ----- panel 1: x-y trajectory ----------------------------------------
    ax1 = subplot(4, 1, 1); cla(ax1); hold(ax1, 'on');
    plot(ax1, x(win), y(win), 'k-', 'LineWidth', 1)
    plot(ax1, x(on),  y(on),  'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'LineWidth', 1.5)
    plot(ax1, x(off), y(off), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 1.5)
    xlabel(ax1, 'X'); ylabel(ax1, 'Y');
    title(ax1, 'Trajectory (X vs Y)');
    legend(ax1, 'path', 'onset', 'offset', 'Location', 'best');
    xlim(ax1, xlims_xy); ylim(ax1, ylims_xy);

    % ----- panel 2: x and y vs time ---------------------------------------
    ax2 = subplot(4, 1, 2); cla(ax2); hold(ax2, 'on');
    plot(ax2, t_ms, x(win), 'b-', 'LineWidth', 1.2)
    plot(ax2, t_ms, y(win), 'r-', 'LineWidth', 1.2)
    plotSaccadeBounds(ax2, sac_on_ms, sac_off_ms);
    xlabel(ax2, 'Time (ms)'); ylabel(ax2, 'Position (px)');
    title(ax2, 'Position vs Time');
    legend(ax2, 'X', 'Y', 'onset', 'offset', 'Location', 'best');
    xlim(ax2, xlims_t); ylim(ax2, ylims_pos);

    % ----- panel 3: velocity vs time --------------------------------------
    ax3 = subplot(4, 1, 3); cla(ax3); hold(ax3, 'on');
    plot(ax3, t_ms, vel(win), 'k-', 'LineWidth', 1.2)
    yline(ax3, p.vi_min, 'b:', 'LineWidth', 1, 'Label', 'thresh');
    plotSaccadeBounds(ax3, sac_on_ms, sac_off_ms);
    xlabel(ax3, 'Time (ms)'); ylabel(ax3, 'Velocity (px/sample)');
    title(ax3, sprintf('Velocity vs Time   [peak = %.2f]', SACs(k, 3)));
    xlim(ax3, xlims_t); ylim(ax3, ylims_vel);

    % ----- panel 4: acceleration vs time ----------------------------------
    ax4 = subplot(4, 1, 4); cla(ax4); hold(ax4, 'on');
    plot(ax4, t_ms, acc(win), 'k-', 'LineWidth', 1.2)
    yline(ax4, p.a_min, 'b:', 'LineWidth', 1, 'Label', 'thresh');
    plotSaccadeBounds(ax4, sac_on_ms, sac_off_ms);
    xlabel(ax4, 'Time (ms)'); ylabel(ax4, 'Acceleration (px/sample²)');
    title(ax4, 'Acceleration vs Time');
    xlim(ax4, xlims_t); ylim(ax4, ylims_acc);

    sgtitle(fig, sprintf('Saccade %d / %d   |   duration: %d samples   |   displacement: %.1f px   |   press any key for next', ...
        k, nSacs, SACs(k, 2), SACs(k, 8)));

    drawnow;

    if k < nSacs
        uiwait(fig);
        if ~ishandle(fig)   % user closed the window — stop early
            break
        end
    end
end

if ishandle(fig)
    sgtitle(fig, sprintf('Done — %d saccades reviewed.  Close window to exit.', nSacs));
    drawnow;
end

end % function findSaccadesWithPlots

% -------------------------------------------------------------------------
% Helper: draw onset/offset vertical lines on an axes
% -------------------------------------------------------------------------
function plotSaccadeBounds(ax, on_ms, off_ms)
    yl = ylim(ax);
    plot(ax, [on_ms  on_ms],  yl, 'g--', 'LineWidth', 1.5)
    plot(ax, [off_ms off_ms], yl, 'r--', 'LineWidth', 1.5)
    ylim(ax, yl);   % restore in case plot rescaled
end