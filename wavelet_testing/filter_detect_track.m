%% I. Read Video and Prepare Timestamps

data = load ('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_720p.mat');

video = data.filteredFramesGray;
times = data.filteredTimeindeces;

% Short snippet to get data on correct form
[height, width] = size(video{1});

% Preallocate a 3D matrix for the frames
numFrames = length(video);
eta = zeros(height, width, numFrames, 'uint8'); % Use 'uint8' for grayscale images

% Populate the 3D matrix
for t = 1:numFrames
    eta(:, :, t) = video{t};
end

disp('Data read and converted to correct form.');

%% MEAN SUBTRACTION TO REMOVE THE BLACK CEILING PANELS
% Compute mean frame across time. The output is a double array.
mean_frame = mean(eta, 3);  % 1080x1920 (double)

% Convert eta to double, then subtract the mean frame.
eta_meansub = double(eta) - mean_frame;

%% II. Wavelet Analysis and Filtering
% Parameters for wavelet filtering
scales = 1:15;
selected_scale = 6;
W_thr = 40;
eccentricity_threshold = 0.85;
solidity_threshold = 0.6;
[x_dim, y_dim, ~] = size(eta);
filtered_all_structures = zeros(x_dim, y_dim, numFrames);
filtered_dimples = zeros(x_dim, y_dim, numFrames);

%% III. Tracking Structures
% Initialize tracking structure (adding an "active" flag)
tracks = struct('id', {}, 'centroids', {}, 'frames', {}, 'active', {});
nextTrackId = 1;
baseYShift = 35;  % Base offset per time unit
radiusFactor = 1.1; % Adjusts the factor for allowed distance from predicted position will be baseYShift*radiusFactor
maxSearchTime = 20;

for t_index = 8:58
    currentTime = times(t_index);
    disp(currentTime)
    snapshot = eta_meansub(:, :, t_index);
    
    % Wavelet transform and filtering (same as before)
    cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);
    wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);
    mask = wavelet_coefficients > W_thr;
    filtered_coefficients = wavelet_coefficients .* mask;
    connected_components = bwconncomp(mask);
    region_props = regionprops(connected_components, 'Area', 'Eccentricity', 'Solidity', 'Centroid');
    validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
                  [region_props.Solidity] > solidity_threshold);
    eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
    filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
    
    filtered_all_structures(:, :, t_index) = filtered_coefficients;
    filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
    
    % Extract centroids of valid regions
    if isempty(validIdx)
        centroids = [];
    else
        centroids = cat(1, region_props(validIdx).Centroid);  % Each row: [x y]
    end
    numDetections = size(centroids, 1);
    
    % IV. Identify/Match Structures (Tracking)
    numTracks = length(tracks);
    costMatrix = Inf(numDetections, numTracks);
    for i = 1:numDetections
        for j = 1:numTracks
            if ~tracks(j).active
                continue;  % Skip dead tracks
            end
            
            lastTime = tracks(j).frames(end);
            dt = currentTime - lastTime;
            
            % --- Skip if the time gap is more than maxSearchTime frames ---
            if dt > maxSearchTime
                continue;  % costMatrix stays Inf, effectively ignoring this track
            end
            
            % Predicted position: same x, y shifted by dt*baseYShift
            predicted = [tracks(j).centroids(end,1), ...
                         tracks(j).centroids(end,2) + dt*baseYShift];
            allowedDistance = dt * baseYShift * radiusFactor;
            
            % Compute distance from detection to predicted position
            d = norm(centroids(i,:) - predicted);
            
            if d <= allowedDistance
                costMatrix(i,j) = d;
            end
        end
    end
    
    % Greedy assignment of detections to tracks
    detectionTrackIDs = zeros(numDetections, 1);
    assignments = [];
    if ~isempty(costMatrix)
        while true
            [minVal, idx] = min(costMatrix(:));
            if isinf(minVal), break; end
            [detIdx, trackIdx] = ind2sub(size(costMatrix), idx);
            assignments = [assignments; detIdx, trackIdx, minVal];  %#ok<AGROW>
            detectionTrackIDs(detIdx) = tracks(trackIdx).id;
            costMatrix(detIdx, :) = Inf;
            costMatrix(:, trackIdx) = Inf;
        end
    end
    
    % Update assigned tracks with new detections
    if ~isempty(assignments)
        for k = 1:size(assignments, 1)
            detIdx = assignments(k, 1);
            trackIdx = assignments(k, 2);
            tracks(trackIdx).centroids(end+1, :) = centroids(detIdx, :);
            tracks(trackIdx).frames(end+1) = currentTime;
        end
    end

    % ***** FIX: Mark tracks not updated in the current frame as inactive *****
    for j = 1:length(tracks)
        if tracks(j).active && tracks(j).frames(end) < currentTime
             tracks(j).active = false;
        end
    end
    
    % Start new tracks for unassigned detections
    for i = 1:numDetections
        if detectionTrackIDs(i) == 0
            tracks(nextTrackId).id = nextTrackId;
            tracks(nextTrackId).centroids = centroids(i, :);
            tracks(nextTrackId).frames = currentTime;
            tracks(nextTrackId).active = true;
            detectionTrackIDs(i) = nextTrackId;
            nextTrackId = nextTrackId + 1;
        end
    end
    
    % Declare lost tracks as dead if not updated in current frame (and if they had multiple updates)
    for j = 1:length(tracks)
        if tracks(j).active && tracks(j).frames(end) < currentTime && numel(tracks(j).frames) >= 2
            tracks(j).active = false;
        end
    end

    % === Optional Visualization === %
    figure(1); clf;
    imagesc(wavelet_coefficients); colormap gray; hold on;
    colorbar;
    if ~isempty(centroids)
        plot(centroids(:,1), centroids(:,2), 'ro', 'MarkerSize', 22);
        for i = 1:numDetections
            % Label the detection with its associated track ID
            text(centroids(i,1)+22, centroids(i,2)+22, num2str(detectionTrackIDs(i)), ...
                 'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    title(['Time = ' num2str(currentTime)]);
    drawnow;

end

%% V. Return Tracks with Lifetime Above a Threshold
lifetimeThreshold = 5;  % Adjust as needed (in number of frames)
numTracks = length(tracks);
trackInfo = struct('id', {}, 'lifetime', {}, 'coordinates', {});
for i = 1:numTracks
    if ~isempty(tracks(i).frames)
        % Lifetime defined as number of frames the structure was tracked.
        lifetime = numel(tracks(i).frames);
        trackInfo(end+1) = struct('id', tracks(i).id, 'lifetime', lifetime, 'coordinates', {tracks(i).centroids});
    end
end

disp('Track Information:');
for i = 1:length(trackInfo)
    if trackInfo(i).lifetime > lifetimeThreshold
    fprintf('Track %d: Lifetime = %d frames\n', trackInfo(i).id, trackInfo(i).lifetime);
    disp(trackInfo(i).coordinates);
    end
end

%% VI. Plot Tracks of Structures with Long Lifetime
figure; hold on;
colors = lines(numel(trackInfo));  % Generate distinct colors

for i = 1:length(trackInfo)
    if trackInfo(i).lifetime > lifetimeThreshold
        coords = trackInfo(i).coordinates;
        % Plot the track line and markers
        plot(coords(:,1), coords(:,2), '-o', 'Color', colors(i,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('Track %d', trackInfo(i).id));
        % Optionally mark start (green square) and end (red square)
        %plot(coords(1,1), coords(1,2), 'gs', 'MarkerSize',10, 'MarkerFaceColor', 'g');
        %plot(coords(end,1), coords(end,2), 'rs', 'MarkerSize',10, 'MarkerFaceColor', 'r');
    end
end

title(sprintf('Tracks with Lifetime > %d Frames', lifetimeThreshold));
xlabel('X coordinate'); ylabel('Y coordinate');
legend('show');
grid on; hold off;

%% -- Example code to inspect track #19 in detail --

% Suppose you want to inspect the centroids for track ID = 19
targetTrackID = 136;

% Find which element in 'tracks' has this ID
trackIndex = find([tracks.id] == targetTrackID, 1);

if isempty(trackIndex)
    fprintf('No track found with ID = %d.\n', targetTrackID);
else
    % Extract the centroids and frame times
    coords = tracks(trackIndex).centroids;   % Nx2 array, regionprops order = [x, y]
    frameTimes = tracks(trackIndex).frames;  % Nx1 array of timestamps
    
    % Print them in a table to see the progression
    T = table(frameTimes(:), coords(:,1), coords(:,2), ...
        'VariableNames', {'FrameTime','X_Col','Y_Row'});
    disp(T);
    
    % Plot the coordinates vs. frameTimes to visualize
    figure('Name','Track 19 Inspection','Color','w');
    
    subplot(1,2,1);
    plot(frameTimes, coords(:,1), '-o','LineWidth',2);
    xlabel('Frame Time'); ylabel('X (Column)'); grid on;
    title('X-Coordinate (Columns) vs. Time');
    
    subplot(1,2,2);
    plot(frameTimes, coords(:,2), '-o','LineWidth',2);
    xlabel('Frame Time'); ylabel('Y (Row)'); grid on;
    title('Y-Coordinate (Rows) vs. Time');
    
    % If you suspect row/column confusion, you can swap coords(:,1) and coords(:,2)
    % and re-plot to see if that yields more reasonable results:
    % figure; plot(coords(:,2), coords(:,1), 'o-'); ...
end

%% Plot relative traveling position (y displacement relative to first detection) vs. relative time
figure; hold on;
colors = lines(length(trackInfo));  % Generate distinct colors

for i = 1:length(trackInfo)
    if trackInfo(i).lifetime > lifetimeThreshold
        coords = trackInfo(i).coordinates;  % Each row: [x, y]
        % Retrieve the timestamps. If not stored in trackInfo, get them from tracks:
        if isfield(trackInfo(i), 'times')
            timeStamps = trackInfo(i).times;
        else
            idx = find([tracks.id] == trackInfo(i).id, 1);
            timeStamps = tracks(idx).frames;
        end
        % Compute relative time (time elapsed since the first detection)
        relativeTime = timeStamps - timeStamps(1);
        % Compute relative y displacement (y position relative to the first detection)
        relativeY = coords(:,2) - coords(1,2);
        
        plot(relativeTime, relativeY, '-o', 'Color', colors(i,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('Dimple index: %d', trackInfo(i).id));
    end
end

xlabel('Lifetime');
ylabel('Relative displacement');
%legend('show', 'Location','northwest');
grid on; hold off;

