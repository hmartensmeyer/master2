%% Kun development. Må gjøre om denne til en funksjon tenker jeg.

data = load ('..\data\filtered_gray_5000t_indices.mat');

%%
data_frame = data.filteredFramesGray;
times = data.filteredTimeindeces;

%%
disp(times(1075:1100))

%% Short snippet to get data on correct form
[height, width] = size(data_frame{1});

% Preallocate a 3D matrix for the frames
numFrames = length(data_frame);
eta = zeros(height, width, numFrames, 'uint8'); % Use 'uint8' for grayscale images

% Populate the 3D matrix
for t = 1:numFrames
    eta(:, :, t) = data_frame{t};
end

disp('Converted filteredFramesGray to 3D matrix.');

%% MEAN SUBTRACTION TO REMOVE THE BLACK CEILING PANELS
% Compute mean frame across time. The output is a double array.
mean_frame = mean(eta, 3);  % 1080x1920 (double)

% Convert eta to double, then subtract the mean frame.
data_frame_demeaned = double(eta) - mean_frame;

%% Display the first frame before and after mean subtraction for verification
figure;  % Create a new figure window
for t = 1000:1100
    % Original frame
    subplot(1,2,1);
    imagesc(eta(:,:,t)); 
    colormap gray; 
    axis image off;
    title(['Original Frame ' num2str(t)]);
    
    % Demeaned frame
    subplot(1,2,2);
    imagesc(data_frame_demeaned(:,:,t)); 
    colormap gray; 
    axis image off;
    title(['Demeaned Frame ' num2str(t)]);
    
    drawnow;  % Update the figure window immediately
    pause(0.1);  % Pause for 0.1 seconds (adjust as needed for your display speed)
end


%% Enhanced Tracking with Time-Dependent Y-Shift and Widened Search Radius
% (Make sure that the variable "eta" is defined as a 3D array of snapshots
%  and that the vector "times" contains the corresponding time for each snapshot.)

% Parameters for wavelet filtering
scales = 1:15;              % Adjust scale range based on feature size
selected_scale = 15;        % Scale index to use
W_thr = 90;                 % Threshold for wavelet coefficients
eccentricity_threshold = 0.85;
solidity_threshold = 0.6;

% Tracking parameters
baseYShift = 35;  % Base offset in the y-direction per time unit.
                  % For consecutive times (dt==1) search is centered at y+35.
                  % For dt>1 the predicted y-shift (and search radius) is dt*35.

% Assume "times" is provided (e.g., times = [56, 59, 62, ...];)
% The number of snapshots is assumed to be length(times).

% Preallocate arrays for filtered snapshots (if you wish to save them)
[x_dim, y_dim] = size(eta(:, :, 1));
filtered_all_structures = zeros(x_dim, y_dim, length(times));
filtered_dimples = zeros(x_dim, y_dim, length(times));

% Initialize tracking structure
% Each track will have an id, a list of centroids, and a list of time stamps.
tracks = struct('id', {}, 'centroids', {}, 'frames', {}, 'active', {});
nextTrackId = 1;

% Loop over each snapshot using the provided "times" array
for t_index = 900:999
    currentTime = times(t_index);
    disp(currentTime)
    
    % IMPORTANT: Adjust the indexing for your snapshots.
    % If "eta" is organized so that the third dimension corresponds to the 
    % snapshot order (and matches "times"), then use:
    snapshot = eta(:, :, t_index);
    % Alternatively, if the time value is used as an index (e.g. snapshot 56, 59, ...)
    % then use:
    %   snapshot = eta(:, :, currentTime);
    
    % === Wavelet Filtering (same as your original code) === %
    % Compute the 2D continuous wavelet transform using the Mexican hat wavelet
    cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);
    wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);
    
    % Threshold coefficients (keep only values above W_thr)
    mask = wavelet_coefficients > W_thr;
    filtered_coefficients = wavelet_coefficients .* mask;
    
    % Label connected regions and compute their properties
    connected_components = bwconncomp(mask);
    region_props = regionprops(connected_components, 'Eccentricity', 'Solidity', 'Centroid');
    
    % Filter regions by eccentricity and solidity
    validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
                      [region_props.Solidity] > solidity_threshold);
    eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
    filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
    
    % Save the filtered snapshots (if desired)
    filtered_all_structures(:, :, t_index) = filtered_coefficients;
    filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
    
    % === Extract Centroids from Valid Regions === %
    if isempty(validIdx)
        centroids = [];
    else
        % Each row is [x, y]
        centroids = cat(1, region_props(validIdx).Centroid);
    end
    numDetections = size(centroids, 1);
    
    % --- Tracking ---
    % For every detected centroid, we want to see if it matches an existing track.
    % For each existing track, the predicted position is the last recorded
    % centroid shifted in the y-direction by (currentTime - lastTime)*baseYShift.
    % The allowed search radius is also (currentTime - lastTime)*baseYShift.
    
    numTracks = length(tracks);
    % Create a cost matrix: rows correspond to detections, columns to existing tracks.
    costMatrix = Inf(numDetections, numTracks);
    for i = 1:numDetections
        for j = 1:numTracks
            if ~tracks(j).active
                costMatrix(i, j) = Inf;
                continue;
            end
            lastTime = tracks(j).frames(end);
            dt = currentTime - lastTime;  % Time gap (could be > 1)
            % Predicted position: same x, but shifted in y by dt*baseYShift
            predicted = [tracks(j).centroids(end, 1), tracks(j).centroids(end, 2) + dt * baseYShift];
            allowedDistance = dt * baseYShift;
            d = norm(centroids(i, :) - predicted);
            if d <= allowedDistance
                costMatrix(i, j) = d;
            else
                costMatrix(i, j) = Inf;
            end
        end
    end
    
    % Use a greedy algorithm to assign detections to tracks.
    % (Each detection and track may be used only once.)
    detectionTrackIDs = zeros(numDetections, 1);  % To record which track (id) each detection belongs to
    assignments = [];  % Each row will be: [detectionIndex, trackIndex, cost]
    if ~isempty(costMatrix)
        while true
            [minVal, idx] = min(costMatrix(:));
            if isinf(minVal)
                break;  % No remaining valid (within allowed distance) assignment
            end
            [detIdx, trackIdx] = ind2sub(size(costMatrix), idx);
            assignments = [assignments; detIdx, trackIdx, minVal]; %#ok<AGROW>
            detectionTrackIDs(detIdx) = tracks(trackIdx).id;  % record the assigned track id
            costMatrix(detIdx, :) = Inf;  % Remove this detection from further assignment
            costMatrix(:, trackIdx) = Inf;  % Remove this track from further assignment
        end
    end
    
    % Update the tracks that were assigned with the new detection.
    if ~isempty(assignments)
        for k = 1:size(assignments, 1)
            detIdx = assignments(k, 1);
            trackIdx = assignments(k, 2);
            tracks(trackIdx).centroids(end+1, :) = centroids(detIdx, :);
            tracks(trackIdx).frames(end+1) = currentTime;
        end
    end
    
    % For any detections that were not assigned to an existing track,
    % start a new track.
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

    % Declare lost tracks as dead (if not updated in the current frame and tracked for at least 2 time steps)
    for j = 1:length(tracks)
        if tracks(j).active && tracks(j).frames(end) < currentTime && numel(tracks(j).frames) >= 2
            tracks(j).active = false;
        end
    end

    % === Optional Visualization === %
    figure(1); clf;
    imagesc(wavelet_coefficients); colormap gray; hold on;
    if ~isempty(centroids)
        plot(centroids(:,1), centroids(:,2), 'ro', 'MarkerSize', 8);
        for i = 1:numDetections
            % Label the detection with its associated track ID
            text(centroids(i,1)+2, centroids(i,2)+2, num2str(detectionTrackIDs(i)), ...
                 'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    title(['Time = ' num2str(currentTime)]);
    drawnow;
    
end
% After the loop, the structure "tracks" holds the full trajectories of all features.
% You can now analyze or further visualize the tracked structures.

%% Display the vortices over time

for t = 1:100
    imagesc(filtered_dimples(:, :, t));
    colormap 'gray';
    title('Timestep: ', times(timesteps(t)))
    pause(0.2);
end

%% Extract and display track information
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
    if trackInfo(i).lifetime > 10
    fprintf('Track %d: Lifetime = %d frames\n', trackInfo(i).id, trackInfo(i).lifetime);
    disp(trackInfo(i).coordinates);
    end
end
