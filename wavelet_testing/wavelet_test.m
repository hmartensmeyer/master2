%% Kun development. Må gjøre om denne til en funksjon tenker jeg.

data = load ('..\data\filtered_gray_5000t_indices.mat');
%%
data_frame = data.filteredFramesGray;
times = data.filteredTimeindeces;

%%
eta = data_frame;

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

%%

t_index = 45;
snapshot = eta(:, :, t_index);

% Perform 2D continuous wavelet transform with the Mexican hat wavelet
scales = 1:20;  % Adjust scale range based on feature size
cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);

% Extract wavelet coefficients at a specific scale
selected_scale = 15;  % Example scale index
wavelet_coefficients = cwt_result.cfs(:,:,selected_scale);

% Define the threshold
W_thr = 90;

% Create a mask for regions where W > W_thr
mask = wavelet_coefficients > W_thr;

% Apply the mask to the wavelet coefficients
filtered_coefficients = wavelet_coefficients .* mask;

% Label connected regions in the binary mask
connected_components = bwconncomp(mask);

% Measure properties of connected regions
region_props = regionprops(connected_components, 'Eccentricity','Area', 'Solidity');

% Create a new mask for regions with eccentricity < 0.85 or circularity >
% 0.85
eccentricity_threshold = 0.85;
solidity_threshold = 0.6;
eccentric_regions = ismember(labelmatrix(connected_components), ...
    find([region_props.Eccentricity] < eccentricity_threshold  & [region_props.Solidity] > solidity_threshold));

% Apply the new mask to the wavelet coefficients
%filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
filtered_by_eccentricity = 1 - eccentric_regions; %binary version

% Plot original surface elevation
hfig = figure('Name', 'Wavelet Analysis', Colormap=gray);

t = tiledlayout(2, 2, "TileSpacing","compact","Padding","compact");

% Original surface elevation
%subplot(2, 2, 1);
nexttile;
imagesc(snapshot);
title(sprintf('Grayscale ceiling at t = %d', times(t_index)));
xlabel('X');
ylabel('Y');
%set(gca, 'XTickLabel', [], 'YTickLabel', []);
colorbar;

% Wavelet coefficients
%subplot(2, 2, 2);
nexttile;
imagesc(wavelet_coefficients);
title(sprintf('Wavelet coefficients (Scale %d)', scales(selected_scale)));
xlabel('X');
ylabel('Y');
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colorbar;

% Visualization of filtered coefficients
%subplot(2, 2, 3);
nexttile;
imagesc(filtered_coefficients);
title('Filtered wavelet coefficients ($W > W_{thr}$)');
xlabel('X');
ylabel('Y');
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colorbar;

% Visualization of the filtered coefficients
%subplot(2, 2, 4);
nexttile;
imagesc(filtered_by_eccentricity);
title('Filtered (Eccentricity $<$ 0.85)');
xlabel('X');
ylabel('Y');
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colorbar;


% Calculate coverage after W-thresholding
nonzero_pixels = nnz(filtered_coefficients);  % Number of nonzero pixels
total_pixels = numel(filtered_coefficients); % Total number of pixels
coverage = nonzero_pixels / total_pixels;    % Fraction of nonzero pixels

% Display coverage
fprintf('Coverage after W-thresholding: %.2f%%\n', coverage * 100);

% % Display eccentricity values of filtered regions
% fprintf('Eccentricity of filtered regions:\n');
% for i = 1:length(region_props)
%     fprintf('Region %d: Eccentricity = %.4f\n', i, region_props(i).Eccentricity);
% end

% Plotting stuff

% Set figure properties
set(findall(hfig, '-property', 'FontSize'), 'FontSize', 14); % Adjust fontsize
set(findall(hfig, '-property', 'Box'), 'Box', 'on'); % Remove box if needed
set(findall(hfig, '-property', 'Interpreter'), 'Interpreter', 'latex');
set(findall(hfig, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
legend('Location', 'southeast', 'FontSize', 14, 'FontWeight', 'bold', 'Box','off');

% Configure figure size and save options
picturewidth = 20; % in centimeters
hw_ratio = 0.6; % height-width ratio
set(hfig, 'Units', 'centimeters', 'Position', [3 3 picturewidth hw_ratio * picturewidth]);
pos = get(hfig, 'Position');
set(hfig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'centimeters', 'PaperSize', [pos(3), pos(4)]);

% Uncomment the appropriate line below to save
%print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-r300'); % Adjust resolution if needed

% %% Try to find out which region is which
% 
% % Get the labeled matrix of connected components
% labeled_regions = labelmatrix(connected_components);
% 
% % Display region information along with eccentricity
% fprintf('Region Eccentricity Mapping:\n');
% for i = 1:length(region_props)
%     fprintf('Region %d: Eccentricity = %.4f, Area = %d, Solidity = %d\n', i, region_props(i).Eccentricity, region_props(i).Area, region_props(i).Solidity);
% end
% 
% % Example: visualize one specific region by its index
% region_index = 48; % Change this to the desired region index
% specific_region_mask = (labeled_regions == region_index);
% 
% % Plot the specific region
% figure('Name', sprintf('Region %d Visualization', region_index), Colormap=gray);
% imagesc(specific_region_mask);
% title(sprintf('Region %d (Eccentricity = %.4f)', region_index, region_props(region_index).Eccentricity));
% xlabel('X Coordinate');
% ylabel('Y Coordinate');
% colorbar;


%% Test for time series

% Parameters
timesteps = 1:99;  % Define the range of timesteps (100 timesteps)
scales = 1:15;  % Adjust scale range based on feature size
selected_scale = 15;  % Scale index to use
W_thr = 90;  % Threshold for wavelet coefficients
eccentricity_threshold = 0.85;  % Threshold for eccentricity
circularity_threshold = 0.8;
solidity_threshold = 0.6;

% Preallocate array for filtered snapshots
[x_dim, y_dim] = size(eta(:, :, 1));  % Dimensions of each snapshot
%original_flow = zeros(x_dim, y_dim, length(timesteps));
%wavelet_coefficients_full = zeros(x_dim, y_dim, length(timesteps));
filtered_all_structures = zeros(x_dim, y_dim, length(timesteps));
filtered_dimples = zeros(x_dim, y_dim, length(timesteps));  % 3D array to store results

% Loop through each timestep
for t_index = 1:length(timesteps)
    disp(t_index)
    t = timesteps(t_index);
    snapshot = eta(:, :, t);

    % Perform 2D continuous wavelet transform with the Mexican hat wavelet
    cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);

    % Extract wavelet coefficients at the selected scale
    wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);

    % Create a mask for regions where W < W_thr
    mask = wavelet_coefficients > W_thr;

    % Apply the mask to the wavelet coefficients
    filtered_coefficients = wavelet_coefficients .* mask;

    % Label connected regions in the binary mask
    connected_components = bwconncomp(mask);

    % Measure properties of connected regions
    region_props = regionprops(connected_components, 'Eccentricity', 'Solidity');

    % Create a new mask for regions with eccentricity < threshold
    eccentric_regions = ismember(labelmatrix(connected_components), ...
        find([region_props.Eccentricity] < eccentricity_threshold  & [region_props.Solidity] > solidity_threshold));

    % Apply the mask to the wavelet coefficients
    filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
    %filtered_by_eccentricity = 1 - eccentric_regions; %binary version

    % Save the filtered snapshot
    %original_flow(:, :, t_index) = snapshot;
    %wavelet_coefficients_full(:, :, t_index) = wavelet_coefficients;
    filtered_all_structures(:, :, t_index) = filtered_coefficients;
    filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
end

% Save the filtered snapshots to a MAT file
%save('filtered_snapshots.mat', 'filtered_snapshots', '-v7.3');

%% Enhanced Tracking with Time-Dependent Y-Shift and Widened Search Radius
% % (Make sure that the variable "eta" is defined as a 3D array of snapshots
% %  and that the vector "times" contains the corresponding time for each snapshot.)
% 
% % Parameters for wavelet filtering
% scales = 1:15;              % Adjust scale range based on feature size
% selected_scale = 15;        % Scale index to use
% W_thr = 90;                 % Threshold for wavelet coefficients
% eccentricity_threshold = 0.85;
% solidity_threshold = 0.6;
% 
% % Tracking parameters
% baseYShift = 35;  % Base offset in the y-direction per time unit.
%                   % For consecutive times (dt==1) search is centered at y+35.
%                   % For dt>1 the predicted y-shift (and search radius) is dt*35.
% 
% % Assume "times" is provided (e.g., times = [56, 59, 62, ...];)
% % The number of snapshots is assumed to be length(times).
% 
% % Preallocate arrays for filtered snapshots (if you wish to save them)
% [x_dim, y_dim] = size(eta(:, :, 1));
% filtered_all_structures = zeros(x_dim, y_dim, length(times));
% filtered_dimples = zeros(x_dim, y_dim, length(times));
% 
% % Initialize tracking structure
% % Each track will have an id, a list of centroids, and a list of time stamps.
% tracks = struct('id', {}, 'centroids', {}, 'frames', {});
% nextTrackId = 1;
% 
% % Loop over each snapshot using the provided "times" array
% for t_index = 1:length(times)
%     currentTime = times(t_index);
% 
%     % IMPORTANT: Adjust the indexing for your snapshots.
%     % If "eta" is organized so that the third dimension corresponds to the 
%     % snapshot order (and matches "times"), then use:
%     snapshot = eta(:, :, t_index);
%     % Alternatively, if the time value is used as an index (e.g. snapshot 56, 59, ...)
%     % then use:
%     %   snapshot = eta(:, :, currentTime);
% 
%     % === Wavelet Filtering (same as your original code) === %
%     % Compute the 2D continuous wavelet transform using the Mexican hat wavelet
%     cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);
%     wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);
% 
%     % Threshold coefficients (keep only values above W_thr)
%     mask = wavelet_coefficients > W_thr;
%     filtered_coefficients = wavelet_coefficients .* mask;
% 
%     % Label connected regions and compute their properties
%     connected_components = bwconncomp(mask);
%     region_props = regionprops(connected_components, 'Eccentricity', 'Solidity', 'Centroid');
% 
%     % Filter regions by eccentricity and solidity
%     validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
%                       [region_props.Solidity] > solidity_threshold);
%     eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
%     filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
% 
%     % Save the filtered snapshots (if desired)
%     filtered_all_structures(:, :, t_index) = filtered_coefficients;
%     filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
% 
%     % === Extract Centroids from Valid Regions === %
%     if isempty(validIdx)
%         centroids = [];
%     else
%         % Each row is [x, y]
%         centroids = cat(1, region_props(validIdx).Centroid);
%     end
%     numDetections = size(centroids, 1);
% 
%     % --- Tracking ---
%     % For every detected centroid, we want to see if it matches an existing track.
%     % For each existing track, the predicted position is the last recorded
%     % centroid shifted in the y-direction by (currentTime - lastTime)*baseYShift.
%     % The allowed search radius is also (currentTime - lastTime)*baseYShift.
% 
%     numTracks = length(tracks);
%     % Create a cost matrix: rows correspond to detections, columns to existing tracks.
%     costMatrix = Inf(numDetections, numTracks);
%     for i = 1:numDetections
%         for j = 1:numTracks
%             lastTime = tracks(j).frames(end);
%             dt = currentTime - lastTime;  % Time gap (could be > 1)
%             % Predicted position: same x, but shifted in y by dt*baseYShift
%             predicted = [tracks(j).centroids(end, 1), tracks(j).centroids(end, 2) + dt * baseYShift];
%             allowedDistance = dt * baseYShift;
%             d = norm(centroids(i, :) - predicted);
%             if d <= allowedDistance
%                 costMatrix(i, j) = d;
%             else
%                 costMatrix(i, j) = Inf;
%             end
%         end
%     end
% 
%     % Use a greedy algorithm to assign detections to tracks.
%     % (Each detection and track may be used only once.)
%     detectionTrackIDs = zeros(numDetections, 1);  % To record which track (id) each detection belongs to
%     assignments = [];  % Each row will be: [detectionIndex, trackIndex, cost]
%     if ~isempty(costMatrix)
%         while true
%             [minVal, idx] = min(costMatrix(:));
%             if isinf(minVal)
%                 break;  % No remaining valid (within allowed distance) assignment
%             end
%             [detIdx, trackIdx] = ind2sub(size(costMatrix), idx);
%             assignments = [assignments; detIdx, trackIdx, minVal]; %#ok<AGROW>
%             detectionTrackIDs(detIdx) = tracks(trackIdx).id;  % record the assigned track id
%             costMatrix(detIdx, :) = Inf;  % Remove this detection from further assignment
%             costMatrix(:, trackIdx) = Inf;  % Remove this track from further assignment
%         end
%     end
% 
%     % Update the tracks that were assigned with the new detection.
%     if ~isempty(assignments)
%         for k = 1:size(assignments, 1)
%             detIdx = assignments(k, 1);
%             trackIdx = assignments(k, 2);
%             tracks(trackIdx).centroids(end+1, :) = centroids(detIdx, :);
%             tracks(trackIdx).frames(end+1) = currentTime;
%         end
%     end
% 
%     % For any detections that were not assigned to an existing track,
%     % start a new track.
%     for i = 1:numDetections
%         if detectionTrackIDs(i) == 0
%             tracks(nextTrackId).id = nextTrackId;
%             tracks(nextTrackId).centroids = centroids(i, :);
%             tracks(nextTrackId).frames = currentTime;
%             detectionTrackIDs(i) = nextTrackId;
%             nextTrackId = nextTrackId + 1;
%         end
%     end
% 
%     % === Optional Visualization === %
%     figure(1); clf;
%     imagesc(snapshot); colormap gray; hold on;
%     if ~isempty(centroids)
%         plot(centroids(:,1), centroids(:,2), 'r*', 'MarkerSize', 8);
%         for i = 1:numDetections
%             % Label the detection with its associated track ID
%             text(centroids(i,1)+2, centroids(i,2)+2, num2str(detectionTrackIDs(i)), ...
%                  'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
%         end
%     end
%     title(['Time = ' num2str(currentTime)]);
%     drawnow;
% 
% end
% 
% % After the loop, the structure "tracks" holds the full trajectories of all features.
% % You can now analyze or further visualize the tracked structures.


%% Display the vortices over time

for t = 1:200
    imagesc(filtered_dimples(:, :, t));
    colormap 'gray';
    title('Timestep: ', times(timesteps(t)))
    pause(0.2);
end

%% Neste på programmet

% Forkaste alle strukturer som er mindre enn 5 tidssteg lange?
% Tracke virvlene, og kunne koordinatfeste dem?
% Hvordan kan jeg koble dette til hastighetsfeltet/virvlingsfeltet under

%% ALTERNATIVE TRACKER

% % Parameters
% timesteps = 1:99;              % Define the range of timesteps
% scales = 1:15;                 % Adjust scale range based on feature size
% selected_scale = 15;           % Scale index to use
% W_thr = 90;                    % Threshold for wavelet coefficients
% eccentricity_threshold = 0.85; % Threshold for eccentricity
% solidity_threshold = 0.6;      % Threshold for solidity
% maxTrackingDistance = 10;      % Maximum allowed distance for linking features (in pixels)
% 
% % Preallocate arrays for filtered snapshots (if needed for further processing)
% [x_dim, y_dim] = size(eta(:, :, 1));  % Dimensions of each snapshot
% filtered_all_structures = zeros(x_dim, y_dim, length(timesteps));
% filtered_dimples = zeros(x_dim, y_dim, length(timesteps));
% 
% % Initialize tracking structure
% % Each track will have an ID, list of centroids, and corresponding frame numbers.
% tracks = struct('id', {}, 'centroids', {}, 'frames', {});
% nextTrackId = 1;
% 
% % Loop through each timestep
% for t_index = 1:length(timesteps)
%     disp(['Processing timestep: ', num2str(timesteps(t_index))])
%     t = timesteps(t_index);
%     snapshot = eta(:, :, t);
% 
%     % === Wavelet Analysis & Filtering === %
%     % Compute the 2D continuous wavelet transform using the Mexican hat wavelet
%     cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);
%     wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);
% 
%     % Create a mask for regions where coefficients exceed the threshold
%     mask = wavelet_coefficients > W_thr;
% 
%     % Apply the mask (keep only coefficients above threshold)
%     filtered_coefficients = wavelet_coefficients .* mask;
% 
%     % Label connected regions in the binary mask
%     connected_components = bwconncomp(mask);
% 
%     % Compute properties for connected regions
%     region_props = regionprops(connected_components, 'Eccentricity', 'Solidity', 'Centroid');
% 
%     % Select regions with eccentricity below and solidity above the thresholds
%     validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
%                       [region_props.Solidity] > solidity_threshold);
%     eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
% 
%     % Apply the eccentricity-based mask to the wavelet coefficients
%     filtered_by_eccentricity = wavelet_coefficients .* eccentric_regions;
% 
%     % Save the filtered snapshots (if needed for later visualization or processing)
%     filtered_all_structures(:, :, t_index) = filtered_coefficients;
%     filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
% 
%     % === Feature Tracking === %
%     % Extract centroids of the valid regions
%     valid_props = region_props(validIdx);
%     if isempty(valid_props)
%         centroids = [];
%     else
%         centroids = cat(1, valid_props.Centroid);  % Each row is [x, y]
%     end
% 
%     % Array to hold the track ID assigned to each detected centroid in this frame
%     currentAssignments = zeros(size(centroids, 1), 1);
% 
%     if t_index == 1
%         % For the first timestep, create a new track for every detected feature.
%         for i = 1:size(centroids, 1)
%             tracks(nextTrackId).id = nextTrackId;
%             tracks(nextTrackId).centroids = centroids(i, :);
%             tracks(nextTrackId).frames = t;
%             currentAssignments(i) = nextTrackId;
%             nextTrackId = nextTrackId + 1;
%         end
%     else
%         % For subsequent timesteps, try to match each current centroid with tracks from the previous frame.
%         % First, gather the centroids from tracks that were updated in the previous timestep.
%         prevCentroids = [];
%         trackIndices = [];
%         for j = 1:length(tracks)
%             % Check if the last frame of the track is the previous timestep.
%             if tracks(j).frames(end) == timesteps(t_index-1)
%                 prevCentroids = [prevCentroids; tracks(j).centroids(end, :)];
%                 trackIndices = [trackIndices; j];
%             end
%         end
% 
%         if isempty(prevCentroids)
%             % If no track was updated in the previous frame, treat all current detections as new tracks.
%             for i = 1:size(centroids, 1)
%                 tracks(nextTrackId).id = nextTrackId;
%                 tracks(nextTrackId).centroids = centroids(i, :);
%                 tracks(nextTrackId).frames = t;
%                 currentAssignments(i) = nextTrackId;
%                 nextTrackId = nextTrackId + 1;
%             end
%         else
%             % For each detected centroid in the current frame, find the closest previous centroid.
%             assignedPrev = false(size(prevCentroids, 1), 1); % To ensure a one-to-one match
%             for i = 1:size(centroids, 1)
%                 % Compute Euclidean distances to all previous centroids.
%                 dists = sqrt(sum((prevCentroids - centroids(i, :)).^2, 2));
%                 [minDist, minIdx] = min(dists);
%                 if minDist <= maxTrackingDistance && ~assignedPrev(minIdx)
%                     % If a close match is found, update that track.
%                     trackID = tracks(trackIndices(minIdx)).id;
%                     tracks(trackIndices(minIdx)).centroids(end+1, :) = centroids(i, :);
%                     tracks(trackIndices(minIdx)).frames(end+1) = t;
%                     currentAssignments(i) = trackID;
%                     assignedPrev(minIdx) = true;
%                 else
%                     % If no suitable match, start a new track.
%                     tracks(nextTrackId).id = nextTrackId;
%                     tracks(nextTrackId).centroids = centroids(i, :);
%                     tracks(nextTrackId).frames = t;
%                     currentAssignments(i) = nextTrackId;
%                     nextTrackId = nextTrackId + 1;
%                 end
%             end
%         end
%     end
% 
%     % === Optional Visualization === %
%     % Overlay the detected centroids and their track IDs on the original snapshot.
%     figure(1); clf;
%     imagesc(snapshot); colormap gray; hold on;
%     if ~isempty(centroids)
%         plot(centroids(:,1), centroids(:,2), 'r*', 'MarkerSize', 8);
%         for i = 1:size(centroids,1)
%             text(centroids(i,1)+2, centroids(i,2)+2, num2str(currentAssignments(i)), ...
%                 'Color','y', 'FontSize',12, 'FontWeight','bold');
%         end
%     end
%     title(['Timestep ', num2str(t)]);
%     drawnow;
% 
% end
% 
% % After the loop, the 'tracks' structure holds the trajectories for all tracked features.
% % You can now analyze or visualize the tracks further.


%% Track filtered regions using line-based distance approach
% Parameters for tracking
num_timesteps = size(filtered_dimples, 3);
centroid_positions = cell(num_timesteps, 1);  % Store centroids for each timestep
structure_labels = cell(num_timesteps, 1);   % Store region labels for tracking
max_distance = 100;  % Maximum distance to associate centroids between frames

% Loop through each timestep to extract region centroids and labels
for t = 1:num_timesteps
    disp(t)
    % Get the binary mask for the current timestep
    binary_mask = filtered_dimples(:, :, t) > 0;

    % Label connected components in the binary mask
    connected_components = bwconncomp(binary_mask);

    % Measure region properties (centroids)
    region_props = regionprops(connected_components, 'Centroid');

    % Store centroids and labels for the current timestep
    if ~isempty(region_props)
        centroids = cat(1, region_props.Centroid);
        centroid_positions{t} = centroids;

        % Assign labels to structures based on line-based distance to previous frame
        if t > 1 && ~isempty(centroid_positions{t-1})
            prev_centroids = centroid_positions{t-1};
            prev_labels = structure_labels{t-1};
            structure_labels{t} = zeros(size(centroids, 1), 1);

            % Create a distance matrix between current and previous centroids
            distances = pdist2(centroids, prev_centroids);

            % Track matched previous centroids
            matched_prev = false(size(prev_centroids, 1), 1);  
            next_new_label = max(prev_labels, [], 'omitnan') + 1;  % Start new labels from the max

            % Match centroids using a greedy algorithm
            for i = 1:size(centroids, 1)
                % Find the closest previous centroid
                [min_distance, closest_idx] = min(distances(i, :));

                if min_distance < max_distance && ~matched_prev(closest_idx) && (centroids(i,2) > prev_centroids(closest_idx,2))
                    % Assign the same label as the closest previous centroid
                    structure_labels{t}(i) = prev_labels(closest_idx);
                    matched_prev(closest_idx) = true;  % Mark this previous centroid as matched
                else
                    % Assign a unique new label for unmatched centroids
                    structure_labels{t}(i) = next_new_label;
                    next_new_label = next_new_label + 1;  % Increment label for next new structure
                end
            end
        else
            % Assign new unique labels for the first frame
            structure_labels{t} = (1:size(centroids, 1))';
        end
    else
        centroid_positions{t} = [];
        structure_labels{t} = [];
    end
end


%% Visualization of tracking over time
figure;
for t = 1:200
    % Display the filtered structure for the current timestep
    subplot(1, 1, 1);
    imagesc(filtered_dimples(:, :, t));
    colormap('gray');
    hold on;

    % Plot centroids with unique colors for tracked structures
    if ~isempty(centroid_positions{t})
        for i = 1:size(centroid_positions{t}, 1)
            label = structure_labels{t}(i);
            %color = lines(max(cellfun(@max, structure_labels, 'UniformOutput', false)));  % Generate unique colors
            color = lines(max(structure_labels{t}));  % Generate unique colors if structure labels is numeric
            scatter(centroid_positions{t}(i, 1), centroid_positions{t}(i, 2), ...
                200, color(label, :));  % Smaller markers
        end
    end

    title(sprintf('Timestep: %d', times(timesteps(t))));
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold off;
    pause(0.5);  % Pause for visualization
end


%% ALTERNATIVE PARTICLE TRACKING

% %% Particle Tracking Algorithm for Structure Tracking (Fixed)
% % Parameters
% num_timesteps = size(filtered_dimples, 3);
% particle_positions = cell(num_timesteps, 1);  % Store tracked positions
% particle_labels = cell(num_timesteps, 1);  % Store particle IDs
% max_search_radius = 50;  % Max allowed movement per frame
% prediction_weight = 0.8;  % Weight for velocity-based prediction
% 
% % Initialize particles from the first frame
% binary_mask = filtered_dimples(:, :, 1) > 0;
% connected_components = bwconncomp(binary_mask);
% region_props = regionprops(connected_components, 'Centroid');
% 
% if ~isempty(region_props)
%     particle_positions{1} = cat(1, region_props.Centroid);
%     particle_labels{1} = (1:size(particle_positions{1}, 1))';
% else
%     particle_positions{1} = [];
%     particle_labels{1} = [];
% end
% 
% % Track particles through time
% for t = 2:num_timesteps
%     disp(['Tracking timestep: ', num2str(t)])
% 
%     % Get current frame's structures
%     binary_mask = filtered_dimples(:, :, t) > 0;
%     connected_components = bwconncomp(binary_mask);
%     region_props = regionprops(connected_components, 'Centroid');
% 
%     if ~isempty(region_props)
%         current_positions = cat(1, region_props.Centroid);
%     else
%         current_positions = [];
%     end
% 
%     % Get previous frame's particle positions and labels
%     previous_positions = particle_positions{t-1};
%     previous_labels = particle_labels{t-1};
% 
%     % Compute past displacement for only matched structures
%     if t > 2 && ~isempty(particle_positions{t-2}) && ~isempty(previous_positions)
%         % Find matches between previous and t-2 positions
%         distance_matrix_past = pdist2(previous_positions, particle_positions{t-2});
%         [~, closest_past_idx] = min(distance_matrix_past, [], 2);
% 
%         % Ensure index validity
%         valid_matches = closest_past_idx <= size(particle_positions{t-2}, 1);
%         past_displacement = zeros(size(previous_positions));
%         past_displacement(valid_matches, :) = ...
%             previous_positions(valid_matches, :) - particle_positions{t-2}(closest_past_idx(valid_matches), :);
%     else
%         past_displacement = zeros(size(previous_positions));
%     end
% 
%     % Predict next positions using past displacement
%     predicted_positions = previous_positions + prediction_weight * past_displacement;
% 
%     % Match new positions to predicted positions (nearest neighbor search)
%     particle_labels{t} = zeros(size(current_positions, 1), 1);
%     if ~isempty(previous_positions) && ~isempty(current_positions)
%         distance_matrix = pdist2(current_positions, predicted_positions);
%         matched_previous = false(size(previous_positions, 1), 1);
% 
%         for i = 1:size(current_positions, 1)
%             [min_distance, closest_idx] = min(distance_matrix(i, :));
% 
%             if min_distance < max_search_radius && ~matched_previous(closest_idx)
%                 % Assign same label if within range
%                 particle_labels{t}(i) = previous_labels(closest_idx);
%                 matched_previous(closest_idx) = true;
%             else
%                 % Assign a new label to unmatched particles
%                 particle_labels{t}(i) = max(previous_labels) + 1;
%             end
%         end
%     else
%         % If no previous particles, assign new labels
%         particle_labels{t} = (1:size(current_positions, 1))';
%     end
% 
%     % Store updated particle positions
%     particle_positions{t} = current_positions;
% end
% 
% %% Visualization of Particle Tracking Over Time
% figure;
% num_timesteps = length(particle_positions);
% 
% for t = 1:num_timesteps
%     % Display the filtered structure for the current timestep
%     clf; % Clear figure for smooth animation
%     imagesc(filtered_dimples(:, :, t));
%     colormap('gray');
%     hold on;
% 
%     % Plot tracked particles with unique colors
%     if ~isempty(particle_positions{t})
%         num_particles = max(cellfun(@max, particle_labels(~cellfun(@isempty, particle_labels))));
%         color_map = lines(num_particles);  % Generate distinct colors
% 
%         for i = 1:size(particle_positions{t}, 1)
%             label = particle_labels{t}(i);
% 
%             % Ensure label is within range to avoid indexing errors
%             if label > 0 && label <= size(color_map, 1)
%                 color = color_map(label, :);
%                 scatter(particle_positions{t}(i, 1), particle_positions{t}(i, 2), ...
%                     100, color, 'filled');  % Adjust marker size as needed
%             end
%         end
%     end
% 
%     title(sprintf('Timestep: %d', t));
%     xlabel('X Coordinate');
%     ylabel('Y Coordinate');
%     axis equal;
%     drawnow;
%     pause(0.5);  % Pause for visualization
% end


%% Visualize structures as small dots at each timestep
% Initialize figure
figure('Name', 'Structure Tracking with Dots', 'Position', [100, 100, 800, 600]);
hold on;
axis equal;
xlim([1, size(filtered_dimples, 2)]);
ylim([1, size(filtered_dimples, 1)]);
title('Tracked Structures');
xlabel('X Coordinate');
ylabel('Y Coordinate');
colormap('gray');

% Assign unique colors for structures
%num_structures = max(cellfun(@max, structure_labels));
num_structures = max(structure_labels{end}(:)); %For a numeric structure_labels
colors = lines(num_structures);

% Loop through timesteps to plot dots for structures
for t = 40:100
    disp(['Processing timestep: ', num2str(times(t))]);  % Debug output

    % Get centroids and labels for the current frame
    if ~isempty(centroid_positions{t}) && ~isempty(structure_labels{t})
        for i = 1:size(centroid_positions{t}, 1)
            label = structure_labels{t}(i);
            if label > 0
                % Plot a small dot at the centroid
                scatter(centroid_positions{t}(i, 1), centroid_positions{t}(i, 2), ...
                    10, colors(label, :), 'filled');  % Small dots
            end
        end
    end
    pause(0.02);  % Pause for visualization
end

hold off;

%% Calculate lifetime of each structure
num_structures = max(cellfun(@max, structure_labels));  % Total number of structures
structure_lifetimes = zeros(num_structures, 1);  % Preallocate lifetimes array

% Loop through all timesteps to count occurrences of each structure
for t = 1:num_timesteps
    if ~isempty(structure_labels{t})
        unique_labels = unique(structure_labels{t});  % Get unique structure labels
        for label = unique_labels'
            if label > 0  % Ignore unmatched structures (label = 0)
                structure_lifetimes(label) = structure_lifetimes(label) + 1;
            end
        end
    end
end

%% Display the list of structures and their lifetimes
disp('Structure Lifetimes:');
for i = 1:num_structures
    if structure_lifetimes(i) > 10
        fprintf('Structure %d: %d timesteps\n', i, structure_lifetimes(i));
    end
end

%% Visualize a single structure and its active timesteps
structure_to_show = 476;  % Specify the structure label to visualize

% Initialize figure for visualization
figure('Name', sprintf('Structure %d Visualization', structure_to_show), 'Position', [100, 100, 800, 600]);

% Track the timesteps where the structure is present
active_timesteps = [];

for t = 1:num_timesteps
    if ~isempty(centroid_positions{t}) && ~isempty(structure_labels{t})
        % Check if the structure exists in this frame
        if ismember(structure_to_show, structure_labels{t})
            % Add to active timesteps
            active_timesteps = [active_timesteps, t];

            % Find the connected component for this structure
            binary_mask = filtered_dimples(:, :, t) > 0;
            connected_components = bwconncomp(binary_mask);
            structure_idx = find(structure_labels{t} == structure_to_show, 1);

            if ~isempty(structure_idx)
                % Create a mask for the structure
                structure_mask = ismember(labelmatrix(connected_components), structure_idx);

                % Visualize the structure
                imagesc(structure_mask);
                colormap('gray');
                title(sprintf('Structure %d at Timestep %d', structure_to_show, times(timesteps(t))));
                xlabel('X Coordinate');
                ylabel('Y Coordinate');
                colorbar;
                pause(0.5);  % Pause for visualization
            end
        end
    end
end

% Display the active timesteps for the structure
disp(['Structure ', num2str(structure_to_show), ' is active during timesteps:']);
disp(active_timesteps);


%% Filter structures based on lifetime and visualize the full timeseries

lifetime_threshold = 6;  % Minimum lifetime for structures to be displayed

% Identify structures that meet the lifetime criterion
valid_structures = find(structure_lifetimes >= lifetime_threshold);

% Initialize figure for visualization
figure('Name', 'Filtered Structures Over Time', 'Position', [100, 100, 800, 600]);
hold on;
axis equal;
xlim([1, size(filtered_dimples, 2)]);
ylim([1, size(filtered_dimples, 1)]);
title('Filtered Structures Over Time');
xlabel('X Coordinate');
ylabel('Y Coordinate');
colormap('gray');

% Assign unique colors for valid structures
colors = lines(length(valid_structures));

% Loop through timesteps to plot valid structures
for t = 1:num_timesteps
    % Display the snapshot as background
    imagesc(filtered_dimples(:, :, t));
    title('Timestep: ', times(timesteps(t)))
    hold on;

    % Get centroids and labels for the current frame
    if ~isempty(centroid_positions{t}) && ~isempty(structure_labels{t})
        for i = 1:size(centroid_positions{t}, 1)
            label = structure_labels{t}(i);
            if ismember(label, valid_structures)
                % Plot a small dot for the valid structure
                structure_idx = find(valid_structures == label, 1);
                scatter(centroid_positions{t}(i, 1), centroid_positions{t}(i, 2), ...
                    20, colors(structure_idx, :), 'filled');
            end
        end
    end

    pause(0.5);  % Pause for visualization
    hold off;
end

%% Save the filtered video with associated timesteps for interpolation prep

selectedTimes = times(400:499);
save('..\data\filtered_dimples_400_499.mat', 'filtered_dimples', 'selectedTimes', '-v7.3');
