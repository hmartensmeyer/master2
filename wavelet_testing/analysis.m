% Import functions
addpath '../utils/'

% import dataset
load '..'\data\filtered_gray_5000t_indices.mat
data = grayscaleVideo_short;

%% Wavelet analysis

[original_flow, wavelet_coefficients_full, filtered_all_structures, filtered_dimples] = wavelet_func(data, 300, 15, 7, 0.2, 0.85, 0.6, 0.6);

%%

for t = 1:num_timesteps
    imagesc(filtered_dimples(:, :, t));
    colormap 'gray';
    title('Timestep: ', t)
    pause(0.1);
end

%% track dimples

[centroid_positions, structure_labels, structure_lifetimes, num_structures, num_timesteps] = dimpletracker(filtered_dimples, 20);

%% Visualization of tracking over time
figure;
for t = 1:num_timesteps
    % Display the filtered structure for the current timestep
    subplot(1, 1, 1);
    imagesc(filtered_dimples(:, :, t));
    colormap('gray');
    hold on;

    % Plot centroids with unique colors for tracked structures
    if ~isempty(centroid_positions{t})
        for i = 1:size(centroid_positions{t}, 1)
            label = structure_labels{t}(i);
            color = lines(max(cellfun(@max, structure_labels)));  % Generate unique colors
            scatter(centroid_positions{t}(i, 1), centroid_positions{t}(i, 2), ...
                20, color(label, :), 'filled');  % Smaller markers
        end
    end

    title(sprintf('Timestep: %d', t));
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold off;
    pause(0.1);  % Pause for visualization
end

%% Display the list of structures and their lifetimes
disp('Structure Lifetimes:');
for i = 1:num_structures
    fprintf('Structure %d: %d timesteps\n', i, structure_lifetimes(i));
end

%% Filter structures based on lifetime and visualize the full timeseries

lifetime_threshold = 6;  % Minimum lifetime for structures to be displayed

% Identify structures that meet the lifetime criterion
valid_structures = find(structure_lifetimes >= lifetime_threshold);

%% Initialize figure for visualization
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
    title('Filtered dimples at timestep: ', t);
    xlabel('X');
    ylabel('Y');
    hold on;

    % Get centroids and labels for the current frame
    if ~isempty(centroid_positions{t}) && ~isempty(structure_labels{t})
        for i = 1:size(centroid_positions{t}, 1)
            label = structure_labels{t}(i);
            if ismember(label, valid_structures)
                % Plot a small dot for the valid structure
                structure_idx = find(valid_structures == label, 1);
                scatter(centroid_positions{t}(i, 1), centroid_positions{t}(i, 2), ...
                    200, colors(structure_idx, :));
            end
        end
    end

    pause(0.1);  % Pause for visualization
    hold off;
end

%% Visualize a single structure and its active timesteps
structure_to_show = 406;  % Specify the structure label to visualize

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
                title(sprintf('Structure %d at Timestep %d', structure_to_show, t));
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
