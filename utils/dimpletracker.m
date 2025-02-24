function [centroid_positions, structure_labels, structure_lifetimes, num_structures, num_timesteps] = dimpletracker(filtered_dimples, max_distance)
% DIMPLE_TRACKER - Track filtered regions using a line-based distance approach
%
%   Input:
%       filtered_dimples   - 3D binary array (x, y, t) representing filtered regions across timesteps.
%       max_distance       - Maximum distance to associate centroids between frames.
%
%   Output:
%       centroid_positions - Cell array where each cell contains centroids of regions at each timestep.
%       structure_labels   - Cell array where each cell contains labels for regions at each timestep.
%       structure_lifetimes - Array where each element represents the lifetime of a structure.
%       num_structures     - Total number of unique structures tracked across all timesteps.
%
%   Description:
%       This function tracks regions (e.g., dimples) across multiple timesteps by computing the
%       distance between region centroids in consecutive frames. Regions in the current frame
%       are associated with regions in the previous frame if the distance between their centroids
%       is below the specified threshold. It also calculates the lifetime of each structure.

% Number of timesteps
num_timesteps = size(filtered_dimples, 3);

% Preallocate outputs
centroid_positions = cell(num_timesteps, 1);  % Store centroids for each timestep
structure_labels = cell(num_timesteps, 1);   % Store region labels for tracking

% Initialize label counter
next_new_label = 1;

% Loop through each timestep to extract region centroids and labels
for t = 1:num_timesteps
    disp(['Processing timestep: ', num2str(t)])
    
    % Get the binary mask for the current timestep
    binary_mask = filtered_dimples(:, :, t) > 0;

    % Label connected components in the binary mask
    connected_components = bwconncomp(binary_mask);

    % Measure region properties (centroids)
    region_props = regionprops(connected_components, 'Centroid');

    % Store centroids and labels for the current timestep
    if ~isempty(region_props)
        centroids = cat(1, region_props.Centroid); % Extract centroids
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

            % Match centroids using a greedy algorithm
            for i = 1:size(centroids, 1)
                % Find the closest previous centroid
                [min_distance, closest_idx] = min(distances(i, :));

                if min_distance < max_distance && ~matched_prev(closest_idx)
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
            % Assign new labels for the first frame or if no previous centroids exist
            structure_labels{t} = next_new_label:(next_new_label + size(centroids, 1) - 1);
            next_new_label = next_new_label + size(centroids, 1);
        end
    else
        % No regions in the current timestep
        centroid_positions{t} = [];
        structure_labels{t} = [];
    end
end

%% Calculate lifetime of each structure
num_structures = next_new_label - 1;  % Total number of unique structures
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
end
