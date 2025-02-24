function [original_flow, wavelet_coefficients_full, filtered_all_structures, filtered_dimples] = wavelet_func(data, end_timestep, end_scale, selected_scale, ...
                                                W_thr, eccentricity_threshold, circularity_threshold, solidity_threshold)
% WAVELET_FUNC - Wavelet analysis of the desired dataset
%
%   Input:
%       data                   - The input dataset containing the flow field
%       end_timestep           - The final timestep up to which the wavelet analysis will be performed.
%       end_scale                 - The maximum scale for wavelet decomposition.
%       selected_scale         - The specific scale at which the structures will be analyzed.
%       W_thr                  - Wavelet coefficient threshold for filtering.
%       eccentricity_threshold - Threshold for filtering based on the eccentricity of structures.
%       circularity_threshold  - Threshold for filtering based on the circularity of structures.
%       solidity_threshold     - Threshold for filtering based on the solidity of structures.
%
%   Output:
%       original_flow              - The original flow field data up to the specified timestep.
%       wavelet_coefficients_full  - Full wavelet coefficients for the entire dataset.
%       filtered_all_structures    - Filtered structures that meet the defined criteria.
%       filtered_dimples           - Filtered dimples based on additional structure criteria.
%
%   Description:
%       This function performs a wavelet analysis on the input dataset, identifying and filtering
%       structures based on their geometric properties and wavelet coefficients. The analysis
%       helps in identifying flow features such as dimples or other specific structures of interest.
%
%   Example:
%       [original_flow, coeff_full, filtered_structures, dimples] = wavelet_func(ceiling_video, 5000, 10, 7, ...
%           0.2, 0.85, 0.6, 0.6);

timesteps = 1:end_timestep;  % Define the range of timesteps (100 timesteps)
scales = 1:end_scale;  % Adjust scale range based on feature size
% selected_scale = selected_scale;  % Scale index to use
% W_thr = W_thr;  % Threshold for wavelet coefficients
% eccentricity_threshold = 0.85;  % Threshold for eccentricity
% circularity_threshold = 0.8;
% solidity_threshold = 0.6;

% Preallocate array for filtered snapshots
[x_dim, y_dim] = size(data(:, :, 1));  % Dimensions of each snapshot
original_flow = zeros(x_dim, y_dim, length(timesteps));
wavelet_coefficients_full = zeros(x_dim, y_dim, length(timesteps));
filtered_all_structures = zeros(x_dim, y_dim, length(timesteps));
filtered_dimples = zeros(x_dim, y_dim, length(timesteps));  % 3D array to store results

% Loop through each timestep
for t_index = 1:length(timesteps)
    disp(t_index)
    t = timesteps(t_index);
    snapshot = data(:, :, t);

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
    original_flow(:, :, t_index) = snapshot;
    wavelet_coefficients_full(:, :, t_index) = wavelet_coefficients;
    filtered_all_structures(:, :, t_index) = filtered_coefficients;
    filtered_dimples(:, :, t_index) = filtered_by_eccentricity;
end
return;
end

