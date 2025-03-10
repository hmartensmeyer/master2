addpath('..\utils\readimx-v2.1.9-win64\readimx-v2.1.9-win64\')

%% Parameters
% Set the grid spacing (adjust as needed)
h = 0.0572102;
% 5-point central difference kernel (for derivatives in x)
kernel = [-1, 8, 0, -8, 1] / (12 * h);

%% Load Data
filePath = fullfile('..\data\StereoPIV_MPd(2x64x64_50%ov)', 'B00157.vc7');
data = readimx(filePath);
frameData = data.Frames{1};

%% Extract ACTIVE_CHOICE and candidate data
activeChoice = frameData.Components{9}.Planes{1};

% U candidates:
U0_raw = frameData.Components{1}.Planes{1};
U1_raw = frameData.Components{3}.Planes{1};
U2_raw = frameData.Components{5}.Planes{1};

% V candidates:
V0_raw = frameData.Components{2}.Planes{1};
V1_raw = frameData.Components{4}.Planes{1};
V2_raw = frameData.Components{6}.Planes{1};

% W candidates:
W0_raw = frameData.Components{11}.Planes{1};
W1_raw = frameData.Components{12}.Planes{1};
W2_raw = frameData.Components{13}.Planes{1};

%% Convert Raw Data to Physical Units (m/s)
U0 = U0_raw * frameData.Components{1}.Scale.Slope + frameData.Components{1}.Scale.Offset;
U1 = U1_raw * frameData.Components{3}.Scale.Slope + frameData.Components{3}.Scale.Offset;
U2 = U2_raw * frameData.Components{5}.Scale.Slope + frameData.Components{5}.Scale.Offset;

V0 = V0_raw * frameData.Components{2}.Scale.Slope + frameData.Components{2}.Scale.Offset;
V1 = V1_raw * frameData.Components{4}.Scale.Slope + frameData.Components{4}.Scale.Offset;
V2 = V2_raw * frameData.Components{6}.Scale.Slope + frameData.Components{6}.Scale.Offset;

W0 = W0_raw * frameData.Components{11}.Scale.Slope + frameData.Components{11}.Scale.Offset;
W1 = W1_raw * frameData.Components{12}.Scale.Slope + frameData.Components{12}.Scale.Offset;
W2 = W2_raw * frameData.Components{13}.Scale.Slope + frameData.Components{13}.Scale.Offset;

%% Build Final Velocity Fields Using ACTIVE_CHOICE
finalU = zeros(size(activeChoice));
finalV = zeros(size(activeChoice));
finalW = zeros(size(activeChoice));

mask0 = (activeChoice == 0);
mask1 = (activeChoice == 1);
mask2 = (activeChoice == 2);

finalU(mask0) = U0(mask0);
finalU(mask1) = U1(mask1);
finalU(mask2) = U2(mask2);

finalV(mask0) = V0(mask0);
finalV(mask1) = V1(mask1);
finalV(mask2) = V2(mask2);

finalW(mask0) = W0(mask0);
finalW(mask1) = W1(mask1);
finalW(mask2) = W2(mask2);

%% Subtract Mean Flow from finalU (only for values >= 0.05)

mask = finalU >= 0.05;
finalU(mask) = finalU(mask) - mean(finalU(mask), 'all');

%% Compute Spatial Derivatives Using Central Differences (imfilter)
% For finalU:
dU_dx = imfilter(finalU, kernel, 'replicate', 'conv');
dU_dy = imfilter(finalU, kernel', 'replicate', 'conv');

% For finalV:
dV_dx = imfilter(finalV, kernel, 'replicate', 'conv');
dV_dy = imfilter(finalV, kernel', 'replicate', 'conv');

% For finalW:
dW_dx = imfilter(finalW, kernel, 'replicate', 'conv');
dW_dy = imfilter(finalW, kernel', 'replicate', 'conv');

%% Compute In-Plane Vorticity and Enstrophy
% Vorticity (omega_z) is defined as dV/dx - dU/dy
omega_z = dV_dx - dU_dy;
enstrophy = 0.5 * omega_z.^2;  % Enstrophy = 0.5 * (vorticity)^2

%% Compute Full Lambda_2 Field Using Eigenvalue Analysis
% Preallocate lambda2Field
[nrows, ncols] = size(finalU);
lambda2Field = nan(nrows, ncols);

% Loop over all grid points
for i = 1:nrows
    for j = 1:ncols
        % Construct the 3x3 velocity gradient tensor at (i,j)
        % (Note: derivatives in z are set to zero)
        J = [ dU_dx(i,j), dU_dy(i,j), 0;
              dV_dx(i,j), dV_dy(i,j), 0;
              dW_dx(i,j), dW_dy(i,j), 0 ];
        % Compute symmetric (S) and antisymmetric (Omega) parts
        S = 0.5 * (J + J');
        Omega = 0.5 * (J - J');
        % Compute T = S^2 + Omega^2
        T = S * S + Omega * Omega;
        % Sort eigenvalues of T in ascending order and take the middle one
        eigVals = sort(eig(T));
        lambda2Field(i,j) = eigVals(2);
    end
end

%% Compute Approximate Lambda_2 in 2D
% Approximation: lambda2 ≈ (du/dx)^2 + (dv/dx)*(du/dy)
lambda2_approx = dU_dx.^2 + dV_dx .* dU_dy;

%% Create Binary Mask for Regions Where Approximate lambda_2 is Negative
negativeRegions = lambda2_approx < 0;

%% Visualization
% Quiver plot of the in-plane (U,V) velocity field
figure;
quiver(finalU, finalV);
title('Final In-Plane Velocity Field (U, V) [m/s]');
xlabel('X grid index');
ylabel('Y grid index');

% Velocity magnitude image
figure;
imagesc(sqrt(finalU.^2 + finalV.^2));
colormap('jet');
colorbar;
title('Velocity Magnitude (U, V) [m/s]');

% Out-of-plane (W) velocity field image
figure;
imagesc(finalW);
colormap('jet');
colorbar;
title('Final Out-of-Plane Velocity Field (W) [m/s]');

% Vorticity (omega_z) image
figure;
imagesc(omega_z);
colormap('jet');
colorbar;
title('In-Plane Vorticity \omega_z = dV/dx - dU/dy');

% Full lambda_2 field image
figure;
imagesc(lambda2Field);
colormap('jet');
colorbar;
title('\lambda_2 Field');

% Approximate lambda_2 field image
figure;
imagesc(lambda2_approx);
colormap('jet');
colorbar;
%clim([-3e4, 11e4])
title('Approximate \lambda_2 Field (2D)');

% Binary mask for negative approximate lambda_2 regions
figure;
imagesc(negativeRegions);
colormap('gray');
colorbar;
title('Regions where \lambda_2 < 0');

% Enstrophy field image
figure;
imagesc(enstrophy);
colormap('hot');
colorbar;
title('Enstrophy Field');
xlabel('X grid index');
ylabel('Y grid index');

%% Compute Approximate Lambda_2 in 2D
% Approximation: lambda2 ≈ (du/dx)^2 + (dv/dx)*(du/dy)
lambda2_approx = dU_dx.^2 + dV_dx .* dU_dy;

%% Create a Binary Mask of Negative Regions
% Use your chosen threshold; here we use -10000.
negativeRegions = lambda2_approx < -0.007;

%% Identify Connected Components and Filter by Size
% Find connected components (using 8-connectivity)
cc = bwconncomp(negativeRegions, 8);

% Compute region properties (area and pixel indices)
stats = regionprops(cc, 'Area', 'PixelIdxList');

% Remove regions smaller than 4 pixels.
minArea = 5;
validIdx = find([stats.Area] > minArea);
stats_valid = stats(validIdx);

% Create a new binary mask including all valid regions.
selectedMask = false(size(lambda2_approx));
for k = 1:length(stats_valid)
    selectedMask(stats_valid(k).PixelIdxList) = true;
end

%% Visualize the Selected Negative Regions
figure;
imagesc(selectedMask);
colormap('gray');  % Gray colormap for binary visualization
colorbar;
title('\lambda_2 < \lambda_{thr} (Area > 5 pixels)');
xlabel('X grid index');
ylabel('Y grid index');

%% ---- Try to superimpose red circles on the detected vortices ----

% (Optional) Clip super outlier vectors in the velocity field
% mag = sqrt(finalU.^2 + finalV.^2);  % Compute magnitudes
% threshold = 0.05;  % Set threshold (adjust as necessary)
% idx = mag > threshold;  % Find indices where magnitude exceeds threshold
% finalU(idx) = threshold * finalU(idx) ./ mag(idx);
% finalV(idx) = threshold * finalV(idx) ./ mag(idx);

% Find the Centers of Valid Negative Regions
% Extract region properties including the centroid
regionStats = regionprops(selectedMask, 'Centroid');

% Extract centroid coordinates
centroids = cat(1, regionStats.Centroid);  % Each row is [x, y]

% Plot the Quiver and Overlay Red Circles at the Centroids
figure;
quiver(finalU, finalV);
title('Velocity Field (U, V) with potential vortices as red circles');
%xlabel('X grid index');
%ylabel('Y grid index');
hold on;

% Overlay a red circle for each centroid
for k = 1:size(centroids,1)
    plot(centroids(k,1), centroids(k,2), 'ro', 'MarkerSize', 20, 'LineWidth', 2);
end
hold off;
