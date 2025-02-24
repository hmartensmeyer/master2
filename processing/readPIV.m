addpath('..\utils\readimx-v2.1.9-win64\readimx-v2.1.9-win64\')

%% Load the first snapshot (change the file path as needed)
filePath = fullfile('..\data\StereoPIV_MPd(2x64x64_50%ov)', 'B00151.vc7');
data = readimx(filePath);

%% Extract the first frame and its components
frameData = data.Frames{1};

% Extract the ACTIVE_CHOICE field.
% This field tells you which candidate (0, 1, or 2) was selected at each grid point.
activeChoice = frameData.Components{9}.Planes{1};

%% Extract the raw U, V, and W candidate data
% U candidates:
U0_raw = frameData.Components{1}.Planes{1};  % Component 1: U0
U1_raw = frameData.Components{3}.Planes{1};  % Component 3: U1
U2_raw = frameData.Components{5}.Planes{1};  % Component 5: U2

% V candidates:
V0_raw = frameData.Components{2}.Planes{1};  % Component 2: V0
V1_raw = frameData.Components{4}.Planes{1};  % Component 4: V1
V2_raw = frameData.Components{6}.Planes{1};  % Component 6: V2

% W candidates:
W0_raw = frameData.Components{11}.Planes{1}; % Component 11: W0
W1_raw = frameData.Components{12}.Planes{1}; % Component 12: W1
W2_raw = frameData.Components{13}.Planes{1}; % Component 13: W2

%% Convert raw data to physical units (m/s) using the respective scale factors
% (The scale conversion is: physical value = raw_value * Slope + Offset)
U0 = U0_raw * frameData.Components{1}.Scale.Slope + frameData.Components{1}.Scale.Offset;
U1 = U1_raw * frameData.Components{3}.Scale.Slope + frameData.Components{3}.Scale.Offset;
U2 = U2_raw * frameData.Components{5}.Scale.Slope + frameData.Components{5}.Scale.Offset;

V0 = V0_raw * frameData.Components{2}.Scale.Slope + frameData.Components{2}.Scale.Offset;
V1 = V1_raw * frameData.Components{4}.Scale.Slope + frameData.Components{4}.Scale.Offset;
V2 = V2_raw * frameData.Components{6}.Scale.Slope + frameData.Components{6}.Scale.Offset;

W0 = W0_raw * frameData.Components{11}.Scale.Slope + frameData.Components{11}.Scale.Offset;
W1 = W1_raw * frameData.Components{12}.Scale.Slope + frameData.Components{12}.Scale.Offset;
W2 = W2_raw * frameData.Components{13}.Scale.Slope + frameData.Components{13}.Scale.Offset;

%% Build the final velocity fields using the ACTIVE_CHOICE mask
% Initialize final fields with the same size as the activeChoice array.
finalU = zeros(size(activeChoice));
finalV = zeros(size(activeChoice));
finalW = zeros(size(activeChoice));

% Create logical masks for each candidate:
mask0 = (activeChoice == 0);
mask1 = (activeChoice == 1);
mask2 = (activeChoice == 2);

% For U components:
finalU(mask0) = U0(mask0);
finalU(mask1) = U1(mask1);
finalU(mask2) = U2(mask2);

% For V components:
finalV(mask0) = V0(mask0);
finalV(mask1) = V1(mask1);
finalV(mask2) = V2(mask2);

% For W components:
finalW(mask0) = W0(mask0);
finalW(mask1) = W1(mask1);
finalW(mask2) = W2(mask2);

%% subtract mean flow

% Create a logical mask for values in finalU that are >= 0.05
mask = finalU >= 0.05;

% Subtract 0.25 from only those elements
finalU(mask) = finalU(mask) - mean(finalU(mask), 'all');

%% Visualize the velocity fields

% 1. Quiver plot for the in-plane (U, V) velocity field:
figure;
quiver(finalU, finalV);
title('Final In-Plane Velocity Field (U, V) [m/s]');
xlabel('X grid index');
ylabel('Y grid index');

% 2. Display the velocity magnitude (from U and V) as an image:
velocityMagnitude = sqrt(finalU.^2 + finalV.^2);
figure;
imagesc(velocityMagnitude);
colormap('jet');
colorbar;
title('Velocity Magnitude (U, V) [m/s]');

% 3. Display the out-of-plane velocity field (W) as an image:
figure;
imagesc(finalW);
colormap('jet');
colorbar;
title('Final Out-of-Plane Velocity Field (W) [m/s]');

%% Calculate the in-plane (z-component) vorticity, omega_z
% Compute the gradients of V and U. Note: gradient assumes unit spacing.
[dV_dx, dV_dy] = gradient(finalV, 0.0572102e-3);  % dV/dx and dV/dy
[dU_dx, dU_dy] = gradient(finalU, 0.0572102e-3);  % dU/dx and dU/dy

% Calculate omega_z (vorticity about the z-axis)
omega_z = dV_dx - dU_dy;

% Display basic statistics for omega_z:
fprintf('omega_z: min = %f, max = %f, mean = %f\n', ...
    min(omega_z(:)), max(omega_z(:)), mean(omega_z(:)));

%% Visualize omega_z
figure;
imagesc(omega_z);
colormap('jet');
%clim([-0.02, 0.02]);
colorbar;
title('In-Plane Vorticity \omega_z = dV/dx - dU/dy');

% %% Compute omega_z using a 5-point central difference method
% 
% % Assume uniform grid spacing h = 1. If your grid spacing is different, set h accordingly.
% h = 1;
% kernel = [-1, 8, 0, -8, 1] / (12 * h);
% 
% % Compute the x-derivative of V using the 5-point formula along the columns:
% % imfilter applies the kernel along the horizontal direction.
% dV_dx = imfilter(finalV, kernel, 'replicate', 'conv');
% 
% % Compute the y-derivative of U using the 5-point formula along the rows:
% % Use the transpose of the kernel for the vertical direction.
% dU_dy = imfilter(finalU, kernel', 'replicate', 'conv');
% 
% % Calculate omega_z (vorticity about the z-axis) as:
% omega_z = dV_dx - dU_dy;
% 
% %% Visualize the 5-point central difference omega_z
% figure;
% imagesc(omega_z);
% colormap('jet');
% clim([-0.02, 0.02]);
% colorbar;
% title('In-Plane Vorticity \omega_z (5-Point Central Difference)');
% xlabel('X grid index');
% ylabel('Y grid index');


%% Parameters and 5-Point Central Difference Kernel
h = 0.0572102e-3;  % grid spacing; change if needed
kernel = [-1, 8, 0, -8, 1] / (12 * h);

%% Compute spatial derivatives for each velocity component using imfilter
% Derivatives for U:
dU_dx = imfilter(finalU, kernel, 'replicate', 'conv');
dU_dy = imfilter(finalU, kernel', 'replicate', 'conv');

% Derivatives for V:
dV_dx = imfilter(finalV, kernel, 'replicate', 'conv');
dV_dy = imfilter(finalV, kernel', 'replicate', 'conv');

% Derivatives for W:
dW_dx = imfilter(finalW, kernel, 'replicate', 'conv');
dW_dy = imfilter(finalW, kernel', 'replicate', 'conv');

%% Compute lambda_2 for each grid point
% Preallocate lambda2Field with the same size as the velocity fields:
[nrows, ncols] = size(finalU);
lambda2Field = nan(nrows, ncols);

% Loop over all grid points:
for i = 1:nrows
    for j = 1:ncols
        % Construct the velocity gradient tensor, J, at point (i,j)
        % Note: We set derivatives with respect to z to zero.
        J = [ dU_dx(i,j), dU_dy(i,j), 0;
              dV_dx(i,j), dV_dy(i,j), 0;
              dW_dx(i,j), dW_dy(i,j), 0 ];
          
        % Compute the symmetric (S) and antisymmetric (Omega) parts
        S = 0.5 * (J + J');
        Omega = 0.5 * (J - J');
        
        % Compute T = S^2 + Omega^2
        T = S * S + Omega * Omega;
        
        % Compute eigenvalues of T and sort them in ascending order
        eigVals = sort(eig(T));
        
        % The middle eigenvalue is lambda_2 (by convention)
        lambda2Field(i,j) = eigVals(2);
    end
end

%% Plot the lambda_2 field
figure;
imagesc(lambda2Field);
colormap('jet');
colorbar;
%clim([-2e-4, 2e-4]);
title('Lambda_2 Field');
xlabel('X grid index');
ylabel('Y grid index');

%% New method of approximating lambda2 in two dimensions

% Assuming finalU and finalV are already defined as your in-plane velocity fields

% % Compute the spatial derivatives using MATLAB's gradient function
% [du_dx, du_dy] = gradient(finalU);
% [dv_dx, dv_dy] = gradient(finalV);

% Approximate lambda_2 using the formula:
%   lambda2 ≈ (du/dx)^2 + (dv/dx) * (du/dy)
lambda2_approx = dU_dx.^2 + dV_dx .* dU_dy;

% Visualize the approximate lambda_2 field
figure;
imagesc(lambda2_approx);
colormap('jet');
colorbar;
%clim([-2e4,2e4])
title('Approximate \lambda_2 Field (2D)');
xlabel('X grid index');
ylabel('Y grid index');

%%

% Assuming lambda2_approx has been computed as:
% lambda2_approx = du_dx.^2 + dv_dx .* du_dy;

% Create a binary mask where lambda2_approx is negative
negativeRegions = lambda2_approx < -10000;

% Visualize the negative lambda_2 regions as a binary image
figure;
imagesc(negativeRegions);
colormap('gray');  % using a grayscale colormap for binary visualization
colorbar;
title('Regions where \lambda_2 < 0');
xlabel('X grid index');
ylabel('Y grid index');

% Optionally, overlay the mask on the velocity magnitude or any other background.

%% ENSTROPHY

% Calculate the in-plane vorticity (omega_z)
[dV_dx, dV_dy] = gradient(finalV, 0.0572102e-3);
[dU_dx, dU_dy] = gradient(finalU, 0.0572102e-3);
omega_z = dV_dx - dU_dy;

% Calculate the enstrophy field.
% Enstrophy is commonly defined as (1/2) * (vorticity)^2.
enstrophy = 0.5 * omega_z.^2;

% Visualize the enstrophy field:
figure;
imagesc(enstrophy);
colormap('hot');  % Using a 'hot' colormap for better contrast.
colorbar;
%clim([0,5e4])
title('Enstrophy Field');
xlabel('X grid index');
ylabel('Y grid index');
