%% Loop over .vc7 files and compute omega_z and lambda_2
clear; clc; close all;
addpath('..\utils\readimx-v2.1.9-win64\readimx-v2.1.9-win64\');

% Specify folder containing vc7 files
folderPath = fullfile('..\data\StereoPIV_MPd(2x64x64_50%ov)');
vc7Files = dir(fullfile(folderPath, '*.vc7'));

% Define 5-point central difference kernel
h = 1;  
kernel = [-1, 8, 0, -8, 1] / (12*h);

for k = 300:length(vc7Files)
    % Read the current vc7 file
    fileName = vc7Files(k).name;
    filePath = fullfile(folderPath, fileName);
    %data = readimx(filePath);
    %frameData = data.Frames{1};  % Use the first snapshot

    % Try reading the file; if it fails, skip this file.
    try
        data = readimx(filePath);
    catch ME
        warning('Skipping file %s due to error: %s', fileName, ME.message);
        break;  % Skip to the next file
    end

    frameData = data.Frames{1};  % Use the first snapshot

    % Extract the ACTIVE_CHOICE mask
    activeChoice = frameData.Components{9}.Planes{1};

    % Convert candidate data to physical units
    % U candidates:
    U0 = frameData.Components{1}.Planes{1} * frameData.Components{1}.Scale.Slope + frameData.Components{1}.Scale.Offset;
    U1 = frameData.Components{3}.Planes{1} * frameData.Components{3}.Scale.Slope + frameData.Components{3}.Scale.Offset;
    U2 = frameData.Components{5}.Planes{1} * frameData.Components{5}.Scale.Slope + frameData.Components{5}.Scale.Offset;

    % V candidates:
    V0 = frameData.Components{2}.Planes{1} * frameData.Components{2}.Scale.Slope + frameData.Components{2}.Scale.Offset;
    V1 = frameData.Components{4}.Planes{1} * frameData.Components{4}.Scale.Slope + frameData.Components{4}.Scale.Offset;
    V2 = frameData.Components{6}.Planes{1} * frameData.Components{6}.Scale.Slope + frameData.Components{6}.Scale.Offset;

    % W candidates:
    W0 = frameData.Components{11}.Planes{1} * frameData.Components{11}.Scale.Slope + frameData.Components{11}.Scale.Offset;
    W1 = frameData.Components{12}.Planes{1} * frameData.Components{12}.Scale.Slope + frameData.Components{12}.Scale.Offset;
    W2 = frameData.Components{13}.Planes{1} * frameData.Components{13}.Scale.Slope + frameData.Components{13}.Scale.Offset;

    % Assemble final velocity fields based on ACTIVE_CHOICE
    finalU = zeros(size(activeChoice));
    finalV = zeros(size(activeChoice));
    finalW = zeros(size(activeChoice));

    finalU(activeChoice==0) = U0(activeChoice==0);
    finalU(activeChoice==1) = U1(activeChoice==1);
    finalU(activeChoice==2) = U2(activeChoice==2);

    finalV(activeChoice==0) = V0(activeChoice==0);
    finalV(activeChoice==1) = V1(activeChoice==1);
    finalV(activeChoice==2) = V2(activeChoice==2);

    finalW(activeChoice==0) = W0(activeChoice==0);
    finalW(activeChoice==1) = W1(activeChoice==1);
    finalW(activeChoice==2) = W2(activeChoice==2);

    % Compute spatial derivatives using imfilter
    dU_dx = imfilter(finalU, kernel, 'replicate', 'conv');
    dU_dy = imfilter(finalU, kernel', 'replicate', 'conv');
    dV_dx = imfilter(finalV, kernel, 'replicate', 'conv');
    dV_dy = imfilter(finalV, kernel', 'replicate', 'conv');
    dW_dx = imfilter(finalW, kernel, 'replicate', 'conv');
    dW_dy = imfilter(finalW, kernel', 'replicate', 'conv');

    % Calculate the in-plane vorticity, omega_z
    omega_z = dV_dx - dU_dy;

    % Calculate lambda_2 field via a pointwise eigenvalue analysis
    [nrows, ncols] = size(finalU);
    lambda2Field = nan(nrows, ncols);
    for i = 1:nrows
        for j = 1:ncols
            % Construct the 3x3 velocity gradient tensor (with z-derivatives set to zero)
            J = [ dU_dx(i,j), dU_dy(i,j), 0;
                  dV_dx(i,j), dV_dy(i,j), 0;
                  dW_dx(i,j), dW_dy(i,j), 0 ];
            S = 0.5*(J + J');      % symmetric part
            Omega = 0.5*(J - J');  % antisymmetric part
            T = S*S + Omega*Omega;
            eigVals = sort(eig(T));  % sort eigenvalues in ascending order
            lambda2Field(i,j) = eigVals(2);  % the middle eigenvalue
        end
    end

    % Plot omega_z and lambda_2 side by side
    figure('Name', fileName, 'NumberTitle', 'off');
    subplot(1,2,1);
    imagesc(omega_z);
    colormap('jet');
    colorbar;
    clim([-0.02, 0.02]);
    title(['\omega_z: ', fileName]);
    xlabel('X grid index'); ylabel('Y grid index');

    subplot(1,2,2);
    imagesc(lambda2Field);
    colormap('jet');
    colorbar;
    clim([-2e-4, 2e-4]);
    title(['\lambda_2: ', fileName]);
    xlabel('X grid index'); ylabel('Y grid index');
    
    drawnow;
end
