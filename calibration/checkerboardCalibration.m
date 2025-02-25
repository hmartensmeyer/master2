% Define video file and create VideoReader object
videoFile = 'calibrationData\LiveCalibration_3_fixed.mp4';
v = VideoReader(videoFile);

% Initialize variables to store image points and corresponding frames
allImagePoints = [];
selectedFrames = {};
frameCount = 0;
frameInterval = 30;  % Check every 30th frame
expectedBoardSize = [6,8];  % Expected internal checkerboard size (yielding 35 points)

% Skip the first 400 frames
while hasFrame(v) && frameCount < 400
    readFrame(v);  % Read and discard the frame
    frameCount = frameCount + 1;
end

% Process frames after the first 400
while hasFrame(v)
    frame = readFrame(v);
    frameCount = frameCount + 1;
    
    % Process only every 30th frame
    if mod(frameCount, frameInterval) ~= 0
        continue;
    end
    
    % Detect checkerboard points in the frame
    [imagePoints, boardSize] = detectCheckerboardPoints(frame);
    
    % Check if the detected board matches the expected size (35 points)
    if ~isempty(imagePoints) && isequal(boardSize, expectedBoardSize)
        allImagePoints(:,:,end+1) = imagePoints; %#ok<SAGROW>
        disp(imagePoints)
        selectedFrames{end+1} = frame; %#ok<SAGROW>
        fprintf('Detected full checkerboard in frame %d\n', frameCount);
    else
        fprintf('Frame %d did not yield expected %d points. Detected %d points instead. Skipping.\n', ...
            frameCount, prod(expectedBoardSize), size(imagePoints,1));
    end
end

% Check if we have enough valid frames for calibration
if isempty(allImagePoints)
    error('No valid frames were detected for calibration.');
end

% Define the size of each square on the checkerboard (e.g., in millimeters)
squareSize = 100;  % Adjust based on your calibration pattern

% Generate world coordinates for the checkerboard keypoints
worldPoints = generateCheckerboardPoints(expectedBoardSize, squareSize);

% Calibrate the camera using the detected image points and world coordinates
[params, imagesUsed, estimationErrors] = estimateCameraParameters(allImagePoints, worldPoints);

%% Display calibration results
displayErrors(estimationErrors, params);
figure; showReprojectionErrors(params);
figure; showExtrinsics(params, 'CameraCentric');

% %%
% 
% % Define the known distance to the ceiling (in millimeters)
% ceilingDistance = 1000;  % mm
% 
% % Retrieve intrinsic parameters from the calibration results
% fx = params.FocalLength(1);
% fy = params.FocalLength(2);
% cx = params.PrincipalPoint(1);
% cy = params.PrincipalPoint(2);
% 
% % Read a frame from your new video (or use a selected frame from your calibration video)
% newVideoFile = 'calibrationData\LiveCalibration_3_fixed.mp4';
% vNew = VideoReader(newVideoFile);
% frameNew = readFrame(vNew);
% 
% % Get the dimensions of the first selected frame
% [frameH, frameW, ~] = size(selectedFrames{1});
% 
% % Create a grid of pixel coordinates based on the frame dimensions
% [U, V] = meshgrid(1:100:frameW, 1:100:frameH);
% u = U(:);
% v = V(:);
% 
% % For one calibration frame, assume you want to use the points from that frame:
% imgPts = allImagePoints(:,:,3); % N x 2 matrix (pixel coordinates)
% worldPts = worldPoints;         % N x 2 matrix (in mm)
% 
% % Compute the projective transformation (homography)
% tform = fitgeotrans(imgPts, worldPts, 'projective');
% 
% % Map the grid points from image coordinates to world coordinates using the homography
% worldCoords = transformPointsForward(tform, [u, v]);
% 
% % Compute the corresponding world coordinates on the ceiling plane
% X_world = ((u - cx) * ceilingDistance) / fx;
% Y_world = ((v - cy) * ceilingDistance) / fy;
% Z_world = repmat(ceilingDistance, size(u));
% 
% % For visualization, we'll overlay the projected pixel positions and label one example point
% figure;
% imshow(frameNew);
% hold on;
% 
% % Plot the selected image points (these are still in pixel coordinates)
% plot(u, v, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
% 
% % Loop over each point and compute/display its real-world coordinates
% for i = 1:length(u)
%     % Convert the image pixel (u,v) into real-world coordinates on the ceiling plane
%     X_world = ((u(i) - cx) * ceilingDistance) / fx;
%     Y_world = ((v(i) - cy) * ceilingDistance) / fy;
% 
%     % Create a label with the X and Y coordinates (in mm)
%     label = sprintf('(%0.1f, %0.1f)', X_world, Y_world);
% 
%     % Display the label next to the red dot with a small offset
%     text(u(i) + 5, v(i), label, 'Color', 'yellow', 'FontSize', 8, ...
%          'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
% end
% 
% % Optionally, adjust the axis ticks to display real-world coordinates.
% ax = gca;
% xTickPixels = ax.XTick;
% xTickWorld = ((xTickPixels - cx) * ceilingDistance) / fx;
% yTickPixels = ax.YTick;
% yTickWorld = ((yTickPixels - cy) * ceilingDistance) / fy;
% 
% ax.XTickLabel = arrayfun(@(x) sprintf('%.1f', x), xTickWorld, 'UniformOutput', false);
% ax.YTickLabel = arrayfun(@(y) sprintf('%.1f', y), yTickWorld, 'UniformOutput', false);
% 
% xlabel('X (mm)');
% ylabel('Y (mm)');
% title('Frame with Real-World Coordinates for All Red Dots');
% hold off;
% 
% %%
% 
% 
% % %% Insert image into this frame
% % 
% % dottedImage = imread('calibrationData\G0051135.JPG');
% % imshow(dottedImage);
% % 
% % %% Step 2: Define Real-World Coordinates for the Dots
% % % Specify the fixed points (water surface coordinates) for the dots.
% % % Adjust these values to match the real distances and order of your dots.
% % fixedPoints = [ 0,   0;
% %                1000,  0;
% %                1000,500;
% %                  0,500];
% % 
% % %% Step 3: Select Corresponding Points in the Dotted Image
% % % cpselect, when 'Wait' is true, returns two outputs.
% % % The second output is ignored here.
% % [movingPoints, ~] = cpselect(dottedImage, zeros(size(dottedImage), 'uint8'), 'Wait', true);
% % 
% % % Verify the number of points selected matches the fixed points.
% % if size(movingPoints, 1) ~= size(fixedPoints, 1)
% %     error('The number of selected moving points (%d) must equal the number of fixed points (%d).', ...
% %           size(movingPoints,1), size(fixedPoints,1));
% % end
% % 
% % %% Step 4: Compute the Projective Transformation (Homography)
% % tform = fitgeotrans(movingPoints, fixedPoints, 'projective');
% % 
% % %% Step 5: Apply the Transformation to the Dotted Image
% % registeredImage = imwarp(dottedImage, tform);
% % 
% % %% Step 6: Display the Registered Image
% % figure;
% % imshow(registeredImage);
% % title('Registered Image in Water Surface Coordinates');
% 
% 
% %% NEW TEST
% 
% % Load a new frame (or use one from your calibration video)
% newVideoFile = 'calibrationData\LiveCalibration_3_fixed.mp4';
% vNew = VideoReader(newVideoFile);
% frameNew = readFrame(vNew);
% 
% % Get the dimensions of the frame
% [frameH, frameW, ~] = size(frameNew);
% 
% % Create a grid of pixel coordinates based on the frame dimensions
% [U, V] = meshgrid(1:100:frameW, 1:100:frameH);
% uGrid = U(:);
% vGrid = V(:);
% 
% % For one calibration frame, assume you want to use the points from that frame:
% imgPts = allImagePoints(:,:,6); % N x 2 matrix (pixel coordinates)
% worldPts = worldPoints;         % N x 2 matrix (in mm)
% 
% % Compute the projective transformation (homography)
% tform = fitgeotrans(imgPts, worldPts, 'projective');
% 
% % Map the grid points from image coordinates to world coordinates using the homography
% worldCoords = transformPointsForward(tform, [uGrid, vGrid]);
% 
% % Display the new frame and overlay the results
% figure;
% imshow(frameNew);
% hold on;
% plot(uGrid, vGrid, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
% 
% % Loop over each point and label with its real-world coordinate
% for i = 1:length(uGrid)
%     label = sprintf('(%0.1f, %0.1f)', worldCoords(i,1), worldCoords(i,2));
%     text(uGrid(i) + 5, vGrid(i), label, 'Color', 'yellow', 'FontSize', 8, ...
%          'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
% end
% hold off;

%% Mapping Using Extrinsics (General Case)
% Assumptions:
%  - The camera extrinsics (R and t) from calibration are available.
%  - The calibration (with squareSize) has set the correct scale.
%  - The ceiling plane in world coordinates is defined as Z = ceilingDistance.

% Known ceiling distance (in millimeters)
ceilingDistance = 2000;  % Adjust as needed

% Retrieve intrinsic parameters from calibration result 'params'
fx = params.FocalLength(1);
fy = params.FocalLength(2);
cx = params.PrincipalPoint(1);
cy = params.PrincipalPoint(2);

% Use extrinsics from a valid calibration frame (e.g., the first valid one)
% Note: t is a row vector; convert to column vector.
R = params.RotationMatrices(:,:,1);
t = params.TranslationVectors(1,:)';  % Column vector (in mm)

% Compute the camera center in world coordinates:
% Given the transformation: X_camera = R * X_world + t,
% the camera center is C = -R' * t.
C = -R' * t;

% Load a new frame
newVideoFile = 'calibrationData\LiveCalibration_3_fixed.mp4';
vNew = VideoReader(newVideoFile);
frameNew = readFrame(vNew);
[frameH, frameW, ~] = size(frameNew);

% Create a grid of pixel coordinates for visualization
[U, V] = meshgrid(1:200:frameW, 1:200:frameH);
uGrid = U(:);
vGrid = V(:);

% Initialize arrays for world coordinates
numPoints = length(uGrid);
worldCoordsExtrinsic = zeros(numPoints, 3);

% Loop over each image point and compute its world coordinate by intersecting
% its ray with the ceiling plane (Z = ceilingDistance).
for i = 1:numPoints
    % Convert pixel coordinate to normalized image coordinate
    x_norm = (uGrid(i) - cx) / fx;
    y_norm = (vGrid(i) - cy) / fy;
    
    % In camera coordinates, the ray is: r(λ) = λ * [x_norm; y_norm; 1]
    % Convert the ray direction to world coordinates:
    % d_world = R' * [x_norm; y_norm; 1]
    d_world = R' * [x_norm; y_norm; 1];
    
    % The ray in world coordinates is: X_world(λ) = C + λ * d_world.
    % We want to find λ such that the Z coordinate of X_world equals ceilingDistance.
    % Let X_world(λ) = [X; Y; Z]. Then:
    %    C(3) + λ * d_world(3) = ceilingDistance
    lambda = (ceilingDistance - C(3)) / d_world(3);
    
    % Compute the intersection point in world coordinates
    Xw = C(1) + lambda * d_world(1);
    Yw = C(2) + lambda * d_world(2);
    Zw = ceilingDistance;  % By definition of the ceiling plane
    worldCoordsExtrinsic(i, :) = [Xw, Yw, Zw];
end

% Display the frame with overlaid grid and labels (world coordinates from extrinsics)
figure;
imshow(frameNew);
hold on;
plot(uGrid, vGrid, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
for i = 1:numPoints
    label = sprintf('(%0.1f, %0.1f)', worldCoordsExtrinsic(i,1), worldCoordsExtrinsic(i,2));
    text(uGrid(i)+5, vGrid(i), label, 'Color', 'yellow', 'FontSize', 8, ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end
hold off;
xlabel('Image X (pixels)');
ylabel('Image Y (pixels)');
title('Projection onto Ceiling Using Extrinsics');

