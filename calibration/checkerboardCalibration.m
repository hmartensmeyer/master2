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
squareSize = 25;  % Adjust based on your calibration pattern

% Generate world coordinates for the checkerboard keypoints
worldPoints = generateCheckerboardPoints(expectedBoardSize, squareSize);

% Calibrate the camera using the detected image points and world coordinates
[params, imagesUsed, estimationErrors] = estimateCameraParameters(allImagePoints, worldPoints);

%% Display calibration results
displayErrors(estimationErrors, params);
figure; showReprojectionErrors(params);
figure; showExtrinsics(params, 'CameraCentric');


%% Insert image into this frame

dottedImage = imread('calibrationData\G0051135.JPG');
imshow(dottedImage);

%% Step 2: Define Real-World Coordinates for the Dots
% Specify the fixed points (water surface coordinates) for the dots.
% Adjust these values to match the real distances and order of your dots.
fixedPoints = [ 0,   0;
               1000,  0;
               1000,500;
                 0,500];

%% Step 3: Select Corresponding Points in the Dotted Image
% cpselect, when 'Wait' is true, returns two outputs.
% The second output is ignored here.
[movingPoints, ~] = cpselect(dottedImage, zeros(size(dottedImage), 'uint8'), 'Wait', true);

% Verify the number of points selected matches the fixed points.
if size(movingPoints, 1) ~= size(fixedPoints, 1)
    error('The number of selected moving points (%d) must equal the number of fixed points (%d).', ...
          size(movingPoints,1), size(fixedPoints,1));
end

%% Step 4: Compute the Projective Transformation (Homography)
tform = fitgeotrans(movingPoints, fixedPoints, 'projective');

%% Step 5: Apply the Transformation to the Dotted Image
registeredImage = imwarp(dottedImage, tform);

%% Step 6: Display the Registered Image
figure;
imshow(registeredImage);
title('Registered Image in Water Surface Coordinates');
