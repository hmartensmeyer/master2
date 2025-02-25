% Load the image
filePath = 'calibrationData\G0051138.JPG';
image = imread(filePath);
imshow(image);
[x, y] = ginput(4); % Manually click on 4 known circles
%%
I = rgb2gray(image);
I = imadjust(I, stretchlim(I), []);

%%
% Load the image
filePath = 'calibrationData\G0051138.JPG';
I = imread(filePath);
% Display image and manually select 4 points
imshow(I);
title('Click on Four Known Circles (Corners)');
[x, y] = ginput(4); % User selects four points
% Convert to grayscale and enhance contrast
%I = rgb2gray(image);
I = imadjust(I, stretchlim(I), []);
% Define the corresponding real-world points (assuming a perfect rectangle)
% These should correspond to where the four selected points "should be" after correction
realWorldPoints = [0, 0; 1000, 0; 1000, 500; 0, 500]; % Adjust scale if necessary
% Compute the homography transformation
tform = fitgeotrans([x, y], realWorldPoints, 'projective');
% Apply the transformation to warp the image
rectifiedImage = imwarp(I, tform);
%% Display the original and rectified images for comparison
figure;
colormap('gray');
subplot(1,2,1);
imshow(I);
title('Original Image');
subplot(1,2,2);
imshow(rectifiedImage);
title('Rectified Image (After Calibration)');

%%

%% Step 1: Load the Dotted Image
dottedImage = imread('calibrationData\G0051138.JPG');  % Replace with your dotted image filename
figure;
imshow(dottedImage);
title('Dotted Image');

%% Step 2: Load the First Frame of the Calibration Video
videoFile = 'calibrationData\LiveCalibration_3_fixed.mp4';  % Replace with your video filename
v = VideoReader(videoFile);
calibFrame = readFrame(v);  % Read the first frame
figure;
imshow(calibFrame);
title('First Frame of Calibration Video');

%% Step 3: Define the Fixed Real-World Coordinates for the Dots
% Replace these values with your known real-world coordinates for the dots.
fixedPoints = [ 0,   0;
               1000,  0;
               1000,500;
                 0,500];

%% Step 4: Select Corresponding Points in the Dotted Image Using cpselect
% Here, use the dotted image as the "moving" image and the first calibration frame as the "fixed" image.
[movingPoints, ~] = cpselect(dottedImage, calibFrame, 'Wait', true);

% Verify that the number of points matches the fixed points.
if size(movingPoints, 1) ~= size(fixedPoints, 1)
    error('The number of selected moving points (%d) must equal the number of fixed points (%d).', ...
          size(movingPoints,1), size(fixedPoints,1));
end

%% Step 5: Compute the Projective Transformation (Homography)
tform = fitgeotrans(movingPoints, fixedPoints, 'projective');

%% Step 6: Apply the Transformation to the Dotted Image
registeredImage = imwarp(dottedImage, tform);

%% Step 7: Display the Registered Image
figure;
imshow(registeredImage);
title('Registered Dotted Image in Water Surface Coordinates');

%% Optional: Save the Registered Image
imwrite(registeredImage, 'registeredDottedImage.png');
