% Step 1: Load the video
videoFile = '..\data\SZ_VFD10p5Hz_TimeResolved_Run1_720p.mp4'; % Replace with your video file name
vidObj = VideoReader(videoFile);


%% Specify frame range to process
startFrame = 1; % Start frame index
endFrame = 7229; % End frame index
frameRate = vidObj.FrameRate;
% Calculate time to skip to the starting frame
startTime = (startFrame - 1) / frameRate;
vidObj.CurrentTime = startTime;
% Preallocate space for the selected frames
numSelectedFrames = endFrame - startFrame + 1;
frames = cell(numSelectedFrames, 1);
% Read specific range of frames
disp('Reading selected video frames...');
selectedIdx = 1; % Index for the preallocated frames array
frameIdx = startFrame;
while hasFrame(vidObj) && frameIdx <= endFrame
disp(frameIdx);
currentFrame = readFrame(vidObj); % Read the current frame
frames{selectedIdx} = currentFrame; % Store the frame
disp(['Reading frame: ', num2str(frameIdx)]); % Display progress
selectedIdx = selectedIdx + 1;
frameIdx = frameIdx + 1;
end
disp(['Finished reading frames from ', num2str(startFrame), ' to ', num2str(endFrame), '.']);


%% Display the selected frames using imagesc
disp('Displaying selected frames...');
figure;
for i = 1:numSelectedFrames
frame = frames{i}; % Get the current frame
% Check if the frame is empty (to avoid errors)
if isempty(frame)
continue;
end
% Convert frame to grayscale or directly use RGB
% Option 1: Grayscale display (optional)
% grayFrame = rgb2gray(frame);
% imagesc(grayFrame); colormap gray;
% Option 2: RGB display
imagesc(frame);
axis equal;
axis tight;
title(['Frame Index: ', num2str(startFrame + i - 1)]);
pause(0.5); % Adjust pause duration for playback speed
end
disp('Finished displaying selected frames.');


%% Step 2: Analyze green channel intensity (Robust version)
disp('Analyzing green channel intensity...');
% Preallocate array to store green channel mean intensity
greenIntensity = nan(15361, 1); % Use NaN to track skipped frames
% Loop through frames and calculate green channel intensity
for i = 1:numSelectedFrames
frame = frames{i}; % Get the current frame
if isempty(frame)
% Skip empty frames
disp(['Frame ', num2str(i), ' is empty. Skipping...']);
continue;
end
greenChannel = frame(:, :, 2); % Extract the green channel
greenIntensity(i) = mean(greenChannel(:)); % Compute mean green intensity
end
% Replace NaN with 0 for plotting, if necessary
greenIntensity(isnan(greenIntensity)) = 0;
%% Plot green channel intensity over frames
figure;
plot(1:numSelectedFrames, greenIntensity(1:numSelectedFrames), 'g', 'LineWidth', 1.5);
xlabel('Frame index');
ylabel('Green intensity');
title('Green channel intensity');
xlim([0,80]);
grid on;
% Threshold for identifying laser frames (adjust as needed)
threshold = 25;
laserFrames = find(greenIntensity > threshold);
disp(['Identified ', num2str(length(laserFrames)), ' laser-dominated frames.']);
%disp('Indices of laser frames:');
%disp(laserFrames);


%% Step 3: Filter frames based on green intensity
disp('Filtering frames with mean green intensity < 25...');
% Preallocate array for mean green intensity
greenIntensity = zeros(numSelectedFrames, 1);
filteredFrames = {}; % Initialize a cell array for filtered frames
filteredIdx = 1; % Index for filtered frames
% Loop through selected frames and calculate green intensity
for i = 1:numSelectedFrames
    frame = frames{i}; % Get the current frame
    greenChannel = frame(:, :, 2); % Extract the green channel
    greenIntensity(i) = mean(greenChannel(:)); % Compute mean green intensity
    % Check if the frame meets the intensity condition
    if greenIntensity(i) < 25
        filteredFrames{filteredIdx} = frame; % Keep the frame
        filteredIdx = filteredIdx + 1;
    end
end
disp(['Filtered ', num2str(length(filteredFrames)), ' frames with mean green intensity < 25.']);
% Display the filtered frames using imagesc
disp('Displaying filtered frames...');
figure;
for i = 1:length(filteredFrames)
    frame = filteredFrames{i}; % Get the filtered frame
    imagesc(frame); % Display the frame
    axis equal;
    axis tight;
    title(['Filtered Frame ', num2str(i)]);
    pause(0.05); % Adjust pause duration for playback speed
end
disp('Finished displaying filtered frames.');


%% Step 4: Convert filtered frames to grayscale and save to a .mat file
disp('Converting filtered frames to grayscale and saving to a .mat file...');
startFrame = 1;
% Retrieve the indices of the filtered frames
filteredFrameIndices = find(greenIntensity < 25); % Get the indices of frames that meet the condition
% Initialize a cell array for grayscale filtered frames
filteredFramesGray = cell(length(filteredFrameIndices), 1);
% Convert each filtered frame to grayscale
for i = 1:length(filteredFrameIndices)
originalFrame = filteredFrames{i}; % Get the original filtered frame
grayFrame = rgb2gray(originalFrame); % Convert to grayscale
filteredFramesGray{i} = grayFrame; % Store the grayscale frame
disp(i)
end
% Calculate timestamps for the filtered frames
filteredTimestamps = (startFrame + filteredFrameIndices - 1) / frameRate; % Convert frame indices to timestamps
filteredTimeindeces = startFrame + filteredFrameIndices - 1; % Convert frame indices to timestamps
% Save grayscale filtered frames and timestamps to a .mat file
save('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_720p.mat', 'filteredFramesGray', 'filteredTimeindeces', "filteredTimestamps", '-v7.3');
disp('Filtered grayscale frames and timestamps saved to filtered_video_gray.mat.');
