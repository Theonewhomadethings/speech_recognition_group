clear all;
close all;

% Start timing the script for performance measurement
tic;

% Dataset Pathing
% Set the root path to your dataset directory
rootPath = "C:\Users\Abdullah\Desktop\speech_audio_pr\GROUP";
dataset = fullfile(rootPath, 'data');

% Parameters for MFCC extraction
% Frame length and hop size define the temporal resolution of the analysis
frameLength = 30e-3; % 30 ms
hopSize = 10e-3; % 10 ms

% Directory and file setup
% Retrieve all MP3 files from the dataset directory
File_names = dir(fullfile(dataset, '*.mp3'));
N_files = length(File_names); % Total number of files
mfccCoeffs = cell(N_files, 1); % Preallocate cell array for MFCCs

% Error handling variable
error = 0;

% Extract MFCC features for each file
for i = 1:N_files
    % Construct the full path for each file
    File_path = fullfile(File_names(i).folder, File_names(i).name);

    try
        % Read the audio file
        [audioIn, fs] = audioread(File_path);

        % Initialize audio feature extractor for MFCCs
        % Configure with Hamming window and specified overlap
        audio_FE = audioFeatureExtractor(...
            'SampleRate', fs, ...
            'Window', hamming(round(frameLength * fs), 'periodic'), ...
            'OverlapLength', round(frameLength * fs) - round(hopSize * fs), ...
            'mfcc', true, ...
            'mfccDelta', false, ...
            'mfccDeltaDelta', false);

        % Extract MFCC features from the audio signal
        mfccCoeffs{i} = extract(audio_FE, audioIn);

    catch
        % Error handling if file is not found
        fprintf('File not found: %s\n', File_path);
        error = error + 1;
    end
end

% Compute global statistics: mean and variance
Dim = 13; % Dimension of MFCC features
int_mean = zeros(N_files, Dim); % Intermediate mean values
int_var = zeros(N_files, Dim);  % Intermediate variance values

% Calculate mean and variance for each file
for i = 1:N_files
    int_mean(i, :) = mean(mfccCoeffs{i})';
    int_var(i, :) = var(mfccCoeffs{i})';
end

% Compute the global mean and variance
g_mean = mean(int_mean); % Global mean of MFCCs
g_var = var(int_var);    % Global variance of MFCCs

% Compute the global covariance matrix and apply variance floor
cov_matrix = cov(cat(1, mfccCoeffs{:}));  % Full covariance matrix
cov_matrix = diag(diag(cov_matrix));      % Convert to diagonal matrix
varianceFloor = 0.0001;                   % Define the variance floor
scaled_g_var = max(g_var, varianceFloor); % Apply variance floor

% Debugging code: Report number of file reading errors
if error > 0
    fprintf("MFCCs not extracted fully, there were %d errors\n", error);
else
    fprintf('MFCCs extracted and statistics computed for %d files\n', N_files);
end

% Display computed statistics for verification
disp('Global Mean:');
disp(g_mean);

disp('Scaled Global Variance:');
disp(scaled_g_var);

disp('Scaled Global Covariance:');
disp(cov_matrix);

% Assume N states for the HMM
N = 8; % 8 states for each HMM

% Calculate average duration per state (assuming equal duration for simplicity)
totalFrames = sum(cellfun(@(c) size(c, 1), mfccCoeffs)); % Total number of frames across all files
avgDurationPerState = totalFrames / (N_files * N); % Average frames per state

% State transition probabilities
selfLoopProb = exp(-1 / (avgDurationPerState - 1)); % Self-loop probability
nextStateProb = 1 - selfLoopProb; % Probability of moving to the next state

% Emission probabilities for each state
emissionMeans = repmat(g_mean, N, 1); % Replicate global mean for each state
emissionCovariances = repmat(diag(scaled_g_var), 1, 1, N); % Replicate diagonal covariance matrix for each state

% Entry and exit probabilities
entryProb = zeros(1, N);
entryProb(1) = 1;
exitProb = zeros(1, N);
exitProb(N) = 1;

% Elapsed time for script execution
elapsedTime = toc;
fprintf("Total elapsed time: %.2f seconds\n", elapsedTime);
