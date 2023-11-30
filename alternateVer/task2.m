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

% Extract MFCC features for each file and check for missing values
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

        % Check for NaN or Inf values in MFCC features
        if any(isnan(mfccCoeffs{i}(:))) || any(isinf(mfccCoeffs{i}(:)))
            fprintf('Warning: NaN or Inf found in MFCC features of file %s\n', File_path);
        end
        
%{
 % Debug: Display some stats about the extracted MFCC features
        fprintf('File: %s\n', File_path);
        disp('Size of MFCC features:');
        disp(size(mfccCoeffs{i}));
        disp('Sample MFCC features:');
        disp(mfccCoeffs{i}(1:min(5, size(mfccCoeffs{i}, 1)), :));  % Display first 5 frames 
%}
    catch
        % Error handling if file is not found
        fprintf('File not found: %s\n', File_path);
        error = error + 1;
    end
end

% Concatenate all MFCC features from different files into one matrix
all_mfccCoeffs = vertcat(mfccCoeffs{:});

% Compute global statistics: mean and variance
Dim = 13; % Dimension of MFCC features
int_mean = zeros(N_files, Dim); % Intermediate mean values

% Calculate mean and variance for each file
for i = 1:N_files
    int_mean(i, :) = mean(mfccCoeffs{i})';
end

 % Compute the global mean and variance
g_mean = mean(int_mean); % Global mean of MFCCs
% Compute the global variance across all frames from all files
g_var = var(all_mfccCoeffs);

 % Display raw global variance for inspection
%disp('Raw Global Variance:');
%disp(g_var);

% Apply variance floor and scaling
varianceFloor = 0.01; % Experiment with different values
scaled_g_var = max(g_var, varianceFloor); % Apply variance floor

% Compare scaled and unscaled variances
%disp('Comparison of Scaled and Unscaled Global Variances:');
%disp([g_var; scaled_g_var]);

% Compute the global covariance matrix
cov_matrix = cov(cat(1, mfccCoeffs{:}));
cov_matrix = diag(diag(cov_matrix)); % Convert to diagonal matrix

% Check consistency between covariance matrix and scaled variance
%disp('Consistency Check: Covariance Matrix Diagonal vs. Scaled Variance');
%disp([diag(cov_matrix), scaled_g_var']); 

 % Check if the diagonal of the covariance matrix matches the global variance
%disp('Check: Diagonal of Covariance Matrix vs. Global Variance');
%disp(diag(cov_matrix) - scaled_g_var'); 

% Debugging code: Report number of file reading errors
if error > 0
    fprintf("MFCCs not extracted fully, there were %d errors\n", error);
else
    fprintf('MFCCs extracted and statistics computed for %d files\n', N_files);
end

%{
  % Display computed global statistics for verification
disp('Global Mean:');
disp(g_mean);

disp('Scaled Global Variance:');
disp(scaled_g_var);

disp('Scaled Global Covariance:');
disp(cov_matrix);
  
%}

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

%% debug code to check the HMM initialisation was successful, comment out if not debugging.
 
 % Display state transition probabilities and their dimensions
disp('State Transition Probabilities:');
fprintf('Self-loop probability: %f\n', selfLoopProb);
fprintf('Next state probability: %f\n', nextStateProb);
disp('Dimensions of State Transition Probabilities:');
disp(size([selfLoopProb, nextStateProb]));  % Assuming a 2-element vector for simplicity
 

 % Display entry and exit probabilities and their dimensions
disp('Entry Probabilities:');
disp(entryProb);
disp('Dimensions of Entry Probabilities:');
disp(size(entryProb));
disp('Exit Probabilities:');
disp(exitProb);
disp('Dimensions of Exit Probabilities:');
disp(size(exitProb));

% Display emission probabilities (means and covariances) for each state and their dimensions
disp('Emission Probabilities:');
for state = 1:N
    fprintf('State %d Mean:\n', state);
    disp(emissionMeans(state, :));
    fprintf('Dimensions of State %d Mean:\n', state);
    disp(size(emissionMeans(state, :)));
    
    fprintf('State %d Covariance:\n', state);
    disp(emissionCovariances(:, :, state));
    fprintf('Dimensions of State %d Covariance:\n', state);
    disp(size(emissionCovariances(:, :, state)));
end 

 
%}


% Elapsed time for script execution
elapsedTime = toc;
fprintf("Total elapsed time: %.2f seconds\n", elapsedTime);
