%% Combined task2 and task3 code
clear all;
close all;

% Start timing the script for performance measurement
tic;

% Dataset Pathing
% Set the root path to your dataset directory
rootPath = "C:\Users\Abdullah\Desktop\speech_audio_pr\GROUP";
dataset = fullfile(rootPath, 'data');

% Parameters for MFCC extraction
frameLength = 30e-3; % 30 ms
hopSize = 10e-3; % 10 ms

% Directory and file setup
% Retrieve all MP3 files from the dataset directory
File_names = dir(fullfile(dataset, '*.mp3'));
N_files = length(File_names); % Total number of files
mfccCoeffs = cell(N_files, 1); % Preallocate cell array for MFCCs

% Extract MFCC features for each file
for i = 1:N_files
    % Construct the full path for each file
    File_path = fullfile(File_names(i).folder, File_names(i).name);

    % Read the audio file
    [audioIn, fs] = audioread(File_path);

    % Initialize audio feature extractor for MFCCs
    audio_FE = audioFeatureExtractor(...
        'SampleRate', fs, ...
        'Window', hamming(round(frameLength * fs), 'periodic'), ...
        'OverlapLength', round(frameLength * fs) - round(hopSize * fs), ...
        'mfcc', true, ...
        'mfccDelta', false, ...
        'mfccDeltaDelta', false);

    % Extract MFCC features from the audio signal
    mfccCoeffs{i} = extract(audio_FE, audioIn);
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
g_var = var(all_mfccCoeffs); % Global variance across all frames from all files

% Apply variance floor and scaling
varianceFloor = 0.01;
scaled_g_var = max(g_var, varianceFloor); % Apply variance floor

% Compute the global covariance matrix
cov_matrix = cov(cat(1, mfccCoeffs{:}));
cov_matrix = diag(diag(cov_matrix)); % Convert to diagonal matrix

% Assume N states for the HMM
N = 8; % 8 states for each HMM

% Calculate average duration per state
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

% Perform k-means clustering
numSymbols = 13; % Choose a suitable number of clusters
[~, symbols] = kmeans(all_mfccCoeffs, numSymbols);

% Discretize each MFCC sequence
discretizedSeqs = cellfun(@(mfccSeq) dsearchn(symbols, mfccSeq), mfccCoeffs, 'UniformOutput', false);

% Ensure symbols are in the correct range
symbolRanges = cellfun(@(seq) [min(seq), max(seq)], discretizedSeqs, 'UniformOutput', false);
disp('Symbol ranges for each sequence (should be between 1 and numSymbols):');
disp(symbolRanges);

% HMM Initialization with Task 2 parameters
HMM.Trans = diag(repmat(selfLoopProb, N, 1)) + diag(repmat(nextStateProb, N-1, 1), 1);
HMM.Emission = zeros(N, numSymbols); % Initialize with zeros for numSymbols columns

%HMM.Emission = emissionCovariances; % Use emission covariances as initialized for continuous HMM

% Training loop initialization
maxIterations = 15; % Maximum number of iterations for training
for iteration = 1:maxIterations
    fprintf('Iteration %d\n', iteration);
    
    % Initialize accumulators for re-estimating model parameters
    transAccumulator = zeros(size(HMM.Trans));
    emisAccumulator = zeros(size(HMM.Emission));

    for s = 1:numel(discretizedSeqs)
        obsSeq = discretizedSeqs{s}; % Discrete symbols for each MFCC sequence

        % Run the forward and backward algorithms
        logForwardMatrix = forward_algorithm_log(HMM, obsSeq);
        logBackwardMatrix = backward_algorithm_log(HMM, obsSeq);
        forwardMatrix = exp(logForwardMatrix);
        backwardMatrix = exp(logBackwardMatrix);

        % State occupation probabilities (gamma)
        gamma = (forwardMatrix .* backwardMatrix) ./ sum(forwardMatrix .* backwardMatrix, 1);

        % Calculate transition probabilities (xi)
        xi = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq);

        % Accumulate for transitions
        for t = 1:(length(obsSeq) - 1)
            transAccumulator = transAccumulator + xi(:, :, t);
        end

        % Update emission accumulator
        for t = 1:length(obsSeq)
            currentSymbol = obsSeq(t);
            emisAccumulator(:, currentSymbol) = emisAccumulator(:, currentSymbol) + gamma(:, t);
        end
    end

    % Re-estimate HMM parameters
    HMM.Trans = transAccumulator ./ sum(transAccumulator, 2);
    HMM.Emission = emisAccumulator ./ sum(emisAccumulator, 2);

    % Display or save the updated HMM parameters
    fprintf('Updated Transition Probabilities:\n');
    disp(HMM.Trans);
    fprintf('Updated Emission Probabilities:\n');
    disp(HMM.Emission);
end

% Elapsed time for script execution
elapsedTime = toc;
fprintf("Total elapsed time: %.2f seconds\n", elapsedTime);

% Define the logsumexp function
function logSum = logsumexp(logV)
    maxLog = max(logV);
    logSum = maxLog + log(sum(exp(logV - maxLog)));
end

% Define the forward algorithm using log probabilities
function logForwardProb = forward_algorithm_log(HMM, obsSeq)
    trans = log(HMM.Trans); % Transition probabilities in log space
    emis = log(HMM.Emission); % Emission probabilities in log space
    numStates = size(trans, 1);
    numObs = size(obsSeq, 1); % Number of observations (frames in MFCC)
    logForwardProb = -inf(numStates, numObs);
    logForwardProb(:, 1) = zeros(numStates, 1); % Initialize with zeros for the first frame

    for t = 2:numObs
        for j = 1:numStates
            logForwardProb(j, t) = logsumexp(logForwardProb(:, t-1) + trans(:, j));
        end
    end
end

% Define the backward algorithm using log probabilities
function logBackwardProb = backward_algorithm_log(HMM, obsSeq)
    trans = log(HMM.Trans); % Transition probabilities in log space
    emis = log(HMM.Emission); % Emission probabilities in log space

    numStates = size(trans, 1);
    T = size(obsSeq, 1); % Length of the observation sequence
    logBackwardProb = -inf(numStates, T);
    logBackwardProb(:, T) = 0; % log(1) at the final step

    for t = (T - 1):-1:1
        for i = 1:numStates
            tempSum = trans(i, :)';
            logBackwardProb(i, t) = logsumexp(tempSum);
        end
    end
end

% Function to calculate transition probabilities (xi)
function xi = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq)
    numStates = size(HMM.Trans, 1);
    T = size(obsSeq, 1); % Length of the observation sequence
    xi = zeros(numStates, numStates, T - 1);

    for t = 1:(T - 1)
        denom = sum(sum(forwardMatrix(:, t) .* HMM.Trans .* backwardMatrix(:, t+1)'), 2);
        for i = 1:numStates
            numer = forwardMatrix(i, t) .* HMM.Trans(i, :) .* backwardMatrix(:, t+1)';
            xi(i, :, t) = numer / denom;
        end
    end
end
