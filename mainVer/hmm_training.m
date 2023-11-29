% Initialization and feature extraction
Folder_path = "C:/Users/Pam Shontelle/Documents/MATLAB/Speech2/development_set";
File_names = dir(fullfile(Folder_path, '*.mp3'));
N_files = 390;
File_arr = cell(N_files, 1);
Fs = cell(N_files, 1);
audio_vector = cell(N_files, 1);
seq = cell(N_files, 1);

for i = 1:N_files
    File_arr{i} = fullfile(File_names(i).folder, File_names(i).name);
    [audio_vector{i}, Fs{i}] = audioread(File_arr{i});

    sampleRate = Fs{i};
    frameSize = round(30e-3 * sampleRate);  % 30ms block
    overlapSize = round(20e-3 * sampleRate);  % 20ms overlap

    % Initialize audio feature extractor
    audio_FE = audioFeatureExtractor(...
        'SampleRate', sampleRate, ...
        'Window', hamming(frameSize, 'periodic'), ...
        'OverlapLength', overlapSize, ...
        'mfcc', true);

    % Extract MFCC features
    seq{i} = extract(audio_FE, audio_vector{i});
end

% Perform k-means clustering to discretize the MFCC sequences into symbols
numSymbols = 13; % Number of clusters for k-means
[~, symbols] = kmeans(cat(1, seq{:}), numSymbols);

% Discretize each MFCC sequence
discretizedSeqs = cellfun(@(mfccSeq) dsearchn(symbols, mfccSeq), seq, 'UniformOutput', false);

% HMM Initialization
numStates = 8; % Number of states in the HMM
trans = ones(numStates) / numStates; % Uniform transition probabilities
emis = ones(numStates, numSymbols) / numSymbols; % Uniform emission probabilities
HMM = struct('Trans', trans, 'Emission', emis);

% Training loop initialization
maxIterations = 15; % Maximum number of iterations for training
for iteration = 1:maxIterations
    fprintf('Iteration %d\n', iteration);
    
    % Initialize accumulators for re-estimating model parameters
    transAccumulator = zeros(size(HMM.Trans));
    emisAccumulator = zeros(size(HMM.Emission));

    for s = 1:numel(discretizedSeqs)
        obsSeq = discretizedSeqs{s};

        % Run the forward and backward algorithms
        logForwardMatrix = forward_algorithm_log(HMM, obsSeq);
        logBackwardMatrix = backward_algorithm_log(HMM, obsSeq);
        forwardMatrix = exp(logForwardMatrix);
        backwardMatrix = exp(logBackwardMatrix);

        % State occupation probabilities (gamma)
        gamma = (forwardMatrix .* backwardMatrix) ./ sum(forwardMatrix .* backwardMatrix, 1);

        % Calculate transition probabilities (xi)
        xi = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq);

        % Accumulate for transitions and emissions
        for t = 1:(length(obsSeq) - 1)
            transAccumulator = transAccumulator + xi(:, :, t);
        end
        for t = 1:length(obsSeq)
            emisAccumulator(:, obsSeq(t)) = emisAccumulator(:, obsSeq(t)) + gamma(:, t);
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
    numObs = length(obsSeq);
    logForwardProb = -inf(numStates, numObs);
    logForwardProb(:, 1) = log(emis(:, obsSeq(1)));
    for t = 2:numObs
        for j = 1:numStates
            logForwardProb(j, t) = logsumexp(logForwardProb(:, t-1) + trans(:, j)) + emis(j, obsSeq(t));
        end
    end
end

% Define the backward algorithm using log probabilities
function logBackwardProb = backward_algorithm_log(HMM, obsSeq)
    trans = log(HMM.Trans); % Transition probabilities in log space
    emis = log(HMM.Emission); % Emission probabilities in log space

    numStates = size(trans, 1);
    T = length(obsSeq);
    logBackwardProb = -inf(numStates, T);
    logBackwardProb(:, T) = 0; % log(1) at the final step

    for t = (T - 1):-1:1
        for i = 1:numStates
            tempSum = trans(i, :).' + emis(:, obsSeq(t + 1)) + logBackwardProb(:, t + 1);
            logBackwardProb(i, t) = logsumexp(tempSum);
        end
    end
end

% Function to calculate transition probabilities (xi)
function xi = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq)
    numStates = size(HMM.Trans, 1);
    T = length(obsSeq);
    xi = zeros(numStates, numStates, T - 1);
    for t = 1:(T - 1)
        denom = sum(sum(forwardMatrix(:, t) .* HMM.Trans .* (HMM.Emission(:, obsSeq(t+1))' .* backwardMatrix(:, t+1)'), 2));
        for i = 1:numStates
            numer = forwardMatrix(i, t) .* HMM.Trans(i, :) .* (HMM.Emission(:, obsSeq(t+1))' .* backwardMatrix(:, t+1)');
            xi(i, :, t) = numer / denom;
        end
    end
end
