rootPath = "C:\Users\Abdullah\Desktop\speech_audio_pr\GROUP";
dataset = fullfile(rootPath, 'data');
File_names = dir(fullfile(dataset, '*.mp3'));
N_files = length(File_names);
File_arr = cell(N_files, 1);
Fs = cell(N_files, 1);
audio_vector = cell(N_files, 1);
seq = cell(N_files, 1);
numCoeffs=13;
for i = 1:N_files
    File_arr{i} = fullfile(File_names(i).folder, File_names(i).name);
    [audio_vector{i}, Fs{i}] = audioread(File_arr{i});

    sampleRate = Fs{i};
    frameSize = round(30e-3 * sampleRate);  % 30ms block
    overlapSize = round(20e-3 * sampleRate);  % 20ms overlap
    hopSize = frameSize - overlapSize;        %10ms hop

    % Initialize audio feature extractor
    audio_FE = audioFeatureExtractor(...
        'SampleRate', sampleRate, ...
        'Window', hamming(frameSize, 'periodic'), ...
        'OverlapLength', overlapSize, ...
        'mfcc', true, ...  
        'mfccDelta', false, ...
        'mfccDeltaDelta', false);

    % Extract MFCC features
    seq{i} = extract(audio_FE, audio_vector{i});
end
% Initialize variables
numClasses = 30;
N=8;
Dim=13;
% Create variables to store feature vectors for each class
classFeatureVectors = cell(1, numClasses);
unique_labels = {}; % Initialize an array to store unique labels
ClassNames = {'say', 'heed', 'hid','head' , 'had','hard','hud','hod','hoard','hood','whod','heard','again'};
% Iterate through each file to organize features by class and collect unique labels
for i = 1:numel(File_names)
    currentFeatures = seq{i};
    % Extract label and class index
    Fname_parts = strsplit(File_names(i).name, '_');
    label = strrep(Fname_parts{3}, '.mp3', '');
    classIndex = find(strcmp(label, ClassNames));

    % Store the feature vector and update unique labels
    if ~isempty(classIndex) && classIndex <= numClasses
        classFeatureVectors{classIndex} = [classFeatureVectors{classIndex}; currentFeatures];
    else
        disp(['Invalid label: ', label]);
    end
    if ~ismember(label, unique_labels)
        unique_labels = [unique_labels, label];
    end
end
disp('Unique Labels:');
disp(unique_labels);

% Concatenate all MFCC features from different files into one matrix
all_mfccCoeffs = vertcat(seq{:});

% Compute the global mean and variance
g_mean = mean(all_mfccCoeffs); % Global mean of MFCCs
g_var = var(all_mfccCoeffs);   % Global variance of MFCCs

% Apply variance floor and scaling
varianceFloor = 0.0001; % Adjust as needed
scaled_g_var = max(g_var, varianceFloor); % Apply variance floor

% Compute the global covariance matrix
cov_matrix = cov(all_mfccCoeffs);
cov_matrix = diag(diag(cov_matrix)); % Convert to diagonal matrix

% Calculate average duration per state
totalFrames = sum(cellfun(@(c) size(c, 1), seq)); % Total number of frames across all files
avgDurationPerState = totalFrames / (N_files * N); % Average frames per state

% State transition probabilities
selfLoopProb = exp(-1 / (avgDurationPerState - 1)); % Self-loop probability
nextStateProb = 1 - selfLoopProb; % Probability of moving to the next state

% Transition matrix for HMMs
trans = diag(repmat(selfLoopProb, N, 1)) + diag(repmat(nextStateProb, N-1, 1), 1);

% Emission probabilities for each state
emissionMeans = repmat(g_mean, N, 1); % Replicate global mean for each state
emissionCovariances = repmat(diag(scaled_g_var), 1, 1, N); % Replicate diagonal covariance matrix for each state

% Set up emission probabilities as a structure array
emis(N) = struct('Mean', [], 'Covariance', []);
for i = 1:N
    emis(i).Mean = emissionMeans(i, :); 
    emis(i).Covariance = emissionCovariances(:, :, i);
end

% Display the computed global mean, variance, and covariance
disp('Global Mean:');
disp(g_mean);
disp('Scaled Global Variance:');
disp(scaled_g_var);
disp('Scaled Global Covariance:');
disp(cov_matrix);

% Initialize HMM models for each class
for classIndex = 1:numClasses
    HMMs{classIndex} = struct('Trans', trans, 'EmissionMean', emissionMeans, 'EmissionCovariance', emissionCovariances);
end
gamma = zeros(N, 1);
% Number of classes/words
numClasses = 13;
maxIterations = 3;
% Cell array to store HMM models for each class
HMMs = cell(1, numClasses);
% Initialize accumulators for re-estimating model parameters
transAccumulator = zeros(N, N);
emisAccumulator = struct('mu', zeros(N, numCoeffs), 'Sigma', zeros(numCoeffs, numCoeffs, N), 'weight', zeros(N, 1));

% Training loop
for classIndex = 1:numClasses
    fprintf('Training HMM for class %d\n', classIndex);

    % Assuming classFeatureVectors is a cell array
    obsSeq = classFeatureVectors{classIndex};

    % Reset accumulators for each class
    transAccumulator(:) = 0;
    emisAccumulator.mu(:) = 0;
    emisAccumulator.Sigma(:) = 0;
    emisAccumulator.weight(:) = 0;

    % Initialize the HMM parameters for this class
    HMM = struct('Trans', trans, 'Emission', emis, 'updated_Trans', transAccumulator, 'updated_emis', emisAccumulator);  

    % Train the HMM model
    for iteration = 1:maxIterations
        fprintf('Iteration %d\n', iteration);

        % Forward algorithm (O, pi, A, mu, Sigma)
        logAlpha = forward_algorithm(obsSeq,gamma,HMM.Trans,repmat(g_mean, 8, 1), emissionCovariances);

        % Backward algorithm
        logBeta = backward_algorithm(obsSeq,HMM.Trans,repmat(g_mean, 8, 1), emissionCovariances);

       % Calculate transition probabilities and occupation likelihoods
        [logtransProb, logoccupationLikelihood] = calculate_transition_and_occupation(HMM.Trans,repmat(g_mean, 8, 1), emissionCovariances,logAlpha,logBeta, obsSeq);

       % Display results
        %disp('Transition Probabilities:');
        % disp(logtransProb);
        % 
        % disp('Occupation Likelihoods:');
        % disp(logoccupationLikelihood);

       % Accumulate for transitions
        for t = 1:(length(obsSeq) - 1)
            transAccumulator = transAccumulator + logtransProb(:, :, t);
        end
        % Implementing the floor for transition probabilities
        %logFloor = log(1e-10);  % A small positive value to prevent -Inf
        %transAccumulator = max(transAccumulator, logFloor);

        for t = 1:length(obsSeq)
            % Accumulate for mean (mu)
            emisAccumulator.mu = emisAccumulator.mu + logoccupationLikelihood(:, t) * obsSeq(t, :);

            % Accumulate for covariance (Sigma)
            for i = 1:N
                diff = obsSeq(t, :) - emisAccumulator.mu(i, :);
                emisAccumulator.Sigma(:, :, i) = emisAccumulator.Sigma(:, :, i) + logoccupationLikelihood(i, t) * (diff' * diff);
            end
            % Accumulate for weight (optional, depending on your model)
            emisAccumulator.weight = emisAccumulator.weight + logoccupationLikelihood(:, t);
            
            % Normalize weight
            emisAccumulator.weight = emisAccumulator.weight / sum(emisAccumulator.weight);
        end
        for i = 1:N
            emisAccumulator.mu(i, :) = emisAccumulator.mu(i, :) / emisAccumulator.weight(i);
            emisAccumulator.Sigma(:, :, i) = emisAccumulator.Sigma(:, :, i) / emisAccumulator.weight(i);
        end
    end

    updated_HMM = struct('updated_Trans', transAccumulator,'updated_emis', emisAccumulator); 

    % Save the trained HMM for this class
    updated_HMMs{classIndex} =updated_HMM;

    % Display or save the final HMM parameters for this class
    disp(['Final HMM parameters for class ' num2str(classIndex) ':']);
    disp('Final Transition Probabilities:');
    disp(updated_HMMs{classIndex}.updated_Trans);

    disp('Final Emission Probabilities:');
    disp(updated_HMMs{classIndex}.updated_emis);
end                      

%{
 

Test_Folderpath = "C:\Users\ms03549\OneDrive - University of Surrey\Desktop\SpeechCW2\EEEM030cw2_DevelopmentSet\testdata";
Test_Filenames = dir(fullfile(Test_Folderpath, '*.m4a'));
Testfiles = length(Test_Filenames);
Test_Filesarr = cell(Testfiles, 1);
TestFs = cell(Testfiles, 1);
Test_audioVector = cell(Testfiles, 1);
Testseq = cell(Testfiles, 1);

for i = 1:Testfiles
    Test_Filesarr{i} = fullfile(Test_Filenames(i).folder, Test_Filenames(i).name);
    [Test_audioVector{i}, TestFs{i}] = audioread(Test_Filesarr{i});

    TestsampleRate = TestFs{i};
    frameSize = round(30e-3 * TestsampleRate);  % 30ms block
    overlapSize = round(20e-3 * TestsampleRate);  % 20ms overlap
    hopSize = frameSize - overlapSize;        %10ms hop

    % Initialize audio feature extractor
    Test_audioFE = audioFeatureExtractor(...
        'SampleRate', TestsampleRate, ...
        'Window', hamming(frameSize, 'periodic'), ...
        'OverlapLength', overlapSize, ...
        'mfcc', true, ...  
        'mfccDelta', false, ...
        'mfccDeltaDelta', false);

    % Extract MFCC features
    Testseq{i} = extract(Test_audioFE, Test_audioVector{i});
end

%most_likely_sequence = zeros(1, length(Testseq));

%[most_likely_sequence,max_cum_log_likelihood] = viterbi_algorithm(Testseq , updated_HMMs{1}, N); 
%}
