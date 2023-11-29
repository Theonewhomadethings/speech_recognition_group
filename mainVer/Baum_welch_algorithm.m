Folder_path = "E:\speech_coursework2\EEEM030cw2_DevelopmentSet\";
    "rsity of Surrey\Desktop\SpeechCW2\EEEM030cw2_DevelopmentSet\Dataset";
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

% Initialize an array to store unique labels
unique_labels = {};

% Iterate through each file
for i = 1:numel(File_names)
    % Split the file name by underscores
    Fname_parts = strsplit(File_names(i).name, '_');
    
    % Extract the label from the third part (after the second underscore)
    label = Fname_parts{3};
    
    % Remove the file extension (.mp3)
    label = strrep(label, '.mp3', '');
    
    % Add the label to the array if it's not already present
    if ~ismember(label, unique_labels)
        unique_labels = [unique_labels,label];
    end
end

% Display the unique labels
disp('Unique Labels:');
disp(unique_labels);

Dim = 13;
ncoeffs = size(seq{1},2);
int_mean = zeros(N_files, Dim);
int_var = zeros(N_files, Dim);

% Compute the mean and variance for each file
for i = 1:N_files
    init_mean(i, :) = mean(seq{i})';
    init_var(i, :) = var(seq{i})';
end

% Initialize variables for global mean and variance
g_mean = zeros(Dim, ncoeffs);
g_var = zeros(Dim, ncoeffs);

% Iterate through each dimension
for d = 1:Dim
    % Extract every 13th element for the current dimension
    current_dim_mean = init_mean(d:Dim:end,:);
    current_dim_var = init_var(d:Dim:end,:);

    % Calculate the mean and variance for the current dimension
    g_mean(d:Dim:end) = mean(current_dim_mean);
    g_var(d:Dim:end) = var(current_dim_var);
end

% Display the computed global mean and variance
disp('Global Mean:');
disp(g_mean);

disp('Global Variance:');
disp(g_var); 

% Compute the global covariance matrix
cov_matrix = cov(cat(1, seq{:}));

% Set the off-diagonal values of the covariance matrix to zero
cov_matrix = diag(diag(cov_matrix));

% Use a scaled version of the global variance as a floor
varianceFloor = 0.0001; 
scaled_g_var = max(g_var, varianceFloor);

% Display the computed global mean, variance, and covariance
disp('Global Mean:');
disp(g_mean);

disp('Scaled Global Variance:');
disp(scaled_g_var);

disp('Scaled Global Covariance:');
disp(cov_matrix);

% states = {'say', 'heed', 'hid', 'head', 'had', 'hard', 'hud', 'hod', 'hoard', 'hood', 'whod', 'heard', 'again'};
% unique_state=(cellstr(repmat(states, 1, 30)))';
% % Set the number of symbols
% numSymbols = 13; 
% numStates=8;
% 
% % Set the maximum number of iterations
% maxIterations = 15;
% %unique_sym=unique_labels';
% % Initialize HMM parameters randomly
% initialGuess = 'continuous';
% HMM = hmmestimate(seq,unique_state,numStates,'symbols',states, 'covtype', initialGuess);

trans =[1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13;];
emis = [1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13,1/13,1/13,1/13;];

% Set the number of clusters (symbols) for discretization
numSymbols = 13;

% Perform k-means clustering
[~, symbols] = kmeans(cell2mat(seq), numSymbols);

% Discretize each MFCC sequence
discretizedSeqs = cellfun(@(mfccSeq) dsearchn(symbols, mfccSeq), seq, 'UniformOutput', false);

% Initialize the HMM structure
HMM = struct('Trans', trans, 'Emission', emis);

% Obtain the number of states and symbols from the matrices
numStates = size(trans, 1);
numSymbols = size(emis, 2);

% Set the number of iterations for training
maxIterations = 3;

% Training loop
for iteration = 1:maxIterations
    fprintf('Iteration %d\n', iteration);

    % Iterate over training sequences
    for s = 1:3
        obsSeq = discretizedSeqs{s};

        % Forward algorithm
        forwardMatrix = forward_algorithm(HMM, obsSeq);
        disp(s)
        disp(forwardMatrix)
        % Backward algorithm
        backwardMatrix = backward_algorithm(HMM, obsSeq);
        
        epsilon = 1e-10;
        % State Occupation Probabilities (gamma)
        stateOccupation = (forwardMatrix .* backwardMatrix) ./ (sum(forwardMatrix .* backwardMatrix, 1)+epsilon);
        disp(s)
        disp(stateOccupation)
        % Transition Probabilities (xi)
        transitionProbabilities = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq);

        % Update transition probabilities
        HMM.Trans = sum(transitionProbabilities, 3) ./ sum(stateOccupation(:, 1:end-1), 2);
        % disp(s)
        % disp(HMM.Trans)
        % Update emission probabilities
        for symbol = 1:numSymbols
            % Calculate the numerator (sum of probabilities for each symbol)
            numerator = sum(stateOccupation(:, obsSeq == symbol), 2);

            % Calculate the denominator (sum of state occupation probabilities)
            denominator = sum(stateOccupation, 2);

            % Update emission probabilities for the current symbol
            HMM.Emission(:, symbol) = numerator ./ denominator;
        end
    end

    % Display or save the updated HMM parameters
    disp('Updated Transition Probabilities:');
    disp(HMM.Trans);

    disp('Updated Emission Probabilities:');
    disp(HMM.Emission);
end


%% Notes :  Forward Matrix becoming '0' after 2 iterations through discretizedseq!! Hence the final probablities are NaN. 