Folder_path = "E:\speech_coursework2\EEEM030cw2_DevelopmentSet\";
File_names = dir(fullfile(Folder_path, '*.mp3'));
N_files = 390;
File_arr = cell(N_files, 1);
Fs = cell(N_files, 1);
audio_vector = cell(N_files, 1);
mfccCoeffs = cell(N_files, 1);

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
    mfccCoeffs{i} = extract(audio_FE, audio_vector{i});
end
%%
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
%%
Dim = 13;
ncoeffs = size(mfccCoeffs{1},2);
int_mean = zeros(N_files, Dim);
int_var = zeros(N_files, Dim);

% Compute the mean and variance for each file
for i = 1:N_files
    int_mean(i, :) = mean(mfccCoeffs{i})';
    int_var(i, :) = var(mfccCoeffs{i})';
end

% Initialize variables for global mean and variance
g_mean = zeros(Dim, ncoeffs);
g_var = zeros(Dim, ncoeffs);

% Iterate through each dimension
for d = 1:Dim
    % Extract every 13th element for the current dimension
    current_dim_mean = int_mean(d:Dim:end,:);
    current_dim_var = int_var(d:Dim:end,:);

    % Calculate the mean and variance for the current dimension
    g_mean(d:Dim:end) = mean(current_dim_mean);
    g_var(d:Dim:end) = var(current_dim_var);
end

% Display the computed global mean and variance
disp('Global Mean:');
disp(g_mean);

disp('Global Variance:');
disp(g_var); 
%%
% Compute the global covariance matrix
cov_matrix = cov(cat(1, mfccCoeffs{:}));

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
% HMM = hmmestimate(mfccCoeffs,unique_state,numStates,'symbols',states, 'covtype', initialGuess);

trans =[1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;];
emis = [1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;
        1/13,1/13,1/13,1/13,1/13,1/13, 1/13,1/13,1/13,1/13, 1/13,1/13,1/13;];


[seq,states] = hmmgenerate(13,trans,emis);
[estimateTR,estimateE] = hmmestimate(seq,states);

% seq1 = hmmgenerate(100,trans,emis);
% seq2 = hmmgenerate(200,trans,emis);
% seqs = {seq1,seq2};
%HMM = struct('A', [], 'B', [], 'pi', []);
% Training loop
for iteration = 1:maxIterations
    fprintf('Iteration %d\n', iteration);
    
    %Forward-backward algorithm
    [PSTATES,logpseq,FORWARD,BACKWARD] = hmmdecode(seq,trans,emis);
    
   


    % Save the model at the end of each iteration
    save(sprintf('trained_hmm_iteration_%d.mat', iteration), 'HMM');
end
