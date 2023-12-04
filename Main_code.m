Folder_path = "C:\Users\ms03549\OneDrive - University of Surrey\Desktop\SpeechCW2\EEEM030cw2_DevelopmentSet\Dataset";
File_names = dir(fullfile(Folder_path, '*.mp3'));
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

% Create variables to store feature vectors for each class
classFeatureVectors = cell(1, numClasses);
% Initialize an array to store unique labels
unique_labels = {};
ClassNames = {'say', 'heed', 'hid','head' , 'had','hard','hud','hod','hoard','hood','whod','heard','again'};
% Iterate through each file
for i = 1:numel(File_names)
    currentFeatures=seq{i};
    % Split the file name by underscores
    Fname_parts = strsplit(File_names(i).name, '_');
    
    % Extract the label from the third part (after the second underscore)
    label = Fname_parts{3};
    
    % Remove the file extension (.mp3)
    label = strrep(label, '.mp3', '');

    % Find the class index based on the label
    classIndex = find(strcmp(label, ClassNames));

     % Check if the class index is valid
    if ~isempty(classIndex) && classIndex <= numClasses
        % Store the feature vector in the corresponding class variable
        classFeatureVectors{classIndex} = [classFeatureVectors{classIndex}; currentFeatures];
    else
        % Handle cases where the label does not match any class or is invalid
        disp(['Invalid label: ', currentLabel]);
    end
    
    % Add the label to the array if it's not already present
    if ~ismember(label, unique_labels)
        unique_labels = [unique_labels,label];
    end
end

% Display the unique labels
disp('Unique Labels:');
disp(unique_labels);
N=8;
Dim=13;
ncoeffs = size(seq{1},2);
int_mean = zeros(N_files, Dim);
int_var = zeros(N_files, Dim);

% Compute the mean and variance for each file
for i = 1:N_files
    init_mean(i, :) = mean(seq{i})';
    init_var(i, :) = var(seq{i})';
end

% Initialize variables for global mean and variance
num=length(File_names);
g_mean = zeros(1,ncoeffs);
g_var = zeros(ncoeffs, ncoeffs);
e_mean=zeros(num,ncoeffs);
e_var=zeros(num,ncoeffs);
Dim=13;
arr=0;

% Iterate through each dimension,,/
for d = 1:numel(File_names)
     for e=1:size(seq{d},1)
         
             e_mean(d,:)= mean(seq{e});
                        
     end  
    arr=arr+size(seq{d},1);  
end 
frame_avg=arr/length(File_arr);
g_mean = mean(e_mean);

% Iterate through each dimension
for d = 1:Dim
    % Extract every 13th element for the current dimension
    
    current_dim_var = init_var(d:Dim:end,:);

    % Calculate the variance for the current dimension
    
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

X = [mvnrnd(g_mean,cov_matrix,390)];
GMModel = fitgmdist(X,1);

emissionCovariances = repmat(diag(diag(scaled_g_var)), 1, 1, N);
% Display the computed global mean, variance, and covariance
disp('Global Mean:');
disp(g_mean);

disp('Scaled Global Variance:');
disp(scaled_g_var);

disp('Scaled Global Covariance:');
disp(cov_matrix);

% Calculate average duration per state
totalFrames = sum(cellfun(@(c) size(c, 1), seq)); % Total number of frames across all files
avgDurationPerState = totalFrames / (num * 8); % Average frames per state

% State transition probabilities
selfLoopProb = exp(-1 / (avgDurationPerState - 1)); % Self-loop probability
nextStateProb = 1 - selfLoopProb; % Probability of moving to the next state

% Set up emission probabilities as a structure array
emis(N) = struct('Mean', [], 'Covariance', []);

% Fill the structure array for each state
for i = 1:N
    emis(i).Mean = repmat(g_mean, N, 1); 
    emis(i).Covariance =repmat(cov_matrix, [1, 1, 8]);
    % repmat(diag(diag(cov_matrix)), 1, 1, N);
end

   % emis = repmat((1/48),13,8);
 
     trans = [
       0.8, 0.2, 0, 0, 0, 0, 0, 0;
       0, 0.8, 0.2, 0, 0, 0, 0, 0;
       0, 0, 0.8, 0.2, 0, 0, 0, 0;
       0, 0, 0, 0.8, 0.2, 0, 0, 0;
       0, 0, 0, 0, 0.8, 0.2, 0, 0;
       0, 0, 0, 0, 0, 0.8, 0.2, 0;
       0, 0, 0, 0, 0, 0, 0.8, 0.2;
       0, 0, 0, 0, 0, 0, 0, 0.8];
  
  %{ 
    trans = [
       0, 1,  0, 0, 0, 0, 0, 0, 0,0;
       0,0.8, 0.2, 0, 0, 0, 0, 0, 0,0;
       0,0, 0.8, 0.2, 0, 0, 0, 0, 0,0;
       0,0, 0, 0.8, 0.2, 0, 0, 0, 0,0;
       0,0, 0, 0, 0.8, 0.2, 0, 0, 0,0;
       0,0, 0, 0, 0, 0.8, 0.2, 0, 0,0;
       0,0, 0, 0, 0, 0, 0.8, 0.2, 0,0;
       0,0, 0, 0, 0, 0, 0, 0.8, 0.2,0;
       0,0, 0, 0, 0, 0, 0, 0, 0.8,0.2;
       0,0, 0, 0, 0, 0, 0, 0, 0,0];
%}
gamma = zeros(N, 1);

% 
% Obtain the number of states and symbols from the matrices
numStates = size(trans, 1);
numSymbols = size(emis, 2);

% Training loop initialization
maxIterations = 3; % Maximum number of iterations for training

% Initialize accumulators for re-estimating model parameters
transAccumulator = zeros(N,N);
% Initialize emisAccumulator with means and covariances
emisAccumulator = struct('mu', zeros(N, numCoeffs), 'Sigma', zeros(numCoeffs, numCoeffs, N));
% Assuming N is the number of states
emisAccumulator.weight = zeros(N, 1);

% Number of classes/words
numClasses = 13;

% Cell array to store HMM models for each class
HMMs = cell(1, numClasses);

% Training loop
for classIndex = 1:numClasses
    fprintf('Training HMM for class %d\n', classIndex);

    % Assuming classFeatureVectors is a cell array
    obsSeq = classFeatureVectors{classIndex};

    % Initialize the HMM parameters for this class
    HMM = struct('Trans', trans, 'Emission', emis,'updated_Trans', transAccumulator,'updated_emis', emisAccumulator);  
    
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

most_likely_sequence = zeros(1, length(Testseq));

[most_likely_sequence,max_cum_log_likelihood] = viterbi_algorithm(Testseq , updated_HMMs{1}, N);
