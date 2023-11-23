clear all;
close all;

% for timing the script duration for debug purposes, ignore
tic;

% Dataset Pathing 
rootPath = '/user/HS400/as05728/assignment2_speech';
dataset = fullfile(rootPath, 'data');

% Parameters for MFCC extraction
frameLength = 30e-3; % 30 ms
hopSize = 10e-3; % 10 ms

% Fixed words and their corresponding identifiers
words = ["say", "heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "whod", "heard", "again"];
wordIDs = ["w00", "w01", "w02", "w03", "w04", "w05", "w06", "w07", "w08", "w09", "w10", "w11", "w12"];

% Initialize a cell array to store MFCCs for each word
mfccsPerWord = cell(length(words), 1);

%debug code
error = 0;

% Iterate over each speaker and word
for sp = 1:30 
    % Loop through all words in the 'words' array
    for w = 1:length(words) 
        % Construct the filename for the current speaker and word combination
        fileName = sprintf('sp%02d_%s_%s.mp3', sp, wordIDs(w), words(w));
        
        % Create the full path to the audio file by combining the dataset directory with the filename
        fullPath = fullfile(dataset, fileName);

        % error handling
        try  
            % Read the audio file specified by fullPath
            [audioIn, fs] = audioread(fullPath);

            % Compute MFCCs for the audio signal
                % 'WindowLength' sets the length of each frame for MFCC calculation
                % 'OverlapLength' sets the amount of overlap between consecutive frames
                % 'NumCoeffs' sets the number of MFCC coefficients to compute (including the zeroth coefficient)
            coeffs = mfcc(audioIn, fs, ...
                'WindowLength', round(frameLength * fs),...
                'OverlapLength', round(frameLength * fs) - round(hopSize * fs),...
                'NumCoeffs', 13); % Keeping 13 MFCCs
            
            % Append the computed MFCCs vertically to the existing MFCCs for the current word, This creates a matrix where each row is a set of MFCCs for a frame
            mfccsPerWord{w} = [mfccsPerWord{w}; coeffs];

        % error handling
        catch 
            % Print an error message indicating the file could not be found
            fprintf('File not found: %s\n', fullPath);
            
            % Increment the error counter
            error = error + 1;
        end
    end
end

% Compute global statistics (mean and covariance) for each word
meanCovPerWord = struct();
for i = 1:length(words)
    % Retrieve the MFCCs for the current word from the cell array
    wordMfccs = mfccsPerWord{i};

    % Calculate the mean of the MFCCs for the current word
    meanCovPerWord.(words(i)).mean = mean(wordMfccs, 2);

    % Calculate the covariance of the MFCCs for the current word
    meanCovPerWord.(words(i)).cov = cov(wordMfccs');
end


%debug code 
if error > 0
    fprint("MFCCs not extracted fully, there were %d errors", error)
else
    fprintf('MFCCs extracted and statistics computed for %d files\n', length(words) * 30);
end

% Initialize HMM parameters using the mean and covariance

elapsedTime = toc;
fprintf("Total elapsed time: %.2f seconds\n", elapsedTime);
