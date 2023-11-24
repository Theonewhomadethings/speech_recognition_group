% Set the directory containing audio files
Dir = 'C:Users/Pam Shontelle/Documents/MATLAB/Speech2';

% Get a list of all audio files in the directory
audioFiles = dir(fullfile(Dir, '*.mp3'));

% Parameters for MFCC
frameSize = 0.03; % 30 ms frame
hopSize = 0.01;   % 10 ms hop size
numCoeffs = 13;   % 13 MFCC coefficients

% Initialize an array to store MFCCs for all files
mfccs = [];

for i = 1:length(audioFiles)
    % Full path to the audio file
    filePath = fullfile(Dir, audioFiles(i).name);
    % Load the audio file
    [audioIn, fs] = audioread(filePath);
end
% Convert frame and hop size from seconds to samples
frameLength = round(frameSize * fs);
hopLength = round(hopSize * fs);

% Create a tapered window (e.g., Hamming window)
window = hamming(frameLength, 'periodic');

% Compute MFCCs
mfccs = mfcc(audioIn, fs, ...
              'Window', window, ...
              'OverlapLength', frameLength - hopLength, ...
              'NumCoeffs', numCoeffs);
% Display the coefficients
disp(coeffs)

% Compute global mean
globalMean = mean(mfccs);

% Compute global variance
globalVariance = var(mfccs);

% Create a diagonal covariance matrix with the global variance
globalCovariance = diag(globalVariance);

% Set a value for variance floor
varianceFloor = 0.1; % sample value, will chnage as needed

% Alternatively, compute variance floor as a fraction of the max value of
% the global variance
%VFFactor = 0.01; % Example factor, adjust as needed
%varianceFloor = VFFactor * max(globalVariance);

scaledVariance = max(globalVariance, varianceFloor);
scaledCovariance = diag(scaledVariance);



