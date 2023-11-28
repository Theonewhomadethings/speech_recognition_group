 %developmentData = load('developmentData.mat');% Load development data
 mfccFeatures = extractMFCC(developmentData); % Extract MFCC features
 viterbiLikelihoods = viterbiAlgorithm(mfccFeatures, trainedHMMs);
 [recognitionOutputs, errorRate] = evaluateRecognizer(viterbiLikelihoods, developmentData.labels);
 [testData, Fs]= audioread('testData.mp3'); % Load test data
 mfccTestFeatures = extractMFCC(testData); % Extract MFCC features from test data
 viterbiTestLikelihoods = viterbiAlgorithm(mfccTestFeatures, trainedHMMs);
 [testRecognitionOutputs, confusionMatrix] = evaluateRecognizer(viterbiTestLikelihoods, testData.labels);
 function mfccFeatures = extractMFCC(audioData) % This function extracts MFCC features from audio data.  
% audioData: A matrix or cell array of audio signals.
 numCoefficients = 13;% Define the number of coefficients to extract   
 mfccFeatures = {}; %Initialize the MFCC features array
 for i = 1:length(audioData)% Process each audio signal
     signal = audioData{i};
     % Compute MFCCs for the current signal        
     coefficients = mfcc(signal, Fs, 'NumCoeffs', numCoefficients);% Fs is the sampling frequency
     mfccFeatures{i} = coefficients; % Store the coefficients    
 end
 end
 function likelihoods = viterbiAlgorithm(features, HMMs) % This function computes the most likely state sequence using the Viterbi algorithm. 
   % features: MFCC features of the audio signals.     % HMMs: Trained Hidden Markov Models.
 likelihoods = zeros(length(features), length(HMMs));% Initialize the likelihoods array   
   % Compute likelihoods for each feature set against each HMM
  for i = 1:length(features)
       for j = 1:length(HMMs)     
           likelihoods(i, j) = viterbi(HMMs{j}, features{i});     
       end
   end
 end
 function [recognitionOutputs, errorRate, confusionMatrix] = evaluateRecognizer(likelihoods, trueLabels)% This function evaluates the recognizer's performance.
% likelihoods: The likelihoods of the observation sequences. % trueLabels: The actual labels of the audio signals.
 numSignals = size(likelihoods, 1);
recognitionOutputs = zeros(numSignals, 1);% Determine the recognized label for each signal
for i = 1:numSignals
    [~, recognizedLabel] = max(likelihoods(i, :));  
    recognitionOutputs(i) = recognizedLabel;
end
  % Calculate the error rate
errors = sum(recognitionOutputs ~= trueLabels);
errorRate = errors / numSignals; 
% Generate the confusion matrix  
  confusionMatrix = confusionmat(trueLabels, recognitionOutputs);
 end

