 % Load development data
 mfccFeatures = extractMFCC(developmentData); % Extract MFCC features
 viterbiLikelihoods = viterbiAlgorithm(mfccFeatures, trainedHMMs);
 [recognitionOutputs, errorRate] = evaluateRecognizer(viterbiLikelihoods, developmentData.labels);
 [testData, Fs]= audioread('testData.mp3'); % Load test data
 mfccTestFeatures = extractMFCC(testData); % Extract MFCC features from test data
 viterbiTestLikelihoods = viterbiAlgorithm(mfccTestFeatures, trainedHMMs);
 [testRecognitionOutputs, confusionMatrix] = evaluateRecognizer(viterbiTestLikelihoods, testData.labels);
 function mfccFeatures = extractMFCC(audioData)   
 numCoefficients = 13;
 mfccFeatures = {};
 for i = 1:length(audioData)
     signal = audioData{i};
     coefficients = mfcc(signal, Fs, 'NumCoeffs', numCoefficients);
     mfccFeatures{i} = coefficients;
 end
 end
 function likelihoods = viterbiAlgorithm(features, HMMs)  
 likelihoods = zeros(length(features), length(HMMs));
   for i = 1:length(features)
       for j = 1:length(HMMs)     
           likelihoods(i, j) = viterbi(HMMs{j}, features{i});     
       end
   end
 end
 function [recognitionOutputs, errorRate, confusionMatrix] = evaluateRecognizer(likelihoods, trueLabels)
numSignals = size(likelihoods, 1);
recognitionOutputs = zeros(numSignals, 1);
for i = 1:numSignals
    [~, recognizedLabel] = max(likelihoods(i, :));  
    recognitionOutputs(i) = recognizedLabel;
end
errors = sum(recognitionOutputs ~= trueLabels);
errorRate = errors / numSignals;  
  confusionMatrix = confusionmat(trueLabels, recognitionOutputs);
 end

