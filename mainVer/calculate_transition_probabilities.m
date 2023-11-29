function transitionProbabilities = calculate_transition_probabilities(HMM, forwardMatrix, backwardMatrix, obsSeq)
    numStates = size(HMM.Trans, 1);
    numTimeSteps = length(obsSeq);
    
    transitionProbabilities = zeros(numStates, numStates, numTimeSteps - 1);
    
    for t = 1:numTimeSteps - 1
        for i = 1:numStates
            for j = 1:numStates
                transitionProbabilities(i, j, t) = forwardMatrix(i, t) * HMM.Trans(i, j) * ...
                    HMM.Emission(j, obsSeq(t + 1)) * backwardMatrix(j, t + 1);
            end
        end
    end
end
