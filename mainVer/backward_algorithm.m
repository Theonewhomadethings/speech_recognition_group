function backwardMatrix = backward_algorithm(HMM, obsSeq)
    numStates = size(HMM.Trans, 1);
    numTimeSteps = length(obsSeq);
    
    backwardMatrix = zeros(numStates, numTimeSteps);
   
    % Initialization
    backwardMatrix(:, end) = 1;
    
    % Recursion
    for t = numTimeSteps-1:-1:1
        for i = 1:numStates
            backwardMatrix(i, t) = sum(HMM.Trans(:, i) .* HMM.Emission(:, obsSeq(t+1)) .* backwardMatrix(:, t+1));

        end
    end
end