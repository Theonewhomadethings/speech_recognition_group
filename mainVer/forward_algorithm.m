function forwardMatrix = forward_algorithm(HMM, obsSeq)
    numStates = size(HMM.Trans, 1);
    numTimeSteps = length(obsSeq);
    
    forwardMatrix = zeros(numStates, numTimeSteps);
    
    % Initialization
    forwardMatrix(:, 1) = HMM.Emission(:, obsSeq(1)) .* HMM.Trans(:, 1);
    forwardMatrix(:, 1) = forwardMatrix(:, 1) / sum(forwardMatrix(:, 1));
    
    % Recursion
    for t = 2:numTimeSteps
        for j = 1:numStates
            forwardMatrix(j, t) = sum(forwardMatrix(:, t-1) .* HMM.Trans(:, j)) * HMM.Emission(j, obsSeq(t));
           
        end
    end
end
