function [Sequence, logLikelihood, Outputs, errorRate] = HMMRecognizer(all_mfccCoeffs, transitionMatrix, emissionMatrix, entryProbs, Labels)
    
    % Initialize matrices for the Viterbi algorithm
    logDelta = -Inf(totalFrames, N);
   maxstateindex = zeros(totalFrames, N);

    % Compute log of initial state distribution
    logInitialStateDist = log(entryProbs);

    % Initialize with the first observation
    for state = 1:N
        logDelta(1, state) = logInitialStateDist(state) + logEmissionProb(all_mfccCoeffs(1, :), emissionMatrix(state, :));
    end

    % Forward pass to fill in the rest of the logDelta matrix
    for t = 2:totalFrames
        for j = 1:N
            for i = 1:N
                logProb = logDelta(t-1, i) + log(transitionMatrix(i, j));
                if logProb > logDelta(t, j)
                    logDelta(t, j) = logProb;
                    maxstateindex(t, j) = i;
                end
            end
            logDelta(t, j) = logDelta(t, j) + logEmissionProb(all_mfccCoeffs(t, :), emissionMatrix(j, :));
        end
    end

    % Backward pass to find the most likely state sequence
    [~, lastState] = max(logDelta(totalFrames, :));
    Sequence = zeros(totalFrames, 1);
    Sequence(totalFrames) = lastState;

    for t = totalFrames-1:-1:1
        Sequence(t) = maxstateindex(t+1, Sequence(t+1));
    end

    % Compute the log likelihood of the recognized sequence
    logLikelihood = max(logDelta(totalFrames, :));

    % Determine the recognition outputs (most likely states for each time step)
    Outputs = Sequence;

    % Score the results (assuming trueLabels is a vector of the true state indices)
    % and calculate the error rate
    numErrors = sum(Outputs ~= Labels);
    errorRate = numErrors / totalFrames;
end