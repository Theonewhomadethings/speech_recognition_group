% Define the backward algorithm using log probabilities
% Backward algorithm for continuous observations

% function beta = backward_algorithm(obsSeq, trans, mu, Sigma)
%     % obsSeq: Observation sequence matrix (each row represents a frame)
%     % trans: Transition probabilities matrix
%     % mu: Mean parameters for Gaussian emission model
%     % Sigma: Covariance matrix parameters for Gaussian emission model
% 
%     N = size(trans, 1);   % Number of states
%     T = size(obsSeq, 1);  % Number of frames
% 
%     % Initialize the backward matrix
%     beta = ones(N, T);
% 
%     % Step 1: Initialization
%     for i = 1:N
%         beta(i, T) = 1;
%     end
% 
%     % Step 2: Recursion
%     for t = T-1:-1:1
%         for i = 1:N
%             beta(i, t) = sum(trans(i, :) .* mvnpdf(obsSeq(t+1, :), mu(i,:), Sigma(:,:,i)) .* beta(:, t+1)');
%         end
%     end
% end#

function logBeta = backward_algorithm(obsSeq, trans, mu, Sigma)
    % obsSeq: Observation sequence matrix (each row represents a frame)
    % trans: Transition probabilities matrix
    % mu: Mean parameters for Gaussian emission model
    % Sigma: Covariance matrix parameters for Gaussian emission model

    N = size(trans, 1);   % Number of states
    T = size(obsSeq, 1);  % Number of frames

    % Initialize the log backward matrix
    logBeta = zeros(N, T);

    % Step 1: Initialization
    for i = 1:N
        logBeta(i, T) = 0;  % Log(1) = 0
    end

    % Step 2: Recursion
    for t = T-1:-1:1
        for i = 1:N
            % Perform element-wise addition in log space using the log-sum-exp trick
            logBeta(i, t) = logsumexp(log(trans(i, :)') + log(mvnpdf(obsSeq(t+1, :), mu(i,:), Sigma(:,:,i))) + logBeta(:, t+1), 1);
        end
    end
end

function result = logsumexp(x, dim)
    % Log-sum-exp trick to prevent numerical instability

    % Find the maximum value along the specified dimension
    max_x = max(x, [], dim);

    % Compute the log-sum-exp
    result = max_x + log(sum(exp(x - max_x), dim));
end
