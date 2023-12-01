
% function alpha = forward_algorithm(O, pi, A, mu, Sigma)
%     % O: observation sequence matrix (each row represents a frame)
%     % pi: initial state probabilities
%     % A: state transition probabilities
%     % mu: mean parameters for Gaussian emission model
%     % Sigma: covariance matrix parameters for Gaussian emission model
% 
%     N = size(A, 1);   % Number of states
%     T = size(O, 1);   % Number of frames
% 
%     % Initialize the forward likelihood matrix
%     alpha = zeros(N, T);
% 
%     % Step 1: Initialization
%     alpha(:, 1) = pi .* mvnpdf(O(1, :), mu(1,:), Sigma(:,:,1))';
% 
%     % Step 2: Recursion
%     for t = 2:T
%         for j = 1:N
%             % Perform element-wise multiplication and sum
%             alpha(j, t) = sum(alpha(:, t-1)' .* A(:, j) .* mvnpdf(O(t, :), mu(j, :), Sigma(:,:,j))',"all");
%         end
%     end
% end

function logAlpha = forward_algorithm(O, logPi, logA, mu, Sigma)
    % O: observation sequence matrix (each row represents a frame)
    % logPi: log initial state probabilities
    % logA: log state transition probabilities
    % mu: mean parameters for Gaussian emission model
    % Sigma: covariance matrix parameters for Gaussian emission model

    N = size(logA, 1);   % Number of states
    T = size(O, 1);   % Number of frames

    % Initialize the log forward likelihood matrix
    logAlpha = zeros(N, T);

    % Step 1: Initialization
    logAlpha(:, 1) = logPi + log(mvnpdf(O(1, :), mu(1,:), Sigma(:,:,1)))';

    % Step 2: Recursion
    for t = 2:T
        for j = 1:N
            % Perform element-wise addition in log space using the log-sum-exp trick
            logAlpha(j, t) = logsumexp(logAlpha(:, t-1) + logA(:, j) + log(mvnpdf(O(t, :), mu(j, :), Sigma(:,:,j))),1);
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

