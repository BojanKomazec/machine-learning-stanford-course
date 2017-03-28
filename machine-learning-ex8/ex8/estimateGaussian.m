function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% m - number of exmples
% n - number of features
% X is mxn

for i = 1 : n % iterate across each feature (column)
  sum = 0; % sum of all values of the feature i across all examples
  for j = 1 : m % iterater across each example (row)
    sum = sum + X(j, i);
  endfor
  mu(i) = (1/m) * sum;
endfor

for i = 1 : n % iterate across each feature
  sum = 0; % sum of all squared differences (x_i(j) - mu_i) of the feature i across all examples
  for j = 1 : m % iterater across each example
    sum = sum + (X(j, i) - mu(i))^2;
  endfor
  sigma2(i) = (1/m) * sum;
endfor


% =============================================================


end
