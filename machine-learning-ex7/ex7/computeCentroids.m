function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k = 1 : K
  % compute the next location of centroid k
  % compute the mean location of all examples assigned to centroid k
  
  % for each example, check if it's assigned to the current centroid (k)
  % if it is, use it in the summation
  
  % this will be k-th row in centroids matrix
  examplesSum = zeros(1, n); 
  
  % we have to keep track of how many examples were assigned to the current 
  % centroid (k-th centroid)
  examplesCount = 0; 
  
  for i = 1 : m
    if (idx(i) == k)
      examplesSum = examplesSum + X(i, :);
      examplesCount += 1;
    endif
  endfor
  
  % relocate k-th centroid to the mean location of its assigned examples
  centroids(k, :) = (1/examplesCount) * examplesSum;
endfor






% =============================================================


end

