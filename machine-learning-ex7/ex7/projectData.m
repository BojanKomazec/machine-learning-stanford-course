function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%


% 1st valid implementation - with loops:

%{
m = size(X, 1)
% project each of m examples onto each of K eigenvectors
for i = 1 : m
  %X(i,:) is 1xn so x is nx1  
  x = X(i, :)';
  for k = 1 : K
    % x' is 1xn, U(:, k) is nx1
    projection_k = x' * U(:, k);
    Z(i, k) = projection_k;
  endfor
endfor
%}

% 2nd valid implementation - vectorized:

% nxK
U_reduce = U(:, 1:K);
% mxn * nxK = mxK (m examples with K features)
Z = X * U_reduce;



% =============================================================

end
