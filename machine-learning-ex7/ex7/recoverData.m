function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               



% 1st valid implementation - with loops:

%{
% number of examples
m = size(Z, 1);

% orignal number of features
n = size(U, 1);

% X_rec shall be mxn

% for each example in Z calculate its projection back 
for i = 1 : m
  % Z(i, :) is 1xK => v is Kx1
  v = Z(i, :)';
  
  %for each dimension in U
  for j = 1 : n
    % v' is 1xK, U(j, 1:K) is nxK => U(j, 1:K)' is Kx1
    % recovered_j is 1x1
    recovered_j = v' * U(j, 1:K)';
    X_rec(i, j) = recovered_j;
  endfor
  
endfor;
%}

%2nd valid implementation - with vectorization:

% Z is mxK
% U is nxn; but we were using first K columns as U_reduce: nxK
U_reduce = U(:, 1:K); 

% X_rec has to be mxn = mxK * Kxn =>
X_rec = Z * U_reduce';


% =============================================================

end
