function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tempSum = 0;
for i = 1 : m
  h = sigmoid(X(i, :) * theta);
  %printf('i = %f, h(theta, x(i)) = %f \n', i, h);
  tempSum = tempSum + (y(i) * log(h) + (1 - y(i)) * log(1 - h));
end;

sumThetaSquares = 0;
for j = 2 : size(theta)
  sumThetaSquares = sumThetaSquares + theta(j)^2;
end;
  
J = (-1/m) * tempSum + (lambda/(2*m)) * sumThetaSquares;

sigmoids = zeros(1, m);
for i = 1 : m
  sigmoids(i) = sigmoid(X(i, :) * theta);
end;

grad(1) = (1/m)*((sigmoids - y') * X(:, 1));

for j = 2 : size(theta)
  grad(j) = (1/m) * ((sigmoids - y') * X(:, j)) + (lambda/m) * theta(j);
end;




% =============================================================

end
