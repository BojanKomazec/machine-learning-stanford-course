function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X contains all inputs from the training set; each example is in one row;
% number of features is equal to number of columns and also equal to input_layer_size
% n = size(X, 2);

% inject 1st column of all "1"s
X = [ones(m, 1) X];
fprintf('\nX: %dx%d (Expected: %dx%d)\n', size(X, 1), size(X, 2), m, input_layer_size + 1);

%z2 = Theta1 * X';
z2 = X * Theta1';
a2 = sigmoid(z2);
% a2: each column contains a2 values for one example from the training set
fprintf('\na2: %dx%d (Expected: %dx%d)\n', size(a2, 1), size(a2, 2), m, hidden_layer_size);

% add ones at the beginning of each column in a2
% a2 = [ones(1, size(a2, 2)); a2];
% inject 1st column of all "1"s
a2 = [ones(m, 1) a2];
fprintf('\na2: %dx%d (Expected: %dx%d)\n', size(a2, 1), size(a2, 2), m, hidden_layer_size + 1);
%fprintf('\na2 after adding ones:');
%a2(:, 1:3)

%z3 = Theta2 * a2;
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% a3: each column contains a3 (= h) values for one example from the training set
fprintf('\na3: %dx%d (Expected: %dx%d)\n', size(a3, 1), size(a3, 2), m, num_labels);


% convert vector y into matrix Y where each row of Y matches 
% i-th example's output value (it's size is: m x num_labels)
fprintf('\ny: %dx%d\n', size(y, 1), size(y, 2));
% m = size(y, 1);
for i = 1 : m % iterate across all training examples
  Y(i, :) = (1 : num_labels) == y(i);
end;
fprintf('\nY: %dx%d (Expected: %dx%d)\n', size(Y, 1), size(Y, 2), m, num_labels);

% debugging
% fprintf('\ny(1):\n');
% y(1)
% fprintf('\nY(1, :):\n');
% Y(1, :)
% fprintf('\ny(2451):\n');
% y(2451)
% fprintf('\nY(2451, :):\n');
% Y(2451, :)

% from these nested for loops we can see what are expected max dimensions for Y 
% and a3 matrices: m x num_labels (size just like Y)

% we have to transpose a3 in order to get desired size
% a3 = a3';

s = 0;
for i = 1 : m % iterate across training set examples
  for k = 1 : num_labels % iterate across output nodes
    s = s + (Y(i, k) * log(a3(i, k)) + (1 - Y(i, k)) * log(1 - a3(i, k)));
  endfor
endfor
   
J = (-1/m) * s;

% -------------------------------------------------------------
% Theta1 is of size: hidden_layer_size x (input_layer_size + 1)
% Theta2 is of size: num_labels x (hidden_layer_size + 1)
% These 1s mean that both Theta matrix contain bias weights

%t1 = sum(Theta1(:, 2:end).^2);
%t2 = sum(Theta2(:, 2:end).^2)
%fprintf('\nt1: %dx%d\n', size(t1, 1), size(t1, 2));
%fprintf('\ny: %dx%d\n', size(y, 1), size(y, 2));
regularization_term = sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2);

J = J + (lambda/(2*m)) * regularization_term;

% =========================================================================

%Delta1 = 0;
%Delta2 = 0;

%for t = 1 : m % iterate over all training examples
  % we've added bias to X earlier
  % a1 is vector of size input_layer_size + 1 x 1
  %a1 = zeros(input_layer_size + 1, 1);
  %a1 = X(i, :)';
  a1 = X;
  
  % delta3 is matrix of size m x num_labels
  % each row matches one training example
  delta3 = a3 - Y;
  
  % Theta2 has size num_labels x (hidden_layer_size + 1)
  % delta3 size = m x num_labels
  % delta2 size iz m x hidden_layer_size => delta3 * Theta2 without 1st column:
  % delta2 = (delta3 * Theta2(:, 2:end)) .* (z2 .* (1 - z2));
  % a2 size: m x (hidden_layer_size + 1) => we have to remove 1st column
  delta2 = (delta3 * Theta2(:, 2:end)) .* (a2(:, 2:end) .* (1 - a2(:, 2:end)));
  
  % a1 size = m x input_layer_size + 1
  % delta2 size = m x hidden_layer_size
  % Delta1 = hidden_layer_size x (input_layer_size + 1)
  % Delta1 = Delta1 + delta2' * a1;
  Delta1 = delta2' * a1;
  
  % a2 size = m x (hidden_layer_size + 1)
  % delta3 size = m x num_labels  
  % Delta2 = num_labels x (hidden_layer_size + 1)
  % Delta2 = Delta2 + delta3' * a2;
  Delta2 = delta3' * a2;
  
%endfor

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1_grad = Theta1_grad + (lambda/m) * Theta1;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
