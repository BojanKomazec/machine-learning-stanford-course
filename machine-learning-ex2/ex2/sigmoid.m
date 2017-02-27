function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

dimensions = size(z);
rowsNum = dimensions(1);
columnsNum = dimensions(2);

for rowIndex = 1 : rowsNum
  for columnIndex = 1 : columnsNum
    g(rowIndex, columnIndex) = 1/(1 + e^(-z(rowIndex, columnIndex)));
  end;
end;



% =============================================================

end
