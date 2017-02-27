function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

indices_y_Positive = find(y); % equivalent to find(y == 1) 
indices_y_Negative = find(y == 0);

plot(X(indices_y_Positive, 1), X(indices_y_Positive, 2), 
      'k+', 'LineWidth', 2, 'MarkerSize', 10);
      
plot(X(indices_y_Negative, 1), X(indices_y_Negative, 2), 
      'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7); 





% =========================================================================



hold off;

end
