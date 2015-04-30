function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% disp(size(X));
% disp(size(theta));

% X 		12x2
% theta		2x1
% hx		12x1

hx = X * theta;

temp_theta = theta;
temp_theta(1) = 0;

reg_term = (lambda/(2*m)) * temp_theta' * temp_theta;
J = (1/(2*m)) * sum((hx - y) .^ 2) + reg_term;

reg_term = (lambda/m) * temp_theta;
grad = ((1/m) * (X' * (hx - y))) + reg_term;

% =========================================================================

grad = grad(:);

end
