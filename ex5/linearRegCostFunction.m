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

n = size(X, 2); % number of features

hx = X * theta;

theta_without_first = theta(2:end);	% Do not regulate the first theta i.e. theta(0) 

J = sum((hx - y) .^ 2) ./ (2 * m) + (lambda/ (2 * m)) * sum(theta_without_first.^2);

grad(1,:) = (hx - y)' * X(:,1) / m

for indx = 2:n
    grad(indx,:) = (hx - y)' * X(:,indx) / m + (lambda / m) * theta(indx);
end


% =========================================================================

grad = grad(:);

end
