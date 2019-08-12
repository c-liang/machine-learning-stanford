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



n = size(X, 2); % number of features

hx = 1 ./ (1 + exp(-(theta' * X')));

theta_without_first = theta;	% Do not regulate the first theta i.e. theta(0) 
theta_without_first(1) = [];

J = sum(-y .* log(hx') - (1 - y) .* log(1 - hx')) / m + (lambda/ (2 * m)) * sum(theta_without_first.^2);

grad(1,:) = (hx' - y)' * X(:,1) / m;

for indx = 2:n
    grad(indx,:) = (hx' - y)' * X(:,indx) / m + (lambda / m) * theta(indx);
end


% =============================================================

end
