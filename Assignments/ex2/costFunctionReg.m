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


%arrange data 
theta2_n = theta(2:end);
hypo = sigmoid( X * theta);

% reg = lambda * (theta* theta') / 2;
regularized = (lambda/( 2* m)) * (theta2_n' * theta2_n);

%for y = 1 term
J1 = -y .* log(hypo);
%for y = 0 term
J0 = (1-y) .* log(1 - hypo);

J = (1/m) * sum(J1 - J0) + regularized;

grad0 = (X(:, 1)' * (hypo - y )) / m; %taking only the first column of X which is X0 and compute gradient without regularization
gradn = (X' * (hypo - y) + (lambda * theta))  / m;  

grad = [grad0; gradn(2:end)];


% =============================================================

end
