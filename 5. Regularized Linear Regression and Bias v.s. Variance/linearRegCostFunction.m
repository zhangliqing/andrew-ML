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

h=X*theta; %12*1
tmp=(h-y).^2;
a=sum(tmp(:))/(2*m);
tmp2=theta(2:end).^2;
b=sum(tmp2(:))*lambda/(2*m);
J=a+b;

grad = X'*(h-y)/m;
thetaT = theta;
thetaT(1,1)=0;
grad= grad + (lambda*thetaT)/m ;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
