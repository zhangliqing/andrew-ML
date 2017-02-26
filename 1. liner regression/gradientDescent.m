function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration 
    tmp=0;
    theta1=theta;
    for i=1:m
        tmp=tmp+(X(i,:)*theta1-y(i))*X(i,1);
    end
    tmp=tmp*alpha/m;
    theta(1)=theta(1)-tmp;
    
    tmp=0;
    for j=1:m
        tmp=tmp+(X(j,:)*theta1-y(j))*X(j,2);
    end
    tmp=tmp*alpha/m;
    theta(2)=theta(2)-tmp;
    
    J_history(iter) = computeCost(X, y, theta);

end

end
