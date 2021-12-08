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

l_input_act = X;
l_input_act_with_bias=[ones(m, 1) l_input_act];  % 5000 * 401

l_2_act = sigmoid(Theta1 * l_input_act_with_bias');  % size 25 * 5000
l_2_act_with_bias = [ones(1, m); l_2_act]; % size 26 * 5000

l_output_act = sigmoid(Theta2 * l_2_act_with_bias); % size 10 * 5000


% cast y values to vectors
y_vectors = zeros(num_labels, m); % size 10 * 5000
for i= 1: m
  y_vectors(y(i), i) = 1;
end


for i = 1: m

  %for y = 1 term
  J1 = sum(-y_vectors(:, i) .* log(l_output_act(:, i)));
  %for y = 0 term
  J0 = sum((1 .- y_vectors(:, i)) .* log(1 .- l_output_act(:, i)));

  J  += sum(J1 - J0);
end


%remove bias column
Theta1_without_bias = Theta1(:, 2:end)(:); % size 25 x 400
Theta2_without_bias = Theta2(:, 2:end)(:); % size 10 x 25

regularization = (lambda / 2) * ((Theta1_without_bias' * Theta1_without_bias) + (Theta2_without_bias' * Theta2_without_bias));


J = (J + regularization) / m ;


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


% this without bias
d_act1 = zeros(size(Theta1));   %  25 x 401
d_act2 = zeros(size(Theta2));   %  10 x 26                 


for i = 1: m
  d_output = l_output_act(:, i) - y_vectors(:, i); % size 10 x 1
                       %  10 x 1       1 * 26  
  d_act2 = d_act2 .+ (d_output * l_2_act_with_bias(:, i)');
              % 25 x 10            10 x 1            25 * 1            25 * 1
  d_layer2 = (Theta2(: , 2:end)' * d_output)  .*   (l_2_act(:, i) .* (1 - l_2_act(:, i)));
                      %  25 x 1       1 * 401
  d_act1 = d_act1 + (d_layer2 * l_input_act_with_bias(i, :));

end

Theta1_grad(:, 1) = d_act1(: , 1) / m;
Theta1_grad(:, 2:end) = (d_act1(: , 2:end)  / m) + ((lambda / m) * Theta1(:, 2:end));

Theta2_grad(:, 1) = d_act2 (: , 1) / m;
Theta2_grad(:, 2:end) = (d_act2(: , 2:end)  / m)  + ((lambda / m)  * Theta2(:, 2:end)); 



% fprintf(['size of Theta1 grad %f x %f \n'], size(Theta1_grad))
% fprintf(['size of Theta2 grad %f x %f \n'], size(Theta2_grad))
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
