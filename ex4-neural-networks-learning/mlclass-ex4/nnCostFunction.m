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



% Part 1
% Feedforward the neural network and return the cost in the variable J

% Theta1 25x401
% Theta2 10x26
% y      5000x1
% X      5000x400

% map y vector of labels containing values from 1..K 
% into a binary vector of 1's and 0's
% that represent the classifications
Y = zeros(m, num_labels); % 5000x10
for i = 1:m
	Y(i, y(i)) = 1;
end

% add ones to the input data matrix for the bias terms
X = [ones(m, 1), X]; % 5000x401

% calculate cost J
for i = 1:m
	% disp(printf("\ni=%d", i));

	% Feedforward pass
	% a1
	a1 = X(i,:);			% 1x401

	% a2
	z2 = Theta1 * a1';		% 25x401 * 401x1 -> 25x1
	a2 = sigmoid(z2);		% 25x1
	a2 = [ones(1,1); a2]; 	% add bias term; 26x1

	% a3
	z3 = Theta2 * a2;		% 10x26 * 26x1 -> 10x1
	a3 = sigmoid(z3);

	hx = a3;
	J += -Y(i,:) * log(hx) - (1 - Y(i,:)) * log(1 - hx);

	% Backpropagation
	% d3
	d3 = a3 - Y(i,:)';		% 10x1

	% d2
	z2 = [ones(1,1); z2]; 	% add bias term; 26x1
	d2 = (Theta2' * d3) .* sigmoidGradient(z2);	% 26x1
	d2 = d2(2:end);			% remove d2_0

	% accumulate the gradient
	Theta1_grad += d2 * a1;		% note: should be 25x401
	Theta2_grad += d3 * a2';	% note: should be 10x26
end

% average cost J
J /= m;

% regularization for Theta1
regTheta1 = 0;
for j = 1:hidden_layer_size
	for k = 2:input_layer_size + 1
		regTheta1 += Theta1(j,k) ^ 2;
	end
end

% regularization for Theta2
regTheta2 = 0;
for j = 1:num_labels
	for k = 2:hidden_layer_size + 1
		regTheta2 += Theta2(j,k) ^ 2;
	end
end

% cost J with regularization
reg_term = (lambda/(2*m)) * (regTheta1 + regTheta2);
J += reg_term;

% unregularized gradient
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% regularized gradient
Theta1_reg = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta1_grad += (lambda/m) * Theta1_reg;
Theta2_reg = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta2_grad += (lambda/m) * Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
