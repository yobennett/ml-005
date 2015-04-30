function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% minimumError = realmax;

% for cStepIndex = 1:length(steps)
% 	cStep = steps(cStepIndex);

% 	for sigmaStepIndex = 1:length(steps)
% 		sigmaStep = steps(sigmaStepIndex);
% 		disp(printf('try C=%f and sigma=%f', cStep, sigmaStep));

% 		model = svmTrain(X, y, cStep, @(x1, x2) gaussianKernel(x1, x2, sigmaStep));
% 		predictions = svmPredict(model, Xval);
% 		crossValidationError = mean(double(predictions ~= yval));
% 		disp(printf('crossValidationError: %f', crossValidationError));

% 		if crossValidationError < minimumError
% 			disp(printf('%f is smaller than %f', crossValidationError, minimumError));
% 			% found smaller cross-validation error so update C and sigma
% 			% reset crossValidationError
% 			C = cStep;
% 			sigma = sigmaStep;
% 			minimumError = crossValidationError;
% 		end

% 	end

% end

% disp(printf('found C=%f and sigma=%f', C, sigma));

% =========================================================================

end
