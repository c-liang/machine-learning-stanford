function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


C_list = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

best_result = 1;
x1 = X(:,1)';
x2 = X(:,2)';



for i = 1:length(C_list)
   for j = 1:length(sigma_list)
       
        model= svmTrain(X, y, C_list(:,i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(:,j))); 

        predictions = svmPredict(model, Xval);
        
        if best_result > mean(double(predictions ~= yval))
            best_result = mean(double(predictions ~= yval))
            best_C = C_list(:,i);
            best_sigma = sigma_list(:,j);
        else
            % do nothing.
        end
   end

end

C = best_C;
sigma = best_sigma;




% =========================================================================

end
