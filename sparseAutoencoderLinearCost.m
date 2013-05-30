function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (yoW2gradur code needs to compute these values). 
% Here, we initialize them to zeros. 

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

num_samples = size(data,2);
a1 = data;
z2 = bsxfun(@plus, W1*a1, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);
a3 = z3;

rho_hat = mean(a2,2);
rho = sparsityParam;
KL = rho*log(rho./rho_hat) + (1-rho)*log((1-rho)./(1-rho_hat));


d3 = -(a1 - a3);
d2 = bsxfun(@plus, W2'*d3 , beta*(-rho./rho_hat + (1-rho)./(1-rho_hat))) .*a2.*(1-a2);

W1grad = d2*a1' /num_samples + lambda* W1;
W2grad = d3*a2'/num_samples + lambda* W2;
b1grad = mean(d2,2);
b2grad = mean(d3,2);

square = (a3-a1).^2;

cost = 1/2*sum(square(:))/num_samples + lambda/2*sum((W1(:).^2 + W2(:).^2)) + beta*sum(KL);


grad = [ W1grad(:); W2grad(:); b1grad(:); b2grad(:)];


end
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end


