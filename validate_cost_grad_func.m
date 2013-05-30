visibleSize = 8;
hiddenSize = 5;
sparsityParam = 0.035;
lambda = 3e-3;
beta = 5;
epsilon =0.1;

patches = rand([8,10]);
theta = initializeParameters(hiddenSize, visibleSize);
[~,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches);

numGrad  = computeNumericalGradient(@(x)sparseAutoencoderLinearCost(x, visibleSize, hiddenSize, lambda, ...
                                                                    sparsityParam, beta, patches) , theta);
disp([numGrad ,grad]);
diff = norm(numGrad - grad )/norm(numGrad + grad);
disp(diff);
assert(diff<1e-9, 'difference too large, check your code');
