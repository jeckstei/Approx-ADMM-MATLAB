function [ A, b, mu, x_true, b_true ] = data_gen_logreg( num_examples, num_features )
%   DATA_GEN_LOGREG Summary of this function goes here
%   Detailed explanation goes here

    rand('seed',0);
    randn('seed', 0);


    w = sprandn(num_features, 1, 0.1);  % N(0,1), 10% sparse
    v = randn(1);            % random intercept

    X = sprandn(num_examples, num_features, 10/num_features);
    b_true = sign(X*w + v);

    % noise is function of problem size use 0.1 for large problem
    b = sign(X*w + v + sqrt(0.1)*randn(num_examples,1)); % labels with noise

    A = spdiags(b, 0, num_examples, num_examples) * X;

    ratio = sum(b == 1)/(num_examples);
    mu = 0.1 * 1/num_examples * norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');

    x_true = [v; w];
end

