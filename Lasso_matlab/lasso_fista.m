function [x,iter,conv_log,opt_obj_val] = lasso_fista(A, b, epsilon)
    t_start = tic;
    max_iter = 150000;
    ncols = size(A,2);
    xinit = zeros(ncols,1);
    %% gradient
    % AtA = A'*A;
    Atb = A'*b;
    lambda = 0.1 * norm(Atb, inf);
    %% Lipschitz constant 
    L = svds(A,1)^2;
    fprintf('L= %8d\n', L);
    %% cost f
    function loss = calc_f(x)
        loss = 0.5 * sum((A * x - b).^2);
    end 
    %% objective function 
    function obj_val = calc_F(x)
        obj_val = calc_f(x) + lambda*norm(x,1);
    end 
    function res = gradf(x)
	res = A*x;
	res = A' * res; 
        res = res - Atb;
    end
    %% shrinkage operator
    function [ z ] = shrinkage( x, kappa )
    %   z = sign(x) .* max(0, abs(x) - (kappa) * ones(ncols,1));    
        z = max(0, x - kappa) - max(0, -x-kappa);
    end
    %% Use fista 
    [x, iter, conv_log,opt_obj_val] = fista(@gradf, @shrinkage, @calc_F, xinit, L, max_iter, lambda, epsilon);
    toc(t_start);
end 
