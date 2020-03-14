function [x,iter,conv_log,opt_obj_val] = lasso_fista_bt(A, b, epsilon)
    t_start = tic;
    max_iter = 200000;
    ncols = size(A,2);
    xinit = zeros(ncols,1);
    %% gradient
    % AtA = A'*A;
    Atb = A'*b;
    lambda = 0.1 * norm(Atb, inf);
    % eta > 1
    eta = 10;
    %% objective function 
    function obj_val = calc_F(x, A, b)
        obj_val = 0.5 * sum((A * x - b).^2) + lambda*norm(x,1);
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
    [x, iter, conv_log,opt_obj_val] = backtrack_fista_lasso(@gradf, @shrinkage, @calc_F, xinit,A,b, max_iter, lambda, eta, epsilon);
    toc(t_start);
end 
