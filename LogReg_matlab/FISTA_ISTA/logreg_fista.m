function [x,iter,conv_log,opt_obj_val] = logreg_fista(A, b, epsilon)
    t_start = tic;
    max_iter = 100000;
    [nrows,ncols] = size(A);
    % calculate nu, use the rule from Boyd's paper
    ratio = sum(b == 1)/(nrows);
    lambda =nrows * 0.1 * 1/nrows * norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');
    xinit = zeros(ncols+1,1);
    % eta > 1
    eta = 1.25;
    %% cost f
    function loss = calc_f(x,A,b)
        aux = exp(-b*x(1)-A*x(2:end));
        loss = sum(log(1 + aux));
    end 
    %% cost function 
    function obj_val = calc_F(x,A,b)
        obj_val = calc_f(x,A,b) + lambda*norm(x(2:end),1);
    end
    %% shrinkage operator
    function [ z ] = shrinkage( x, kappa )
    %   z = sign(x) .* max(0, abs(x) - (kappa) * ones(ncols,1));    
        z = max(0, x - kappa) - max(0, -x-kappa);
    end
    %% Use fista with backtrack line search procedure 
    [x, iter, conv_log,opt_obj_val] = backtrack_fista(@shrinkage, @calc_F, xinit,A,b,max_iter, lambda, eta, epsilon);
    toc(t_start);
end 