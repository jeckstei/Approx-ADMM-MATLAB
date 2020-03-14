function [ x, p, conv_log ] = admm_exact_logreg_fg( A, b, c, epsilon, startL, beta )
%   ADMM_EXACT_LOGREG_FG solve the logistic regression with l1 regularization
%   Uses "fast" gradient for subproblem
%   problem:
%                   min sum(log(exp(-b_i(w'*t_i +v)))) + nu||w||_1
%   via ADMM. w is the weight vector and v is the intercept. t_i is the a training example and b_i
%   is the response variable such that b_i \in {-1,1}. nu is the positive
%   scalar. 
%   Let a_i = -b_i * t_i, and A = [a_1 a_2 ... a_m]' be the modified data matrix
%   that absorbs the response vector b, that is A = diag(b) * A_data, where
%   A_data is the original data matrix.
%   Input parameters: 
%         A: modified data matrix.
%         b: response vector.
%         c: postive scalar in ADMM, like the augmented Lagrangian
%         parameter.
%         epsion: overall error tolerance.
%         startL: initial Lipschitz constant estimate
%         beta:   shrinkage constant for stepsize
%   Output:
%         x: primal solution such that x = [v;w]
%         p: dual solution.
%         conv_log: contains objective value, primal residual, dual
%         residual, inner iteration number at each outer iteration.

    t_start = tic;
    
    stepsize = 1/startL;
    
    [nrows,ncols] = size(A);
    % calculate nu, use the rule from Boyd's paper
    ratio = sum(b == 1)/(nrows);
    nu =nrows * 0.1 * 1/nrows * norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');
    % initialization
    max_out_iter = 10000;
    max_inner_iter = 10000;
    total_inner_iter = 0;
    outer_iter = 0;
    % the extra element represents the intercept v
    x = zeros(ncols+1,1);
    y = zeros(ncols+1,1);
    z = zeros(ncols+1,1);
    p = zeros(ncols+1,1);
    
    while outer_iter < max_out_iter
        inner_iter = 0;
        while inner_iter < max_inner_iter
            xold = x;
            [objVal,fVal,gradf,grad] = funcval_logreg(y,z,p,A,b,c,nu);
            inner_iter = inner_iter + 1;
            if norm(grad) < epsilon / 10
                break;
            end
            t = 1;
            while inner_iter < max_inner_iter  % Backtrack loop
                x = y - stepsize*grad;
                [~,newfVal,~,~] = funcval_logreg(x,z,p,A,b,c,nu);
                inner_iter = inner_iter + 1;
                if newfVal <= fVal + dot(grad,x-y) + (1/(2*stepsize))*dot(x-y,x-y)
                    break
                end
                stepsize = beta*stepsize;
            end
            told = t;
            t = (1 + sqrt(1 + 4*t^2))/2;
            y = x + ((told - 1)/t)*(x - xold);
        end
        x = y;
        total_inner_iter = total_inner_iter + inner_iter;
        z_old = z;
        % update z
        z = x + (1/c)*p;
        z(2:end) = shrinkage(z(2:end), nu/c);
        % update multiplier
        p = p + c * (x - z);
        outer_iter = outer_iter + 1;
        % logging the results
        conv_log.inner_iter(outer_iter) = inner_iter;
        conv_log.obj_val(outer_iter) = objVal;
        conv_log.prim_res(outer_iter) = norm(x-z);
        conv_log.dual_res(outer_iter) = norm(z - z_old);
        conv_log.tol_grad(outer_iter) = opttol_logreg(gradf(1),gradf(2:end),nu,x,epsilon);
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol_grad = %10.8f\t total inner: %6.0f\n',outer_iter,conv_log.inner_iter(outer_iter),conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) < 1e-6 
            break;
        end
    end
    toc(t_start);
end