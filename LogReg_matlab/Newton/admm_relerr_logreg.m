function [ x, p, conv_log ] = admm_relerr_logreg( A, b, c, sigma, epsilon )
%   ADMM_ABSSUM_LOGREG solve the logistic regression with l1 regularization
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
%   Output:
%         x: primal solution such that x = [v;w]
%         p: dual solution.
%         conv_log: contains objective value, primal residual, dual
%         residual, inner iteration number at each outer iteration.

    t_start = tic;
    
    [nrows,ncols] = size(A);
    % calculate nu, use the rule from Boyd's paper
    ratio = sum(b == 1)/(nrows);
    nu =nrows * 0.1 * 1/nrows * norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');
    % initialization
    max_out_iter = 10000;
    max_inner_iter = 200;
    total_inner_iter = 0;
    outer_iter = 0;
    % this sequence is used for relative error conditions
    w = zeros(ncols+1,1);
    % the extra element represents the intercept v
    x = zeros(ncols+1,1);
    z = zeros(ncols+1,1);
    p = zeros(ncols+1,1);
    % backtracking constants
    backtrack_coeff = 0.1;
    backtrack_factor = 0.5;
    
    while outer_iter < max_out_iter
        inner_iter = 0;
        % inner iteration, apply Newton method to the ADMM subproblem until termination
        % conditions are satisfied
        while inner_iter < max_inner_iter
            % aux is exp(-b*v - A*w)
            aux = exp(-b*x(1)-A*x(2:end));
            assist = aux ./ (1+aux);
            % the gradient of the loss function is gradf = [-b';-A'] * assist;
            % the first element of gradf is the gradfv = -b'* assist;
            gradfv = -b'*assist;
            % the rest of the gradf is the gradfw = -A' * assist;
            gradfw = -A'*assist;
            % gradf = [gradfv;gradfw]
            [~,x,g] = newton([gradfv;gradfw],x,z,p,c,backtrack_coeff,backtrack_factor,ncols,A,b,aux);
            inner_iter = inner_iter + 1;
            % termination condition for inner Newton's method. This is
            % relative type of error conditions
            if (2/c) * abs(dot((w-x),g)) + sum(g.^2) <=  sigma * sum((x-z).^2) || norm(g) < epsilon / 10
            % if abs(dfx) < epsilon / 10 % this is affine invariant type of
            % error condition
                conv_log.inner_iter(outer_iter + 1) = inner_iter;
                break;
            end
        end
        total_inner_iter = total_inner_iter + inner_iter;
        z_old = z;
        % update z
        z = x + (1/c)*p;
        z(2:end) = shrinkage(z(2:end), nu/c);
        % update multiplier
        p = p + c * (x - z);
        % udpate w
        w = w - c * g;
        outer_iter = outer_iter + 1;
        % logging the results
        conv_log.obj_val(outer_iter) = funeval_logreg(nu,aux,z(2:end));
        conv_log.prim_res(outer_iter) = norm(x-z);
        conv_log.dual_res(outer_iter) = norm(z - z_old);
        conv_log.tol_grad(outer_iter) = opttol_logreg(gradfv,gradfw,nu,x,epsilon);
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol_grad = %10.8f\t total inner: %6.0f\n',outer_iter,inner_iter,conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) < epsilon 
            break;
        end
    end
    toc(t_start);
end