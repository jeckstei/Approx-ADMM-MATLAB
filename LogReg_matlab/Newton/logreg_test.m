function [ x, p,z, conv_log ] = logreg_test( A, b, c, epsilon )
%   ADMM_EXACT_LOGREG solve the logistic regression with l1 regularization
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
    % the extra element represents the intercept v
    x = rand(ncols+1,1);
    z = rand(ncols+1,1);
    p = rand(ncols+1,1);
    % backtracking constants
    alpha = 0.1;
    BETA = 0.5;
    
    while outer_iter < max_out_iter
        inner_iter = 0;
        % inner iteration, run Newton iteration until termination
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
            [~,x,g] = newton([gradfv;gradfw],x,z,p,c,alpha,BETA,ncols,A,b,aux);
            inner_iter = inner_iter + 1;
            % termination condition for inner Newton's method.
            if norm(g) < epsilon / 10 
            % if abs(dfx) < epsilon / 10
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
        outer_iter = outer_iter + 1;
        % logging the results
        conv_log.obj_val(outer_iter) = funeval_logreg(A,b,nu,x,z);
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

function [dfx, x, g] = newton(gradf,x,z,lambda,c,alpha,BETA,ncols,A,b,aux)
    I = eye(ncols+1);
    % define the f subproblem, as well as the f subproblem of ADMM. x = [v;w]
    f = @(v,w) (sum(log(1 + exp(-b*v-A*w))) + (c/2)* sum (([v;w]-z+(1/c)*lambda).^2));
    fx = f(x(1),x(2:end));
    % g is the gradient of the f subproblem
    g = gradf + c*(x-z+(1/c)*lambda);
    % The Hessian matrix: H = [-b';-A'] * diag(exp(-b*x(1)-A*x(2:end))./(1 + exp(-b*x(1)-A*x(2:end)).^2)) * [-b -A] + c*I;
    H = [-b';-A'] * diag(aux./(1 + aux.^2)) * [-b -A] + c*I;
    % Newton step, the descent direction.
    dx = -H\g;
    % Newton decrement
    dfx = g' * dx;
    % line search strategy, backtracking
    t = 1;
    while f(x(1) + t * dx(1),x(2:end) + t * dx(2:end)) > fx + alpha * t * dfx
        t = BETA * t;
    end
    x = x + t * dx;
end

function [ obj_val ] = funeval_logreg( A, b, nu, x, z )
%   This function is used to evaluating objevtive value

    obj_val = sum(log(1 + exp(-A*x(2:end) - b * x(1)))) + nu * norm(z,1);
end