function [ x, p, conv_log ] = admm_absgeom_logreg( A, b, c, eta, factor, epsilon, m )
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
%         ita: a large positive number to enlarge the error torlerance at
%         begining phase
%         m: limited memory parameter: positive integer
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
    max_inner_iter = 50;
    total_inner_iter = 0;
    outer_iter = 0;
    innerTol = eta;
    % the extra element represents the intercept v
    x = zeros(ncols+1,1);
    z = zeros(ncols+1,1);
    p = zeros(ncols+1,1);
    % Weak Wolfe conditions parameters that satisfied 0 < wolfeCondC1 <
    % wolfeCondC2 < 1
    wolfeCondC1 = 1e-4;
    wolfeCondC2 = 0.9;
    if wolfeCondC1 < 0 || wolfeCondC1 > wolfeCondC2 || wolfeCondC2 >= 1
        fprintf('Wolfe weak conditions parameters should satify 0< wolfe_c1 < wolfe_c2 < 1')
    end
    
    while outer_iter < max_out_iter
        inner_iter = 0;
        % function and gradient value at initial points, gradf is the gradient
        % of logistic loss function
        [~,fVal,~,grad] = funcval_logreg(x,z,p,A,b,c,nu);
        %fprintf('grad_before: %2.10f\n',norm(grad));
        % inner iteration, apply Newton method to the ADMM subproblem until termination
        % conditions are satisfied
        while inner_iter < max_inner_iter
            if inner_iter == 0
                % initial search direction
                d = -grad;
                S = [];
                Y = [];
                rho = []; 
            else
                % compute quantities that used in BFGS iteration
                s = x - xOld;
                y = grad - gradOld;
                sTy = s'*y;
                yTy = y'*y;
                rhok = 1/sTy;
                gamma = sTy/yTy;
                % only use m most recent iterations
                if inner_iter <= m
                    S = [s S];
                    Y = [y Y];
                    rho = [rhok rho];
                % discard previous s, y and rho
                else
                    S = [s S(:,1:(end-1))];
                    Y = [y Y(:,1:(end-1))];
                    rho = [rhok rho(1:(end-1))];
                end
                % obtain the number of available iterations for two-loop
                % recursion
                twoLoopIter = size(S, 2);
                % begin two-loop recursion
                q = grad;
                for i = 1:twoLoopIter
                    alpha(i) = rho(i)*S(:,i)'*q;
                    q = q - alpha(i)*Y(:,i);
                end
                r = gamma * q;
                for i = twoLoopIter:-1:1
                    beta = rho(i)*Y(:,i)'*r;
                    r = r + (alpha(i)-beta)*S(:,i);
                end
                % end of two-loop recursion
                % new search direction 
                d = -r;
            end
            xOld = x;
            gradOld = grad;
            [~,x,fVal,gradf,grad,objVal] = linesearch(x,z,p,fVal,grad,d,wolfeCondC1,wolfeCondC2,A,b,c,nu);
            %fprintf('norm(grad): %10.20f\n',norm(grad,1));
            %fprintf('step size:%3.20f\n',s);
            inner_iter = inner_iter + 1;
            %fprintf('step size is: %3.9f\n',stepSize);
            % termination condition for inner Newton's method.
			normg = norm(grad);
            if normg <  innerTol
                conv_log.inner_iter(outer_iter+1) = inner_iter;
                break;
            end                
        end
        innerTol = max(epsilon/10,innerTol*factor);
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
        if conv_log.tol_grad(outer_iter) < epsilon 
            break;
        end
    end
    toc(t_start);
end
