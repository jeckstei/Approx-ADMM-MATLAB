function [ x, lambda, conv_log ] = admm_primDR_lasso( A, b, c, sigma, epsilon )
%   admm_primDR_lasso solves the LASSO problem with an instance of alternating direction
%   method of multipliers (ADMM): primal D-R splitting method and relative error criterion. The LASSO 
%   problem is genericly described by
%               minimize  (1/2)\|Ax-b\|^2 + \nu\|x\|_1
%   where A is the data (observations) matrix and b is the response vector,
%   \nu is the a positive scalar. The ADMM form of this problem is the
%   following
%               minimize (1/2)\|Ax-b\|^2 + \nu\|z\|_1
%               subject to         x-z = 0
%   Applying primal D-R splitting method to the LASSO problem, we have
%   Primal D-R splitting for LASSO with relative error criterion:
%       1. Initialize: x^0 , z^1, \lambda^0 
%       2. At iteration k+1, apply any iterative linear solver to 
%                  (A'A + (1/c)I) * x = A' * b + 1/c * z^{k+1} + \lambda^k
%          to get x^{k,l}. 
%          2a. \lambda^{k,l} =  A'*(A*x-b)
%          2b. z^{k,l} = sgn(x^{k,l} - (1/c) * \lambda^{k,l}) * max{0, | x^{k,l} - c * \lambda^{k,l}|-(\nu*c)}
%          Until 
%                   ||\lambda^{k,l}-c*x^{k,l}-(\lambda^k-c*z^k)|| <=
%                                   \sigma*||\lambda^{k,l}-c*z^{k,l}-(\lambda^k-c*z^k)||
%       3. Accept approximate solution:  
%   Note: to get equivalent classical ADMM the 'c' in this file should be
%   replaced with (1/c) and lambda should be replaced with -lambda
    
    t_start = tic;
    % compute some quantities that will be used frequently
    ncols = size(A,2);
    Atb = A' * b;
    % gradf = \partial (1/2)||Ax-b||^2, x is initialized to be 0, 
    % so  gradf = - Atb
    gradf = - Atb;
    % select coefficient in front of the l_1 norm, 0.1 is from Boyd's paper
    nu = 0.1 * norm(Atb, inf);                              
    % initialization
    max_out_iter = 10000;
    max_inner_iter = 200;
    total_inner_iter = 0;
    x = zeros(ncols,1);
    z = zeros(ncols,1);
    lambda = zeros(ncols,1);
    outer_iter = 0;
    while outer_iter < max_out_iter
        inner_iter = 0;
        z_old = z;
        lambda_old = lambda;
        % r is the residual of the linear system, r = A'A * x + (1/c)*x -(Atb +lambda + (1/c)*z)
        % thus, r = gradf + (1/c) * x - lambda - (1/c) * z 
        r = gradf + (1/c) * x - lambda - (1/c) * z;
        p = -r;
        while inner_iter < max_inner_iter            
            Ap = A * p;
            Ap = A' * Ap;
            Ap = Ap + (1/c) * p;                                
            [x,p,r] = linear_CG(Ap, x, p, r);

            gradf = A * x;
            gradf = A' * gradf;
            gradf = gradf - Atb;
            % note this line. This is Primal ADMM, we don't relabel any
            % variables.
            lambda = gradf;
            z = sign(x - c * lambda) .* max(0, abs(x - c * lambda) - nu * c);
            inner_iter = inner_iter + 1;
            % relative error conditrion for inner iteration
            if norm(c * lambda - c * lambda_old + x - z_old) <= sigma * norm(z + c * lambda - z_old - c * lambda_old)
                % if this update is ignored, the algorithm converges even
                % quicker. Need to prove the general convergence.
                lambda = lambda_old + (1/c) * (z_old - x);
                break;
            end
            if norm(r) <= epsilon / 10
                lambda = lambda_old + (1/c) * (z_old - x);                
                break;
            end            
        end
        total_inner_iter = total_inner_iter + inner_iter;
        outer_iter = outer_iter + 1;
		% record the number of inner iterations 
		conv_log.inner_iter(outer_iter+1) = inner_iter;
        % record the objective function value at eather outer iteration
        conv_log.obj_val(outer_iter) = funeval_lasso(A,b,x,x,nu);
        % record the primal residual after each outer iteration
        conv_log.prim_res(outer_iter) = norm(x-z);
        % record the dual residual after each outer iteration
        conv_log.dual_res(outer_iter) = norm(z-z_old);
        % return the smallest magnitude element in subdifferential of
        % objective function
        conv_log.tol_grad(outer_iter) = opttol_lasso(gradf,nu,x,epsilon);
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol.grad = %10.8f\t total inner: %6.0f\n',outer_iter,inner_iter,conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) <= epsilon
            break;
        end
    end
    toc(t_start);
end