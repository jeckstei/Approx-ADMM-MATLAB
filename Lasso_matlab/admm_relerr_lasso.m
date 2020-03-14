function [ x, lambda,conv_log ] = admm_relerr_lasso( A, b, c, sigma, epsilon)
%   admm_relerr_lasso solves the LASSO problem with alternating direction
%   method of multipliers (ADMM) and relative error criterion. The LASSO 
%   problem is genericly described by
%               minimize  (1/2)\|Ax-b\|^2 + \nu\|x\|_1
%   where A is the data (observations) matrix and b is the response vector,
%   \nu is the a positive scalar. The ADMM form of this problem is the
%   following
%               minimize (1/2)\|Ax-b\|^2 + \nu\|z\|_1
%               subject to         x-z = 0
%   Applying ADMM to the LASSO problem, we have
%   ADMM for LASSO with relative error criterion:
%       1. Apply any iterative linear solver to 
%                  (A'A + cI) * x = A' * b + c * z^k - \lambda^k
%          unitl 
%               (2/c) * |(w^k - x^{k,l})' * y^{k,l}| + norm(y^{k,l})^2 <=
%               \sigma * norm(x^{k,l} - z^k)^2
%       2. Accept approximate solution: x^{k+1} = x^{k,l}, y^{k+1} = y^{k,l} 
%       3. z^{k+1} = sgn(x^{k+1} + (1/c) * \lambda^k) * max{0, | x^{k+1} + (1/c) * \lambda^k|-(\nu/c)}
%       4. Update multipliers:  
%               \lambda^{k+1} = \lambda^k + c * (x^{k+1} - z^{k+1})
%       5. Update "drift accumulator": w^{k+1} = w^{k} - c * y^{k+1}

    % compute some quantities that will be used frequently
    t_start = tic;
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
    w = zeros(ncols,1);
    outer_iter = 0;
    while outer_iter < max_out_iter
        inner_iter = 0;
        % r is the residual of the linear system r = A'A * x + c*x -(Atb -lambda + c*z)
        % thus, r = gradf + c * x + lambda - c * z 
        r = gradf + c * x + lambda - c * z;
        p = -r;
        while inner_iter < max_inner_iter            
            Ap = A * p;
            Ap = A' * Ap;
            Ap = Ap + c * p;
            [x,p,r] = linear_CG(Ap, x, p, r);

            gradf = A * x;
            gradf = A' * gradf;
            gradf = gradf - Atb;
            y = gradf + lambda + c * (x - z);
            inner_iter = inner_iter + 1;
            if norm(r) <= epsilon / 10
                break;
            end
            % relative error condition for inner iteration mixed with the
            % good enough error tolorance
            if (2/c) * abs(dot((w-x), y)) + sum(y.^2) <= sigma * sum((x - z).^2) 
                break;
            end 
        end
        total_inner_iter = total_inner_iter + inner_iter;
        z_old = z;
        z = sign(x + (1/c) * lambda) .* max(0, abs(x + (1/c) * lambda) - (nu/c));
        lambda = lambda + c * (x - z);
        w = w - c * y;
        outer_iter = outer_iter + 1;
        % record number of inner iterations between each outer iteration
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
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol_grad = %10.8f\t total inner: %6.0f\n',outer_iter,inner_iter,conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) <= epsilon
            break;
        end
    end
    toc(t_start);
end