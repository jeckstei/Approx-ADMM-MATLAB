function [ x, lambda, conv_log ] = admm_exact_lasso_dist( A, b, c, epsilon, target )
%   admm_exact_lasso solves the LASSO problem with alternating direction
%   method of multipliers (ADMM). The LASSO 
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
%          to get x^{k,l} unitl 
%               norm(r) <= epsilon/10
%       2. Accept approximate solution: x^{k+1} = x^{k,l}, y^{k+1} = y^{k,l} 
%       3. z^{k+1} = sgn(x^{k+1} + (1/c) * \lambda^k) * max{0, | x^{k+1} + (1/c) * \lambda^k|-(\nu/c)}
%       4. Update multipliers:  
%               \lambda^{k+1} = \lambda^k + c * (x^{k+1} - z^{k+1})

    t_start = tic;
    % compute some quantities that will be used frequently
    ncols = size(A,2);
    Atb = A' * b;
    % gradf = \partial (1/2)||Ax-b||^2, x is initialized to be 0, 
    % so gradf = - Atb
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
        % r is the residual of the linear system r = A'A * x + c*x -(Atb -lambda + c*z)
        % thus, r = gradf + c * x + lambda - c * z
        r = gradf + c * x + lambda - c * z;
        p = -r;
        while inner_iter < max_inner_iter            
            Ap = A * p;
            Ap = A' * Ap;
            Ap = Ap + c * p;                                
            [x,p,r] = linear_CG(Ap, x, p, r);

            inner_iter = inner_iter + 1;
            % error conditrion for inner iteration
            if norm(r) <= epsilon / 10 
                break;
            end 
        end
        total_inner_iter = total_inner_iter + inner_iter;
        z_old = z;
        z = sign(x + (1/c) * lambda) .* max(0, abs(x + (1/c) * lambda) - (nu/c));
        gradf = A * x;
        gradf = A' * gradf;
        gradf = gradf - Atb;
        lambda = lambda + c * (x - z);
        outer_iter = outer_iter + 1;
        % record number of inner iterations between each outer iteration
        conv_log.inner_iter(outer_iter) = inner_iter;
        % record the objective function value at eather outer iteration
        conv_log.obj_val(outer_iter) = funeval_lasso(A,b,x,x,nu);
        % record the primal residual after each outer iteration
        conv_log.prim_res(outer_iter) = norm(x-z);
        % record the dual residual after each outer iteration
        conv_log.dual_res(outer_iter) = norm(z-z_old);
        % return the smallest magnitude element in subdifferential of
        % objective function
        conv_log.tol_grad(outer_iter) = opttol_lasso(gradf,nu,x,epsilon);
        % distance from target solution
        conv_log.dist(outer_iter) = norm(z - target);
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol_grad = %10.8f\t total inner: %6.0f\n',outer_iter,inner_iter,conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) <= epsilon
            break;
        end
    end
    toc(t_start);
end