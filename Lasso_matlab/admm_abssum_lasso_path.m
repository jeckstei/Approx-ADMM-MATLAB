function [ x, lambda, conv_log, path ] = admm_abssum_lasso_path( A, b, c, eta, K, power, epsilon )
%   admm_abssum_lasso solves the LASSO problem with alternating direction
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
%               norm(r) <= tau_k  tau_k =(eta/k) when k <= K   eta/k^2 when k >= K 
%       2. Accept approximate solution: x^{k+1} = x^{k,l}, y^{k+1} = y^{k,l} 
%       3. z^{k+1} = sgn(x^{k+1} + (1/c) * \lambda^k) * max{0, | x^{k+1} + (1/c) * \lambda^k|-(\nu/c)}
%       4. Update multipliers:  
%               \lambda^{k+1} = \lambda^k + c * (x^{k+1} - z^{k+1})

    if (K<0)
        error('K has to be a positve integer');
    end
    if (power<= 1)
        error('power has be to greater than 1');
    end
    t_start = tic;
    % compute some quantities that will be used frequently
    [nrows,ncols] = size(A);
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
    path = [];
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
            innerTol = max(eta/(outer_iter+1)^power,epsilon/10);
            if norm(r) <= innerTol
                conv_log.inner_iter(outer_iter+1) = inner_iter;
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
        path = [path,lambda + c*z];
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
        fprintf('outter iter:%5.0f (%3.0f)  obj_val = %10.8f\t tol_grad = %10.8f\t total inner: %6.0f\n',outer_iter,inner_iter,conv_log.obj_val(outer_iter),conv_log.tol_grad(outer_iter),total_inner_iter);
        % global termination condition
        if conv_log.tol_grad(outer_iter) <= epsilon
            break;
        end
    end
    toc(t_start);
end