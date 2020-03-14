function [ x_next, p_next, r_next ] = linear_CG( Ap, x_start, p_start, r_start )
%   linear_CG Summary of this function goes here
%   This function is actaully one iteration of conjugate gradient (CG) method.
%   We use CG method to solve the linear system: Ax = b for x.
%   where A must be a square symmetric positive definite matrix.
%
%   CG method we are using here is: (P112, Algorithm 5.2, Numerical Optimization, 2nd Edition)
%       Given initial point x_0
%       Set r_0 <-- Ax_0 - b, p_0 <-- -r_0, k <-- 0;
%       while r_k ~= 0
%           alpha_k <-- norm(r_k)/ p_k' * A * p_k;
%           x_{k+1} <-- x_k + alpha_k * p_k;
%           r_{k+1} <-- r_k + alpha_k * A * p_k;
%           beta_{k+1} <-- norm(r_{k+1})/ norm{r_k};
%           p_{k+1} <-- -r_{k+1} + beta_{k+1} * p_k;
%           k <-- k + 1;
%       end while
%
%   Since the size of our data matrix is large and we use Ap = A * p_start
%   as a input argument.

%    alpha = norm(r_start)^2 / dot(p_start, Ap);
    alpha = sum(r_start.^2) / dot(p_start, Ap);
    x_next = x_start + alpha * p_start;
    r_next = r_start + alpha * Ap;
%    beta = norm(r_next)^2 / norm(r_start)^2;
    beta = sum(r_next.^2) / sum(r_start.^2);
    p_next = -r_next + beta * p_start;
    
end

