function [dfx, x, g] = newton(gradf,x,z,lambda,c,alpha,BETA,ncols,A,b,aux)
%   this function is one pass of Newton iteration, with backtracking line
%   search strategy.
    I = eye(ncols+1);
    % define the f subproblem, as well as the f subproblem of ADMM. x = [v;w]
    f = @(v,w) (sum(log(1 + exp(-b*v-A*w))) + (c/2)* sum (([v;w]-z+(1/c)*lambda).^2));
    fx = f(x(1),x(2:end));
    % g is the gradient of the f subproblem
    g = gradf + c*(x-z+(1/c)*lambda);
    % The Hessian matrix: H = [-b';-A'] * diag(exp(-b*x(1)-A*x(2:end))./(1 + exp(-b*x(1)-A*x(2:end)).^2)) * [-b -A] + c*I;
    H = [-b';-A'] * diag(aux./((1 + aux).^2)) * [-b -A] + c*I;
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

