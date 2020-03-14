function [objFunVal, fSubVal, gradf, grad] = funcval_logreg(x,z,p,A,b,c, nu)
% This function is used for evaluating the objective value of f subproblem,
% and the its gradient. gradf is the gradient of logistic loss function.
    
    %E = [-b -A];
    core = -b * x(1) - A * x(2:end);
    if isnan(core)
        error('core is NaN');
    end
    e2Ex = exp(core);
    if isnan(e2Ex)
        error('e2Ex is NaN');
    end
    % value of loss function
    lossFunc = sum(log(1 + e2Ex));
    % objective function value
    objFunVal = lossFunc + nu * norm(x(2:end),1);
    % objective value of f subproblem
    fSubVal = lossFunc + (c/2)*sum((x - z + (1/c)*p).^2);
    % gradient of loss function
    gradf = [-b -A]'*(e2Ex./(1 + e2Ex));
    % gradient of f subproblem
    grad = gradf + c*(x - z + (1/c)*p);
end