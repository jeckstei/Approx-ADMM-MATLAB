function [ tol ] = opttol_logreg( gradfv,gradfw, nu, x, epsilon )
%   opttol_logreg returns the max element in the subdifferential of 
%   objective function sum( log(1 + exp(-b_i*(a_i'*u + v)) ) + nu*norm(u,1)
%   the (sub)gradient of the logistic loss function is [gradfv;gradfw]

    % x = [v;w], where v is real number representing intercept
    w = x(2:end);
    ix = abs(w) <= epsilon;
    w(ix) = 0;
    % v here is different from the v in objective function
    u = abs(gradfw + sign(w) * nu);
    u = u - (1-abs(sign(w))) * nu;
    u = [gradfv;u];
    tol = max(u);
    if (tol < 0)
        tol = 0;
    end
end

