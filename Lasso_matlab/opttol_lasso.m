function tol = opttol_lasso(gradf, nu, x,epsilon)
%   opttol returns the max element in the subdifferential of 
%   objective function (1/2)||A * x - b||^2 + \nu ||x||_1 at point x.

    ix  = abs(x) <= epsilon;
    x(ix) = 0;
    v = abs(gradf + sign(x) * nu);
    v = v - (1 - abs(sign(x)))*nu;
    tol = max(v);
    if (tol < 0)
        tol = 0;
    end
end
