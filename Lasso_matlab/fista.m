function [x, iter, conv_log, opt_obj_val] = fista(gradf, proj, calc_F, xinit, L, max_iter, lambda, epsilon)   
%  INPUT:
%       gradf    : a function calculating gradient of f(x) given x.
%       proj     : a function calculating backward step.
%       xinit    : a initial start point.
%       L        : the Lipschitz constant of the gradient of f(x).
%       lambda   : regularization parameter, positive scalar. 
%       max_iter : maximum iterations of the algorithm. 
%       epsilon  : a tolerance, the algorithm will stop.
%       calc_F   : optional, a function calculating value of F at x via feval(calc_F, X). 
%  OUTPUT:
%       x        : solution
%       iter     : number of iterations
%       opt_obj_val  : value of objective function at x
%       conv_log : a struct contains objective values and gradient tolerance

    Linv = 1/L;    
    kappa = lambda*Linv;
    x_old = xinit;
    y_old = xinit;
    t_old = 1;
    iter = 0;
    %% MAIN LOOP
    while  iter < max_iter
        iter = iter + 1;
        x_new = feval(proj, y_old - Linv*feval(gradf, y_old), kappa);
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        %% show progress
        conv_log.obj_val(iter) = feval(calc_F, y_new);
        %% check stop criteria
        conv_log.tol(iter) = opttol_lasso(feval(gradf,y_new),lambda,x_new,epsilon);
        if mod(iter,100) ==0
            fprintf('iter = %3d, obj_val = %10.8f, tol = %10.8f\n', iter, conv_log.obj_val(iter),conv_log.tol(iter));
        end
        if conv_log.tol(iter) < epsilon
            fprintf('iter = %3d, obj_val = %10.8f, tol = %10.8f\n', iter, conv_log.obj_val(iter),conv_log.tol(iter));         
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
    end
    x = x_new;
    opt_obj_val = feval(calc_F, x);
end 
function tol = opttol_lasso(gradf, lambda, x,epsilon)
%   opttol returns the max element in the subdifferential of 
%   objective function (1/2)||A * x - b||^2 + \lambda ||x||_1 at point x.
    ix  = abs(x) <= epsilon;
    x(ix) = 0;
    v = abs(gradf + sign(x) * lambda);
    v = v - (1 - abs(sign(x)))*lambda;
    tol = max(v);
    if (tol < 0)
        tol = 0;
    end
end