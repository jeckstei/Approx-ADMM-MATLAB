function [x, iter, conv_log, obj_val] = backtrack_fista_lasso(gradf, proj, calc_F, xinit,A,b, max_iter, lambda, eta, epsilon)   
%  INPUT:
%       proj     : a function calculating backward step.
%       xinit    : a initial start point.
%       lambda   : regularization parameter, positive scalar. 
%       max_iter : maximum iterations of the algorithm. 
%       epsilon  : a tolerance, the algorithm will stop.
%       calc_F   : optional, a function calculating value of F at x via feval(calc_F, X). 
%  OUTPUT:
%       x        : solution
%       iter     : number of iterations
%       conv_log : struct that contains array of tol and objective value
%       obj_val  : value of objective function at x

    x_old = xinit;
    x_new = xinit;
    y_old = xinit;
    t_old = 1;
    iter = 0;
    %% MAIN LOOP
    while  iter < max_iter
        iter = iter + 1;
        % line search procedure
        L = 0.1;
        ls_power = 1;
        grdf = feval(gradf,y_old);
        while feval(calc_F, feval(proj,y_old,lambda/L),A,b) > 1 *calc_Q(feval(proj,y_old,lambda/L),y_old,A,b,lambda,L,grdf)
            L = eta^ls_power * L;  
            ls_power = ls_power + 1;
            % fprintf('power = %3d\n',ls_power);
        end
        L = eta^(ls_power + 1) * L;
        if L > 144814.3
            L_e = L;
            L = 144814.3;
        end
        % shrinkage operation
        x_new = feval(proj, y_old - (1/L)*grdf, lambda/L);
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        %% show progress
        conv_log.obj_val(iter) = feval(calc_F, x_new,A,b);
        %% check stop criteria
        conv_log.tol(iter) = opttol_lasso(grdf,lambda,x_new,epsilon);
        if mod(iter,10) == 0
            fprintf('L = %3d, L_e = %3d\n',L,L_e);
            fprintf('iter = %3d, obj_val = %10.8f, tol = %10.8f\n',iter, conv_log.obj_val(iter),conv_log.tol(iter));
        end
        if conv_log.tol(iter) < epsilon
            fprintf('iter = %3d, obj_val = %10.8f, tol = %10.8f\n',iter, conv_log.obj_val(iter),conv_log.tol(iter));            
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
    end
    x = x_new;
    obj_val = feval(calc_F, x,A,b);
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
% Q function that is used in line search procedure
function val = calc_Q(x,y,A,b,lambda,L,grdf)
   val = lambda*norm(x,1) + calc_f(y,A,b) + (x-y)'*grdf + (L/2) * sum((x-y).^2); 
end
    function loss = calc_f(x,A,b)
        loss = 0.5 * sum((A * x - b).^2);
    end 