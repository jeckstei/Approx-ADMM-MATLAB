function [x, iter, conv_log, obj_val] = backtrack_fista(proj, calc_F, xinit,A,b, max_iter, lambda, eta, epsilon)   
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
        L = 0.7;
        %L = 2000;
        ls_power = 1;
        [~,grdfw,grdf] = gradf(y_old,A,b);
        while feval(calc_F, feval(proj,y_old,lambda/L),A,b) > calc_Q(feval(proj,y_old,lambda/L),y_old,A,b,lambda,L,grdf)  
            ls_power = ls_power + 1;
            L = eta^ls_power * L;
            fprintf('power = %3d\n',ls_power);
        end
        L = eta^(ls_power + 1) * L;
        % shrinkage operation
        x_new(2:end) = feval(proj, y_old(2:end) - (1/L)*grdfw, lambda/L);
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        [~,grdfw,~] = gradf(y_new,A,b);
        %% show progress
        conv_log.obj_val(iter) = feval(calc_F, y_new,A,b);
        %% check stop criteria
        conv_log.tol(iter) = opttol_logreg(grdfw,lambda,y_new,epsilon);
        fprintf('iter = %3d, obj_val = %10.8f, tol = %10.8f\n',iter, conv_log.obj_val(iter),conv_log.tol(iter));
        if conv_log.tol(iter) < epsilon
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
    end
    x = x_new;
    obj_val = feval(calc_F, x, A, b);
end 
function [ tol ] = opttol_logreg( gradfw, nu, x, epsilon )
%   opttol_logreg returns the max element in the subdifferential of 
%   objective function sum( log(1 + exp(-b_i*(a_i'*u + v)) ) + nu*norm(u,1)
%   the (sub)gradient of the logistic loss function is [gradfv;gradfw]

    % x = [v;w], where v is real number representing intercept
    w = x(2:end);
    ix = abs(w) <= epsilon;
    w(ix) = 0;
    u = abs(gradfw + sign(w) * nu);
    u = u - (1-abs(sign(w))) * nu;
    %u = [gradfv;u];
    tol = max(u);
    if (tol < 0)
        tol = 0;
    end
end
%% Q function that is used in line search procedure
function val = calc_Q(x,y,A,b,lambda,L,grdf)
   val = lambda*norm(x(2:end),1) + calc_f(y,A,b) + (x-y)'*grdf + (L/2) * sum((x-y).^2); 
end
%% gradient of f
function [gradfv,gradfw,grdf] = gradf(x,A,b) 
        % aux is exp(-b*v - A*w)
        aux = exp(-b*x(1)-A*x(2:end));
        assist = aux ./ (1+aux);
        % the gradient of the loss function is gradf = [-b';-A'] * assist;
        % the first element of gradf is the gradfv = -b'* assist;
        gradfv = -b'*assist;
        % the rest of the gradf is the gradfw = -A' * assist;
        gradfw = -A'*assist;
        grdf = [gradfv;gradfw];
end
function loss = calc_f(x,A,b)
    aux = exp(-b*x(1)-A*x(2:end));
    loss = sum(log(1 + aux));
end 