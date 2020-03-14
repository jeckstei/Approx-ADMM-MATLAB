function [ obj_val ] = funeval_logreg( nu, aux, z )
%   This function is used to evaluating objevtive value
%   aux = exp(-b*x(1)-A*x(2:end))
    obj_val = sum(log(1 + aux)) + nu * norm(z,1);
end