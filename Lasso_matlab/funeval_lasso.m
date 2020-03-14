function obj_val = funeval_lasso(A,b,x,z,nu)
    obj_val = 0.5 * sum((A * x - b).^2) + nu * norm(z,1);
end

