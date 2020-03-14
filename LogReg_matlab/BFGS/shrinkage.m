function [ z ] = shrinkage( x, kappa )
%   this is the function of shrinkage operator
%   another way to write this function is
%   z = sign(x) .* max(0, abs(x) - (kappa) * ones(ncols,1));
    
    z = max(0, x - kappa) - max(0, -x-kappa);

end

