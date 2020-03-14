function [ stepSize, xNext, funcNext, gradfNext,gradNext,objValNext] = linesearch_MneI( xStart, z, p, funcStart, gradStart, searchDirection, wolfeC1, wolfeC2, A, b, c, nu)
%   LINEARSEARCH is used to find a stepsize that step-size that satisfies
%   the weak Wolfe conditions.
%   Input:
%   xStat: start point
%   z: z value
%   p: p value, i.e. the Lagrangian multiplier
%   funcStart: f subproblem function value at xStart
%   gradStart: gradient of f subproblem at xStart
%   searchDirection: as the name explains
%   wolfeC1: Weak Wolfe conditions coefficient for sufficient decrese,
%       default value: 1e-4
%   wolfeC2: Weak Wolfe conditions coefficient for vurvature conditions,
%       default value: 0.9. Generally, we should enforce 0 < wolfe_c1 < wolfe_c2 < 1 
%   A: the modified data matrix: A = diag(b)* A_original, where A_original
%       is the original data matrix
%   b: the response vector
%   c: the penalty parameter of augmented Lagrangian
%   Output:
%   stepSize: the step size satisfied the weak Wolfe conditions, or the
%       right end point of bracketing interval
%   xNext: xNext = xStart + stepSize * searchDirection
%   funcNext: objective value of f subproblem at x_next
%   gradNext: gradient of f subproblem at x_next
%   rightEndPoint: the right end point of bracketing interval

% step size should be nonnegative
    stepSize = 0;
    xNext = xStart;
    funcNext = funcStart;
    gradNext = gradStart;
    gradfNext1 = gradStart(1);
    gradfNext2 = gradStart(2:end) - c*(xStart(2:end)-z+(1/c)*p);
    gradfNext = [gradfNext1;gradfNext2];
% always set initial tiral to be 1
    stepSizeTrial = 1;
% bisection and expansion control parameters
    numBisection = 0;
    numExpansion = 0;
    rightEndPoint = inf;
% this boolean variable is used to terminate the bisection/expansion loops
    isFound = 0;
    gstd = gradStart' * searchDirection;
    if gstd >=0
        %fprintf('search direction is not a descent direction\n')
        searchDirection = gradStart;
    end
    sdNorm = norm(searchDirection);
    if sdNorm == 0
        error('search direction is 0\n')
    end
    % the algorithm is unstale if this value is too large
    maxNumBisection = 45;
    maxNumExpansion = 60;    
% bisection/expansion loops
    while ~isFound
        xTrial = xStart + stepSizeTrial * searchDirection;
%         if isnan(stepSizeTrial)
%             fprintf('stepSizeTrial is nan\n');
%         else
%             fprintf('step size trial: %2.20f\n',stepSizeTrial);
%         end
        if isnan(xTrial)
           fprintf('xTrial is nan\n');
        end
        [objValTrial,funcTrial, gradfTrial,gradTrial] = funcval_MneI(xTrial,z,p,A,b,c,nu);
        if isnan(funcTrial)
            stepSize = 0;
            %xNext = xTrial;
            xNext = xStart;
            %funcNext = funcTrial;
            funcNext = funcStart;
            %gradfNext = gradfTrial;
            gradfNext = gradStart - c*(xStart(2:end)-z+(1/c)*p);
            %gradNext = gradTrial;
            gradNext = gradStart;
            % will generate error objValNext is not assigned
            fprintf('function value is NaN\n');
            return
        end
        directionDerivative = gradTrial'*searchDirection;
        % the first condtion is violated, step size is not small enough
        if funcTrial > funcStart + wolfeC1 * stepSizeTrial * gstd || isnan(funcTrial)
            rightEndPoint = stepSizeTrial;
        % the section condition is violated, step size is not big enough
        elseif directionDerivative < wolfeC2 * gstd || isnan(directionDerivative)
            stepSize = stepSizeTrial;
            xNext = xTrial;
            funcNext = funcTrial;
            gradNext = gradTrial;
        % both conditions are satified
        else
            stepSize = stepSizeTrial;
            xNext = xTrial;
            funcNext = funcTrial;
            rightEndPoint = stepSizeTrial;
            gradNext = gradTrial;
            gradfNext = gradfTrial;
            objValNext = objValTrial;
            return;
        end
        % begin bisection/expansion
        % step size is not small enough
        if rightEndPoint < inf
            if numBisection < maxNumBisection
                numBisection = numBisection + 1;
                % bisection
                %stepSizeTrial = (stepSize + rightEndPoint)/2;
                stepSizeTrial = (stepSize + rightEndPoint)*0.9;
            else
                isFound = 1;
 %               fprintf('max number of bisection has been reached\n');
                xNext = xTrial;
                funcNext = funcTrial;
                gradfNext = gradfTrial;
                gradNext = gradTrial;
                objValNext = objValTrial;
            end
        % step size is too small    
        else
            if numExpansion < maxNumExpansion
               numExpansion = numExpansion + 1;
               %stepSizeTrial = 2 * stepSize;
               stepSizeTrial = 1.1 * stepSize;
               %fprintf('expansion\n');
            else
                isFound = 1;
                stepSizeTrial = 1;
                fprintf('max number of expansion has been reached\n');
            end
        end
    end
end