function [ objmin ] = distPlotLasso(name,A,b,c,epsilon)
% Produce comparison plot

[xtarget,lamtarget,targetlog] = admm_primDR_lasso(A,b,c,0.99,epsilon/1000);

[xe,lame,elog] = admm_exact_lasso_dist(A,b,c,epsilon,xtarget);
[xr,lamr,rlog] = admm_primDR_lasso_dist(A,b,c,0.99,epsilon,xtarget);
[xa,lama,alog] = admm_abssum_lasso_dist(A,b,c,1,0,1.5,epsilon,xtarget);

esize = length(elog.obj_val);
rsize = length(rlog.obj_val);
asize = length(alog.obj_val);

size = min(esize,min(rsize,asize));

semilogy(1:size,elog.dist(1:size),'--b', ...
         1:size,rlog.dist(1:size),'-r',  ...
         1:size,alog.dist(1:size),':k','LineWidth',2);
         
legend('admm\_exact','admm\_primDR','admm\_abssum')
title(name)
xlabel('Iterations')
ylabel('Distance to Optimal Solution')
set(gca,'FontSize',18,'FontName','Times','TitleFontWeight','normal')

fprintf('\nIterations: exact=%d primDR=%d abssum=%d\n',...
        sum(elog.inner_iter),...
        sum(rlog.inner_iter),...
        sum(alog.inner_iter));
    
fprintf('Primal differences: exact=%f primDR=%f abssum=%f\n',...
        norm(xtarget-xe),...
        norm(xtarget-xr),...
        norm(xtarget-xa));
    
fprintf('Dual differences: exact=%f primDR=%f abssum=%f\n\n\n',...
        norm(lamtarget-lame),...
        norm(lamtarget-lamr),...
        norm(lamtarget-lama));
end

