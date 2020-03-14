function [ objmin ] = objValPlotLasso(name,A,b,c,epsilon)
% Produce comparison plot

[xe,lame,elog] = admm_exact_lasso(A,b,c,epsilon);
[xr,lamr,rlog] = admm_primDR_lasso(A,b,c,0.99,epsilon);
[xa,lama,alog] = admm_abssum_lasso(A,b,c,1,0,1.5,epsilon);

objmin = min([elog.obj_val,rlog.obj_val,alog.obj_val]);

esize = length(elog.obj_val);
rsize = length(rlog.obj_val);
asize = length(alog.obj_val);

semilogy(1:esize,elog.obj_val-objmin,'--b',1:rsize,rlog.obj_val-objmin,'-r',1:asize,alog.obj_val-objmin,':k')
         
legend('admm\_exact','admm\_primDR','admm\_abssum')
title(name)
xlabel('Iterations')
ylabel('Objective Error')

sum(elog.inner_iter)
sum(rlog.inner_iter)
sum(alog.inner_iter)

end

