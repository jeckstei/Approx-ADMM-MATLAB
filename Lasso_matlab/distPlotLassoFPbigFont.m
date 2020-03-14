function [ objmin ] = distPlotLassoFPbigFont(name,A,b,c,epsilon)
% Produce comparison plot

[xe,lame,elog,epath] = admm_exact_lasso_path(A,b,c,epsilon/100);
[xr,lamr,rlog,rpath] = admm_primDR_lasso_path(A,b,c,0.99,epsilon/1000);
[xa,lama,alog,apath] = admm_abssum_lasso_path(A,b,c,1,0,1.5,epsilon/1000);
[xg,lamg,glog,gpath] = admm_absgeom_lasso_path(A,b,c,1,0.99,1e-6/1000);

eouter = min(find(elog.tol_grad <= epsilon));
router = min(find(rlog.tol_grad <= epsilon));
aouter = min(find(alog.tol_grad <= epsilon));
gouter = min(find(glog.tol_grad <= epsilon));

fprintf('Exact:  outer=%d, inner=%d\n',eouter,sum(elog.inner_iter(1:eouter)));
fprintf('primDR: outer=%d, inner=%d\n',router,sum(rlog.inner_iter(1:router)));
fprintf('abs15:  outer=%d, inner=%d\n',aouter,sum(alog.inner_iter(1:aouter)));
fprintf('absgeo: outer=%d, inner=%d\n',gouter,sum(glog.inner_iter(1:gouter)));

size = min([eouter,router,aouter,gouter]);

fprintf('Plotting through outer iterate %d\n',size);

edists = pathDist(epath,size);
rdists = pathDist(rpath,size);
adists = pathDist(apath,size);
gdists = pathDist(gpath,size);

semilogy(1:size,edists,'--b', ...
         1:size,rdists,'-r',  ...
         1:size,adists,':k',  ...
         1:size,gdists,'-.g', ...
         'LineWidth',3);
         
legend('admm\_exact','admm\_primDR','admm\_abssum','admm\_absgeom');
title(name);
xlabel('Outer Iteration');
ylabel('Distance from Fixed Point');
set(gca,'FontSize',30,'FontName','Times','TitleFontWeight','normal');
    
end

