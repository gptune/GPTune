close all
clear all
clc
m = [4022 1204 4812 2958 3790 2980 4125 3578 513 3241 4674 1565 2678 1011 4487 1912 809 2475 2229 1825];
n = [1557 3781 533 2165 1020 1913 1404 3293 2507 4020 3608 4440 2885 4662 3024 899 4839 1639 4247 2571];
opt = [0.132649 0.036429 0.04248 0.099081 0.061807 0.106152 0.095165 0.266041 0.020529 0.135992 0.252904 0.108413 0.168156 0.063548 0.16516 0.038 0.048174 0.087301 0.137455 0.092709];
flop = (max(m,n).*min(m,n).^2 - 1/3*min(m,n).^3)/1e9;

m1 = [4674];
n1 = [3608];
opt1 = [0.260];
flop1 = (max(m1,n1).*min(m1,n1).^2 - 1/3*min(m1,n1).^3)/1e9;


%% Plot figure 1
axisticksize = 40;
origin = [200,60];
markersize = 10;
LineWidth = 3;




figure(1)


hd1 = plot3(m,n,opt,'ro','MarkerSize',markersize,'MarkerFaceColor','r','LineWidth',LineWidth);
hold on
hd1 = plot3(m1,n1,opt1,'bo','MarkerSize',markersize,'MarkerFaceColor','b','LineWidth',LineWidth);
hold on

mlin = linspace(0,5000,100);
nlin = linspace(0,5000,100);
[M,N]=meshgrid(mlin,nlin);
Z = griddata(m,n,opt,M,N,'cubic');
mesh(M,N,Z,'FaceAlpha',0.5);
hold on;



gca = get(gcf,'CurrentAxes');
% set(gca,'XTick',[1:length(SOLVE_SLU(:,1))])
% set(gca,'TickLabelInterpreter','none')
% xticklabels(xtick)
% xtickangle(45)
% xlim([0,2.5]);
% ylim([0,2000]);


legs = {};
legs{1,1} = ['\delta=20'];
legs{1,2} = ['\delta=1'];
% legs{1,3} = ['Pareto optima'];

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('m');
ylabel(str,'interpreter','Latex')
str = sprintf('n');
xlabel(str,'interpreter','Latex')
str = sprintf('time (s)');
zlabel(str,'interpreter','Latex')


gca=legend(legs,'interpreter','tex','color','none','NumColumns',1,'Location','northwest');

set(gcf,'Position',[origin,1000,700]);

fig = gcf;

str = 'scalapack_MLA_optimum_time.eps';
saveas(fig,str,'epsc')





figure(2)

hd1 = plot3(m,n,flop./opt,'ro','MarkerSize',markersize,'MarkerFaceColor','r','LineWidth',LineWidth);
hold on
zlim([min(flop./opt),max(flop./opt)]);
% hd1 = plot3(m,n,opt,'ro','MarkerSize',markersize,'MarkerFaceColor','r','LineWidth',LineWidth);
% hold on
hd1 = plot3(m1,n1,flop1./opt1,'bo','MarkerSize',markersize,'MarkerFaceColor','b','LineWidth',LineWidth);
hold on

mlin = linspace(0,5000,100);
nlin = linspace(0,5000,100);
[M,N]=meshgrid(mlin,nlin);
Z = griddata(m,n,flop./opt,M,N,'cubic');
mesh(M,N,Z,'FaceAlpha',0.5);
hold on;




gca = get(gcf,'CurrentAxes');
% set(gca,'XTick',[1:length(SOLVE_SLU(:,1))])
% set(gca,'TickLabelInterpreter','none')
% xticklabels(xtick)
% xtickangle(45)
% xlim([0,2.5]);
% ylim([0,2000]);


legs = {};
legs{1,1} = ['\delta=20'];
legs{1,2} = ['\delta=1'];
% legs{1,3} = ['Pareto optima'];

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('m');
ylabel(str,'interpreter','Latex')
str = sprintf('n');
xlabel(str,'interpreter','Latex')
str = sprintf('GFlops');
zlabel(str,'interpreter','Latex')


gca=legend(legs,'interpreter','tex','color','none','NumColumns',1,'Location','northwest');

set(gcf,'Position',[origin,1000,700]);

fig = gcf;

str = 'scalapack_MLA_optimum_gflops.eps';
saveas(fig,str,'epsc')
