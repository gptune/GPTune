close all
clear all
clc
m = [9496,
21487,
37073,
11070,
27958,
17026,
23884,
6666,
23324,
8381];
n = [7549,
20981,
20774,
34241,
39976,
15910,
4712,
11170,
26545,
29070];
opt = [0.741164
4.497037
7.047603
1.071485
7.212767
2.186616
0.924713
0.411944
5.811236
1.501592];
flop = (max(m,n).*min(m,n).^2 - 1/3*min(m,n).^3)/1e9;



%% Plot figure 1
axisticksize = 40;
origin = [200,60];
markersize = 10;
LineWidth = 3;




figure(1)


hd1 = plot3(m,n,opt,'ro','MarkerSize',markersize,'MarkerFaceColor','r','LineWidth',LineWidth);
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
legs{1,1} = ['\delta=10'];
% legs{1,2} = ['\delta=1'];
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
% hd1 = plot3(m1,n1,flop1./opt1,'bo','MarkerSize',markersize,'MarkerFaceColor','b','LineWidth',LineWidth);
% hold on

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
legs{1,1} = ['\delta=10'];
% legs{1,2} = ['\delta=1'];
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
