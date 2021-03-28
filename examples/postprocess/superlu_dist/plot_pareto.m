close all
clear all
clc
default = [0.7, 1272.5];
opt1 = [0.121, 2321.52];
opt2 = [0.53,83.60];
opt3 = [1.7747e-01 1.6908e+02;
 1.1657e-01 1.5375e+03;
 1.0736e-01 2.3215e+03;
 3.4224e+00 7.5065e+01;
 6.2960e-01 8.6416e+01;
 4.3432e+00 6.9654e+01;
 3.7338e-01 1.2919e+02];

[tmp,I] = sort(opt3(:,1));
opt3=opt3(I,:);

%% Plot figure 1
axisticksize = 40;
origin = [200,60];
markersize = 20;
LineWidth = 3;
lineColors = line_colors(7);

figure(1)
hd3 = loglog(opt3(:,1),opt3(:,2),'--o','MarkerSize',markersize,'MarkerFaceColor','k','LineWidth',LineWidth,'Color','k');
hold on
hd1 = loglog(opt1(:,1),opt1(:,2),'o','MarkerSize',markersize,'MarkerFaceColor',[lineColors(2,:)],'LineWidth',LineWidth,'Color',[lineColors(2,:)]);
hold on
hd2 = loglog(opt2(:,1),opt2(:,2),'o','MarkerSize',markersize,'MarkerFaceColor',[lineColors(6,:)],'LineWidth',LineWidth,'Color',[lineColors(6,:)]);
hold on
hd1 = loglog(default(:,1),default(:,2),'co','MarkerSize',markersize,'MarkerFaceColor',[lineColors(4,:)],'LineWidth',LineWidth,'Color',[lineColors(4,:)]);
hold on


gca = get(gcf,'CurrentAxes');
set(gca,'YTick',[70, 100, 200, 400, 800, 1600, 2500])
set(gca,'XTick',[0.1, 0.25,0.5,1,2,4])

% set(gca,'TickLabelInterpreter','none')
% xticklabels(xtick)
% xtickangle(45)
xlim([0.08,4]);
ylim([70,2600]);


legs = {};
legs{1,1} = ['Pareto optima'];
legs{1,2} = ['Time optimum'];
legs{1,3} = ['Memory optimum'];
legs{1,4} = ['Default'];

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('Memory (MB)');
ylabel(str,'interpreter','Latex')
str = sprintf('Time (s)');
xlabel(str,'interpreter','Latex')

gca=legend(legs,'interpreter','none','color','none','NumColumns',1);

set(gcf,'Position',[origin,1000,700]);

fig = gcf;

str = 'pareto_superlu.eps';
saveas(fig,str,'epsc')

