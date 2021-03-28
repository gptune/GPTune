clc
clear all
close all

origin = [200,60];
axisticksize = 32;
markersize = 10;
LineWidth = 3;

lineColors = line_colors(8);

lineColors(3,:) = lineColors(5,:);
lineColors(4,:) = [0 0 0];

linspecs = {'-o','-x','-*','-^'};







%% 1-vs-32 cores for demo 
 
p_model = [20 40 80 160 320];
time_model = [1.0623 1.4366 3.33519819 13.573 33.647];
time_model_ref = [9.35E-01 2.14E+00 2.57E+01 2.07E+02 1.34E+03];

p_search = [20 40 80 160 250 320];
time_search = [14.47	19.96	27	88.633 209.877225068 355.2];
time_search_ref = [62.361	101.54	259.226804	976.78 2400 3920.2];



figure(1)
loglog(p_model,time_model,'-o', 'MarkerSize',markersize,'LineWidth',LineWidth,'Color',[lineColors(3,:)])
hold on
loglog(p_model,time_model_ref,'--v', 'MarkerSize',markersize,'LineWidth',LineWidth,'Color',[lineColors(3,:)])
hold on
loglog(p_search,time_search,'-o', 'MarkerSize',markersize,'LineWidth',LineWidth,'Color',[lineColors(4,:)])
hold on
loglog(p_search,time_search_ref,'--v', 'MarkerSize',markersize,'LineWidth',LineWidth,'Color',[lineColors(4,:)])
hold on




xlim([0,max(p_model)*1.1])
% ylim([1.8,300])
gca = get(gcf,'CurrentAxes');
set(gca,'Xtick',p_model)
lgd=legend('Modeling (32 MPIs)','Modeling (1 MPI)','Search (32 MPIs)','Search (1 MPIs)','Location','NorthWest');
lgd.NumColumns=1; 
lgd.Interpreter='latex';
% lgd.Box='off'
    
grid on

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('Time ($s$)');
ylabel(str,'interpreter','latex')
% str = sprintf('$\epsilon_{tot}$');
xlabel('$\epsilon_{tot}$','interpreter','latex')



set(gcf,'Position',[origin,1000,700]);

fig = gcf;
% style = hgexport('factorystyle');
% style.Bounds = 'tight';
% hgexport(fig,'-clipboard',style,'applystyle', true);

str = ['parallel_model_search.eps'];
saveas(fig,str,'epsc')   


