close all
clear all
clc


lineColors = line_colors(5);
lineColors(2,:)=[0.8000    0.6667    0.4000];
lineColors(3,:)=[0.4000    0.8000    0.4000];


m = [9496
21487
37073
11070
27958
17026
23884
6666
23324
8381];
n= [
    7549
20981
20774
34241
39976
15910
4712
11170
26545
29070
    ];

flop = (max(m,n).*min(m,n).^2 - 1/3*min(m,n).^3);


m=[23324];
n=[26545];
flop_single = (max(m,n).*min(m,n).^2 - 1/3*min(m,n).^3);


flopcont = linspace(3.9e11,2.4e13,1000);
Gflops=flopcont*1e-12/1.3;


obj_single1= [[30.3332]
[5.1184]
[127.9447]
[19.4316]
[8.962]
[14.9832]
[33.6084]
[33.6659]
[25.3302]
[94.819]
[50.6015]
[31.8478]
[39.7525]
[23.5615]
[14.9301]
[4.7249]
[39.2624]
[127.2035]
[13.283]
[7.4347]
[43.4579]
[15.3983]
[66.3604]
[22.8402]
[28.4627]
[11.9121]
[6.6447]
[25.6822]
[24.0042]
[21.941]
[16.2487]
[6.8109]
[50.6732]
[5.1232]
[14.2083]
[17.5796]
[88.9034]
[46.7796]
[7.2491]
[10.2281]
[19.021]
[78.5112]
[40.6886]
[15.3224]
[47.7587]
[9.1368]
[22.7905]
[17.4145]
[21.104]
[123.133]
[4.2829]
[2.6092]
[7.85]
[4.7535]
[5.7166]
[3.2264]
[3.2743]
[4.2434]
[3.1858]
[6.8621]
[4.3513]
[17.3624]
[2.8926]
[4.0636]
[11.6552]
[4.7547]
[6.2543]
[3.3795]
[3.6121]
[4.9226]
[6.9123]
[4.333]
[3.3067]
[7.017]
[10.2587]
[8.0953]
[6.9048]
[4.1478]
[2.695]
[3.0534]
[4.2261]
[3.0462]
[3.4524]
[5.1041]
[6.5323]
[4.2049]
[3.3595]
[126.3577]
[6.3863]
[4.9065]
[4.7088]
[3.5118]
[3.3414]
[4.0031]
[3.5079]
[3.3827]
[2.3167]
[4.8538]
[3.334]
[3.8512]
];

obj_single1=[sort(obj_single1(1:end/2,1));obj_single1(1+end/2:end,1)]



ntask=10;

obj_multi_all1= [[1.3702]
[6.5246]
[2.0339]
[13.9461]
[1.9585]
[13.9374]
[29.2536]
[41.1536]
[10.3977]
[19.5714]
[37.5024]
[13.0769]
[18.6236]
[54.7632]
[14.4785]
[12.7883]
[11.7156]
[24.6745]
[22.3947]
[10.2616]
[44.8877]
[11.7739]
[8.8059]
[24.126]
[104.9646]
[31.5115]
[12.797]
[12.3754]
[21.0039]
[7.4402]
[0.9247]
[3.3093]
[3.4022]
[3.6753]
[1.8029]
[2.5583]
[10.5893]
[2.3864]
[0.7979]
[1.8619]
[10.7236]
[92.8273]
[15.602]
[32.7604]
[8.7618]
[5.9406]
[42.974]
[3.9385]
[18.808]
[31.869]
[1.1546]
[7.329]
[10.092]
[2.7391]
[7.2128]
[2.1866]
[1.121]
[1.1134]
[7.3862]
[6.4631]
[0.7539]
[6.3451]
[7.0879]
[1.5658]
[8.112]
[3.4386]
[2.306]
[1.007]
[11.5079]
[4.3746]
[0.9931]
[4.497]
[7.3785]
[2.7528]
[11.3781]
[3.1597]
[1.1111]
[0.83]
[7.8526]
[10.9815]
[0.8513]
[4.8999]
[10.9818]
[2.9916]
[11.3796]
[62.9811]
[1.8015]
[0.4119]
[12.628]
[1.5016]
[0.7412]
[6.0281]
[7.0476]
[1.0715]
[9.9746]
[17.4081]
[2.2814]
[1.0728]
[3.8112]
[31.3925]
];




obj = ones(ntask,2);
obj_rand = ones(ntask,1);
for jj=1:ntask
tmp=zeros(length(obj_multi_all1)/ntask,1);
for ii=1:length(tmp)/2
tmp(ii) = obj_multi_all1((jj-1)*length(tmp)/2+ii,1) ; 
end
obj_rand(jj)=min(tmp(1:length(tmp)/2));
for ii=1:length(tmp)/2
tmp(ii+length(tmp)/2) = obj_multi_all1(length(obj_multi_all1)/2+(ii-1)*ntask+jj,1); 
end

obj(jj,1)=min(tmp);
obj(jj,2)=max(tmp);
end




m = [3000 3500 4000 4500 5000 5500 6000 6500 7000 7500];
m_single = [ 7000];
opt_single = [ 9.41];



%% Plot figure 1
axisticksize = 48;
origin = [200,60];
markersize = 14;
LineWidth = 5;




figure(1)
barsize=14;
hd1=errorbar(flop,obj(:,1),[],obj(:,2)-obj(:,1),'o','MarkerSize',markersize,'MarkerFaceColor',[lineColors(4,:)],'LineWidth',LineWidth,'Color',[lineColors(4,:)]);
hd1.CapSize = barsize;
hold on
hd1=errorbar(flop_single*1.008^2,min(obj_single1(:)),[],max(obj_single1(:))-min(obj_single1(:)),'o','MarkerSize',markersize,'MarkerFaceColor',[lineColors(2,:)],'LineWidth',LineWidth,'Color',[lineColors(2,:)]);
hold on
hd1.CapSize = barsize;
hd1 = plot(flopcont,Gflops,'--k','MarkerSize',markersize,'MarkerFaceColor','k','LineWidth',LineWidth);
hold on

hd1 = plot(flopcont,Gflops/2.77,'-.k','MarkerSize',markersize,'MarkerFaceColor','k','LineWidth',LineWidth);
hold on

% hd1=plot(m,obj_rand,'x','MarkerSize',markersize,'MarkerFaceColor','k','LineWidth',LineWidth,'Color','k');
% hold on
% hd1=plot(m*1.008,obj_rand1,'x','MarkerSize',markersize,'MarkerFaceColor','k','LineWidth',LineWidth,'Color','k');
% hold on



set(gca, 'XScale','log', 'YScale','log')

% mlin = linspace(0,5000,100);
% nlin = linspace(0,5000,100);
% [M,N]=meshgrid(mlin,nlin);
% Z = griddata(m,n,opt,M,N,'cubic');
% mesh(M,N,Z,'FaceAlpha',0.5);
% hold on;



gca = get(gcf,'CurrentAxes');
set(gca,'YTick',[0.25 0.5 1 2 5 10 20 40 80])

% set(gca,'TickLabelInterpreter','none')
% xticklabels(xtick)
% xtickangle(45)
% xlim([3000,7300]);
% ylim([0,2000]);


legs = {};
legs{1,1} = ['$\delta=10$, $\epsilon_{tot}=10$'];
legs{1,2} = ['$\delta=1$, $\epsilon_{tot}=100$'];
legs{1,3} = ['1.3 TFLOPS'];
legs{1,4} = ['3.6 TFLOPS'];

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('runtime ($s$)');
ylabel(str,'interpreter','Latex')
str = sprintf('flop cnt');
xlabel(str,'interpreter','Latex')
str = sprintf('time (s)');
zlabel(str,'interpreter','Latex')


gca=legend(legs,'interpreter','Latex','color','none','NumColumns',1,'Location','northwest');
legend('boxoff')
set(gcf,'Position',[origin,1000,700]);

fig = gcf;

str = 'scalapack_qr_MLA_optimum_64nodes.eps';
saveas(fig,str,'epsc')



