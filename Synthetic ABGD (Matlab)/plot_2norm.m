clear all
clc

cases_list = {'kvsn','pvsn','n'};

sigma=0;
cases = cases_list{2};
fname = ['./Results/res_' 'sigma' num2str(sigma) cases '.mat']; 

load(fname); %or res_SNR20


% plot graph for sfnorm 
algs = {'am','gd','sgd'};

figure(1)

 for idx_alg = 1:numel(param.algs)
     
      subplot(1,numel(param.algs),idx_alg); 
      plotmtx = log10(median(res{idx_alg}.error_2norm,3)); 
   
     
     imagesc((flipud(plotmtx)),[-2.5 0]);
     hold on
     axis image; 
%      colormap(gray(255));  
     colormap('jet'); 
 
    set(gca,'Xtick',1+(0:1:size(plotmtx,2)),'XTickLabel',{'1','','3','','5','','7','','9',''},'fontsize',24,'fontname','Times New Roman');
    set(gca,'Ytick',0.5+(0:size(plotmtx,1)/19:size(plotmtx,1)),'YTickLabels',fliplr({'','100','','200','','300','','400','','500','','600','','700','','800','','900','','1000'
    }),'fontsize',24,'fontname', 'Times New Roman');  %,'750','800','850','900','950','1000'
    colorbar
 
 
     colorbar
 
     ylabel('$L$','interpreter','LaTeX');
     xlabel('$K$','interpreter','LaTeX');
     xtickangle(0);
     
      title(algs{idx_alg}); 
  
 end


 
