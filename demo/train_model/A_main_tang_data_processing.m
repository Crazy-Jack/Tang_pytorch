clear;clc;

% A_visualization_image_modified， 去掉%可以处理vis imgs
path='C:\Users\J\Desktop\';
path=createfolders(path,'tang_imgs_results\') 
Rimgs_path = 'I:\GS\Go_abroad\results\CMU\course_research\Research\tangdata\V1_Data\M151023\Site1\val\valPics\';
Rimgs_savepath =[path];%TODO
V_imgs_datapath='C:\Users\J\Desktop\tang_results\visualization_results_m2s1\';
V_save_path=[path];%TODO
pred=load("C:\Users\J\Desktop\tang_results\M2S1_results_from_cluster\train_model\TCpred_ep500_matfile\TCpred_all_cell_m2s1.mat");
real=load("C:\Users\J\Desktop\tang_results\_m2s1_ep500_dataProcessing\TCreal_validation_m2s1.mat");
predrsp=pred.TCunderprocess;
realrsp=(real.valRsp)';
[Correlation,VE]=getcorr_and_ve(predrsp,realrsp);
% % % % % % % % % 获取top real images
Neuronnum=1:size(realrsp,2);
% get_top_images(Neuronnum,Rimgs_path,Rimgs_savepath,realrsp,predrsp,Correlation,VE);

% get_sorted_tuning_curve(realrsp,predrsp,Correlation,VE,'C:\Users\J\Desktop\tang_imgs_results\')
vis_path='C:\Users\J\Desktop\tang_results\visualization_results_m2s1_modified\visulization_results_m2s1\visualization_m2s1_annotated\';

mov_vis_img(vis_path,Neuronnum,path)

function mov_vis_img(vis_path,Neuronnum,savepath)
for i=Neuronnum
o=dir([vis_path,'Cell_',num2str(i),'_*']);o2=o.name;
s=dir([savepath,'Cell_',num2str(i),'_*']);s2=s.name;
copyfile([vis_path,o2],[savepath,s2],'f')
fprintf('save visualization imgs, Neuron: %d \n',i);
end
end

function get_sorted_tuning_curve(realrsp,predrsp,Correlation,VE,savepath)
% % % % % % % 
% Sorted tuning curve, for each neurons in m2s1
for i=1:299 % TODO
% data prepared and processed
g=realrsp(:,i);[realrspsorted,realindex]=sort(g,'descend');
[predsorted,predindex]=sort(predrsp(:,i),'descend');
HMreal=max(g)./2;
HMpred=max(predrsp(:,i))./2;
gg=realrspsorted(realrspsorted>=HMreal);
realnum=length(gg);  % response number that are larger than HM.
prednum=length(predsorted(predsorted>=HMpred));
% plot figures
figure('visible','off'); plot(realrspsorted,'m');hold on
plot(HMreal,'r*'); plot(gg,'k'); plot(gg,'kx'); plot(predrsp(realindex))
legend('sorted-real response',['HM-point, Value: ' num2str(HMreal)],'real response ≥ HM','real response ≥ HM','pred response with real sorted index')
title({['Neuron:' num2str(i) ', ' 'Site: M2S1' ', Sorted tuning curve'];...
    ['Corr(real,pred):' num2str(Correlation(i)),', VE(real,pred): ' num2str(VE(i))];...
    ['RealRspNum>=HMreal:',num2str(realnum),', PredRspNum>=HMpred:',num2str(prednum)]})
xlabel('Fake img index')
ylabel('response intensity: dF/F')
hold off
f = gcf;
k=dir([savepath,'Cell_',num2str(i),'_*']);k2=k.name;
exportgraphics(f,[savepath,k2,'\','Neuron_',num2str(i),'_sorted tuning curve.png'],'Resolution',1000)
fprintf('Finish get the img of Neuron: %s \n',num2str(i))
end
% % % % % % % % 
end


function [numNRStHM,a]=get_top_images(Neuronnum,Rimgs_path,Rimgs_savepath,realrsp,predrsp,Correlation,VE)
for Nnum=Neuronnum
[HM,LAEHM,LAEindex]=get_image(Rimgs_path,Rimgs_savepath,Nnum,realrsp,'Real',Correlation,VE);
end

for i=Neuronnum
   [HM,LAEHM,LAEindex] = get_image(Rimgs_path,Rimgs_savepath,i,predrsp,'Pred',Correlation,VE);
    a(i,:)=length(LAEHM);
end
numNRStHM=a(a<=10); % numNRStHM:in the m2s1 prediction, there are 19 neurons' response(which > Half maximum) that are lower than (<=) 10
end

function [HM,LAEHM,LAEindex]=get_image(img_org_path,path,neuronnum,rsp_data_matrix,RealorPred,Correlation,VE)
% % create folders, contain real, and pred folders
cd(path);
i = neuronnum;
folderName{i} = ['Cell_', num2str(i),'_Corr=',num2str(Correlation(neuronnum)),'_VE=',num2str(VE(neuronnum))];
mkdir(folderName{i});
path=[path,folderName{neuronnum},'\'];
cd(path);subfoldername={'top10real','top10pred'};mkdir(subfoldername{1});mkdir(subfoldername{2});
pathreal=[subfoldername{1},'\'];pathpred=[subfoldername{2},'\'];% TODO, this name paring with the LAEHM
if RealorPred=='Real'
    subpath=pathreal;
else
    subpath=pathpred;
end
% ————————————————————————————
% find the corresponding images
[rspsort,index]=sort(rsp_data_matrix(:,neuronnum),'descend');
HM=max(rspsort)./2;
% LAEHM=rspsort(rspsort>=HM)%TODO if we need HM threshold, we use this LAE means: ≥
LAEHM=rspsort(1:10);%TODO  if we need top 10, we use this
judge=(LAEHM>=HM);%TODO, judge means, if response>=HM, it will be 1, or it will be 0
LAEindex=index(1:length(LAEHM));
str2=num2str(LAEindex(:)); str1='.bmp';
k=strcat(str2,str1);
% copy images we want
for i=1:size(k,1)
g_real=strrep(k(i,:),' ','');
spath = [img_org_path,g_real];
dpath =[path,subpath,'Cell_',num2str(neuronnum),'_',RealorPred,'_Rsp_',num2str(round(rspsort(i),4))...
    ,'_Judge',num2str(judge(i)),'_index_',num2str(g_real)];
copyfile(spath,dpath);
% save HM value
save([path,subpath,'Cell_',num2str(neuronnum),'_',RealorPred,'_HM_',num2str(round(HM,4))],"HM")
end
end

function [Correlation,VE]=getcorr_and_ve(predrsp,realrsp)
for i=1:size(predrsp,2)
    k=corrcoef(predrsp(:,i),realrsp(:,i));
    g=k(1,2);
    Correlation(i,:)=g;
end
for i=1:size(predrsp,2)
    k=1-var(predrsp(:,i)-realrsp(:,i))./var(realrsp(:,i));
    VE(i,:)=k;
end
Correlation=Correlation;VE=VE;
end


function path=createfolders(mainpath,folders_names) 
cd(mainpath);mkdir(folders_names);
path=strcat(mainpath,folders_names);
end