% Extract descriptors from train to baseline idea
    tic;

clear all;close all;clc;

run('../vlfeat-0.9.20/toolbox/vl_setup');


folder_train=fullfile('..','database','Train');
folder_test=fullfile('..','database','Test');
folderImages_train=fullfile('..','database','TrainImages');
folderImages_test=fullfile('..','database','TestImages');
% folder_results=fullfile('.','TextonsResults');

dir_train=dir(folder_train);
dir_train=dir_train(3:end);
dir_test=dir(folder_test);
dir_test=dir_test(3:end);

% Concatenate all images from train
% load('TrainImages.mat');

% load(fullfile(folderImages_train,'TrainImages.mat'));
% TrainMat=[];%For Kmeans
% AllTrain=[];%Train images cated at dim=3
% TrainLabels=[];
% 
% %Labels:
% % Classes:
% % awake = 'W' -----------------> 0
% % NREM Sleep Stage 1 = '1' ----> 1
% % NREM Sleep Stage 2 = '2' ----> 2
% % NREM Sleep Stage 3 = '3' ----> 3
% % NREM Sleep Stage 4 = '4' ----> 4
% % REM Sleep Stage = 'R' -------> 5
% % Movement_time = 'e','?'------> 6
% 
% fprintf('\n\n ---- Start Catenation for Train ---- \n\n');
% 
% for i=1:size(ImagesW,3)
%     image=ImagesW(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(ImagesW,3)));
% fprintf('\n ---- W Cated ---- \n');
% 
% for i=1:size(Images1,3)
%     image=Images1(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(Images1,3))+1);
% fprintf('\n ---- 1 Cated ---- \n');
% 
% for i=1:size(Images2,3)
%     image=Images2(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(Images2,3))+2);
% fprintf('\n ---- 2 Cated ---- \n');
% 
% for i=1:size(Images3,3)
%     image=Images3(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(Images3,3))+3);
% fprintf('\n ---- 3 Cated ---- \n');
% 
% for i=1:size(Images4,3)
%     image=Images4(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(Images4,3))+4);
% fprintf('\n ---- 4 Cated ---- \n');
% 
% for i=1:size(ImagesR,3)
%     image=ImagesR(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(ImagesR,3))+5);
% fprintf('\n ---- R Cated ---- \n');
% 
% for i=1:size(ImagesArt,3)
%     image=ImagesArt(:,:,i);
%     TrainMat=cat(1,TrainMat,image');
%     AllTrain=cat(3,AllTrain,image);
% end
% TrainLabels=cat(2,TrainLabels,zeros(1,size(ImagesArt,3))+6);
% fprintf('\n ---- Art Cated ---- \n');
% 
% fprintf('\n\n ---- Done Catenation for Train ---- \n\n');
% %% Concatenate all images from test
% fprintf('\n\n ---- Start Catenation for Test ---- \n\n');
% load(fullfile(folderImages_test,'TestImages.mat'));
% 
% AllTest=[];%Train images cated at dim=3
% TestLabels=[];
% 
% for i=1:size(ImagesW,3)
%     image=ImagesW(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(ImagesW,3)));
% fprintf('\n ---- W Cated ---- \n');
% 
% for i=1:size(Images1,3)
%     image=Images1(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(Images1,3))+1);
% fprintf('\n ---- 1 Cated ---- \n');
% 
% for i=1:size(Images2,3)
%     image=Images2(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(Images2,3))+2);
% fprintf('\n ---- 2 Cated ---- \n');
% 
% for i=1:size(Images3,3)
%     image=Images3(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(Images3,3))+3);
% fprintf('\n ---- 3 Cated ---- \n');
% 
% for i=1:size(Images4,3)
%     image=Images4(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(Images4,3))+4);
% fprintf('\n ---- 4 Cated ---- \n');
% 
% for i=1:size(ImagesR,3)
%     image=ImagesR(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(ImagesR,3))+5);
% fprintf('\n ---- R Cated ---- \n');
% 
% for i=1:size(ImagesArt,3)
%     image=ImagesArt(:,:,i);
%     AllTest=cat(3,AllTest,image);
% end
% TestLabels=cat(2,TestLabels,zeros(1,size(ImagesArt,3))+6);
% fprintf('\n ---- Art Cated ---- \n');
% 
% fprintf('\n\n ---- Done Catenation for Test ---- \n\n');
% 
% toc;
%% Kmeans
%tic;
% TrainMat=TrainMat';
% TrainLabels=TrainLabels';
% TestLabels=TestLabels';
% AllTrain(4:7,:,:)=[];
% AllTest(4:7,:,:)=[];
 
% save(fullfile(folderImages_train,'TrainMat.mat'),'TrainMat');
% save(fullfile(folderImages_train,'AllTrain.mat'),'AllTrain','TrainLabels');
% save(fullfile(folderImages_test,'AllTest.mat'),'AllTest','TestLabels');

load(fullfile(folderImages_train,'TrainMat.mat'));
load(fullfile(folderImages_train,'AllTrain.mat'));
load(fullfile(folderImages_test,'AllTest.mat'));

TrainMat(4:7,:)=[];
AllTrain(4:7,:,:)=[];
AllTest(4:7,:,:)=[];

%size(TrainMat)
%size(AllTrain)
%size(AllTest)

fprintf('\n\n ---- Start K-MEANS ---- \n\n');
K=100;%Numero de clusters para textones
[kcenters,klabels]=vl_kmeans(double(TrainMat),K);
fprintf('\n\n ---- Finish K-MEANS ---- \n\n');

save(fullfile(folderImages_train,'kmeansResults.mat'),'kcenters','klabels');
%% Asignar etiquetas con vl_alldist a Train

% load(fullfile(folderImages_train,'kmeansResults.mat'));

fprintf('\n\n ---- Start Histograms Asignation ---- \n\n');
TrainHists=zeros(size(AllTrain,3),K);
parfor j=1:size(AllTrain,3)
    [~, values] = min(vl_alldist(double(AllTrain(:,:,j)), kcenters),[],2);
    [N,~] = histcounts(values,K);
    TrainHists(j,:)=N;
end

TestHists=zeros(size(AllTest,3),K);
parfor j=1:size(AllTest,3)
    [~, values] = min(vl_alldist(double(AllTest(:,:,j)), kcenters),[],2);
    [N,~] = histcounts(values,K);
    TestHists(j,:)=N;
end
fprintf('\n\n ---- Finish Histograms Asignation ---- \n\n');

save(fullfile(folderImages_train,'TrainHyL.mat'),'TrainHists','TrainLabels');
save(fullfile(folderImages_test,'TestHyL.mat'),'TestHists','TestLabels');

save(fullfile(folderImages_train,'Train_cated.mat'),'AllTrain');
save(fullfile(folderImages_test,'Test_cated.mat'),'AllTest');

save(fullfile(folderImages_train,'Train_labels.mat'),'TrainLabels');
save(fullfile(folderImages_test,'Test_labels.mat'),'TestLabels');

toc;
