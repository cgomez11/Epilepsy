% Train and Test classificator evaluation
tic;

clear all;close all;clc;

run('../vlfeat-0.9.20/toolbox/vl_setup');
% run('vlfeat-0.9.20\toolbox\vl_setup');


folder_train=fullfile('..','database','Train');
folder_test=fullfile('..','database','Test');
folderImages_train=fullfile('..','database','TrainImages');
folderImages_test=fullfile('..','database','TestImages');

load(fullfile(folderImages_train,'TrainHyL.mat'));
load(fullfile(folderImages_test,'TestHyL.mat'));

%load('TrainHyL.mat');
%load('TestHyL.mat');
%% Train SVM

lambda=1/(10*length(TrainLabels));
%lambda=1/100;

fprintf('\n\n-------Start Training models---------\n\n')
%Modelo para W
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=0)=-1;
TrainLabelAux(TrainLabelAux==0)=1;
[W_W,B_W]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%modelW=fitcsvm(TrainHists,TrainLabelAux);
%modelW=fitPosterior(modelW,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model W---------\n\n')
%Modelo para 1
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=1)=-1;
[W_1,B_1]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%model1=fitcsvm(TrainHists,TrainLabelAux);
%model1=fitPosterior(model1,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model 1---------\n\n')
%Modelo para 2
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=2)=-1;
TrainLabelAux(TrainLabelAux==2)=1;
[W_2,B_2]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%model2=fitcsvm(TrainHists,TrainLabelAux);
%model2=fitPosterior(model2,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model 2---------\n\n')
%Modelo para 3
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=3)=-1;
TrainLabelAux(TrainLabelAux==3)=1;
[W_3,B_3]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%model3=fitcsvm(TrainHists,TrainLabelAux);
%model3=fitPosterior(model3,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model 3---------\n\n')
%Modelo para 4
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=4)=-1;
TrainLabelAux(TrainLabelAux==4)=1;
[W_4,B_4]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%model4=fitcsvm(TrainHists,TrainLabelAux);
%model4=fitPosterior(model4,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model 4---------\n\n')
%Modelo para R
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=5)=-1;
TrainLabelAux(TrainLabelAux==5)=1;
[W_R,B_R]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%modelR=fitcsvm(TrainHists,TrainLabelAux);
%modelR=fitPosterior(modelR,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model R---------\n\n')
%Modelo para artifactos
TrainLabelAux=TrainLabels;
TrainLabelAux(TrainLabelAux~=6)=-1;
TrainLabelAux(TrainLabelAux==6)=1;
[W_Art,B_Art]=vl_svmtrain(TrainHists',TrainLabelAux',lambda,'Solver','sdca');
%modelE=fitcsvm(TrainHists,TrainLabelAux);
%modelE=fitPosterior(modelE,TrainFeature,TrainLabelAux);
fprintf('\n\n-------Model Artifacts---------\n\n')

save('SVMModels.mat','W_W','B_W','W_1','B_1','W_2','B_2','W_3','B_3'...
    ,'W_4','B_4','W_R','B_R','W_Art','B_Art');

%% Test Models
%load('SVMModels.mat');

fprintf('\n\n-------Start Testing Models---------\n\n')

%Test en modelW
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=0)=-1;
%TestLabelAux(TrainLabelAux==0)=1;
%[predictedLabelW,probEstimatesW]=predict(modelW,TestHists);

scoresW=W_W'*TestHists' + B_W;
fprintf('\n\n-------Model W Tested---------\n\n')
%Test en model1
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=1)=-1;
%TestLabelAux(TrainLabelAux==1)=1;
%[predictedLabel1,probEstimates1]=predict(model1,TestHists);

scores1=W_1'*TestHists' + B_1;
fprintf('\n\n-------Model 1 Tested---------\n\n')
%Test en model2
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=2)=-1;
%TestLabelAux(TrainLabelAux==2)=1;
%[predictedLabel2,probEstimates2]=predict(model2,TestHists);

scores2=W_2'*TestHists' + B_2;
fprintf('\n\n-------Model 2 Tested---------\n\n')
%Test en model3
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=3)=-1;
%TestLabelAux(TrainLabelAux==3)=1;
%[predictedLabel3,probEstimates3]=predict(model3,TestHists);

scores3=W_3'*TestHists' + B_3;
fprintf('\n\n-------Model 3 Tested---------\n\n')
%Test en model4
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=4)=-1;
%TestLabelAux(TrainLabelAux==4)=1;
%[predictedLabel4,probEstimates4]=predict(model4,TestHists);

scores4=W_4'*TestHists' + B_4;
fprintf('\n\n-------Model 4 Tested---------\n\n')
%Test en modelR
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=5)=-1;
%TestLabelAux(TrainLabelAux==5)=1;
%[predictedLabelR,probEstimatesR]=predict(modelR,TestHists);

scoresR=W_R'*TestHists' + B_R;
fprintf('\n\n-------Model R Tested---------\n\n')
%Test en modelE
%TestLabelAux=TestLabels;
%TestLabelAux(TrainLabelAux~=6)=-1;
%TestLabelAux(TrainLabelAux==6)=1;
%[predictedLabelE,probEstimatesE]=predict(modelE,TestHists);

scoresArt=W_Art'*TestHists' + B_Art;
fprintf('\n\n-------Model E Tested---------\n\n')

%% Scores
ScoresMat=[scoresW;scores1;scores2;scores3;scores4;scoresR;scoresArt];

[maxs,resultLabels]=max(ScoresMat,[],1);

resultLabels=resultLabels-1;

ConfMatrix=confusionmat(TestLabels',resultLabels);
ACA=trace(ConfMatrix)/sum(sum(ConfMatrix));

display(ConfMatrix);
fprintf('\n---- ACA vlfeat: %.f ',ACA*100);


%ResultProbab=[probEstimatesW,probEstimates1,probEstimates2,...
%    probEstimates3,probEstimates4,probEstimatesR,probEstimatesE];

%[~,ResultLabels]=max(ResultProbab,[],2);
%ResultLabels=ResultLabels-1;
%fprintf('\n\n-------Final Test Results Labels Obtained---------\n\n')

%ConfMatrix=confusionmat(TestLabel,ResultLabels);
%ACA=trace(confMatrix)/sum(sum(confMatrix));

%display(ConfMatrix);
%fprintf('\n---- ACA matlab: %.f ',ACA*100);