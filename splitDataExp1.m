%prepare raw data for BOW
%addpath /hpcfs/home/c.gomez10/inv
load('pos_instances_pat.mat');
load('neg_instances_pat.mat');

ww = 4;
srate = 256;
train_instances ={};
val_instances = {};
test_instances = {};
train_mat(:,:,1) = zeros(23, ww * srate);
train_mat(:,:,1) = [];

val_mat(:,:,1) = zeros(23, ww * srate);
val_mat(:,:,1) = [];

test_mat(:,:,1) = zeros(23, ww * srate);
test_mat(:,:,1) = [];

for i = 1:numel(pos_instances_pat)
    NP = round(numel(pos_instances_pat{i})/3);
    NN = round(numel(neg_instances_pat{i})/3);
    train_instances{end+1} = pos_instances_pat{i}(1:NP);
    train_instances{end+1} = neg_instances_pat{i}(1:NN);
    
    val_instances{end+1} = pos_instances_pat{i}(NP+1:NP*2);
    val_instances{end+1} = neg_instances_pat{i}(NN+1:NN*2);
    
    test_instances{end+1} = pos_instances_pat{i}(NP*2+1:end);
    test_instances{end+1} = neg_instances_pat{i}(NN*2+1:end);
end 

train_label = [];

for i =1:numel(train_instances) %24 pacientes X 2 (pos luego neg)
    for j=1:numel(train_instances{i})
        train_mat(:,:,end+1) = train_instances{i}{j};
        if mod(i,2) == 0 %par = negativos
            train_label(end+1) = 0;
        else
            train_label(end+1) = 1;
        end
    end
end

val_label = [];
for i =1:numel(val_instances) %24 pacientes X 2 (pos luego neg)
    for j=1:numel(val_instances{i})
        val_mat(:,:,end+1) = val_instances{i}{j};
        if mod(i,2) == 0 %par = negativo
            val_label(end+1) = 0;
        else
            val_label(end+1) = 1;
        end
    end
end

test_label = [];
for i =1:numel(test_instances) %24 pacientes X 2 (pos luego neg)
    for j=1:numel(test_instances{i})
        test_mat(:,:,end+1) = test_instances{i}{j};
        if mod(i,2) == 0 %par = negativo
            test_label(end+1) = 0;
        else
            test_label(end+1) = 1;
        end
    end
end

save('train_mat.mat','train_mat')
save('train_label.mat','train_label')

save('val_mat.mat','val_mat')
save('val_label.mat','val_label')

save('test_mat.mat','test_mat')
save('test_label.mat','test_label')

all_data = cat(3,train_mat, val_mat);
all_data = cat(3,all_data, test_mat);
save('all_data.mat','all_data','-v7.3')

%Normalization
%all_data_norm = zscore(all_data,1,3); %normalized matrix with all data
%min_abs = min(all_data_norm(:));
%res = bsxfun(@minus, all_data_norm, min_abs);
%max_new = max(res(:));
%fact = max_new/255;
%scaled_data = bsxfun(@rdivide, res, fact);

%% 



