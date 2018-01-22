%Code for generating images
%datasetPath = 'C:\Users\catag\Documents\Maestria\Dataset';
datasetPath = '/hpcfs/home/c.gomez10/inv/Dataset';
dirDataset = dir(datasetPath);
load('AllTimes_cut.mat');
load('regNumber_cut.mat');
rng('shuffle')
ww = 4;
srate = 256;
% pos_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
% neg_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
% pos_cont = 1;
% neg_cont = 1;

for id = 3:numel(dirDataset) %recorrer pacientes
%for id = 3:3
    %neg_instances(:,:,1) = zeros(23, ww * srate);
    %neg_instances(:,:,1) = [];
    %pos_instances(:,:,1) = zeros(23, ww*srate);
    %pos_instances(:,:,1) = [];
    neg_instances = {};
    pos_instances = {};
    pos_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
    neg_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
    pos_cont = 1;
    neg_cont = 1;
    
    patient = dirDataset(id).name;
    patientPath = fullfile(datasetPath,patient);
    %addpath(sprintf('C:\\Users\\catag\\Documents\\Maestria\\Dataset\\%s',patient))
    addpath(sprintf('/hpcfs/home/c.gomez10/inv/Dataset/%s',patient))
   
    dataTimesPatient = allTimes{id-2};
    regNum = regNumber(:,id-2);
    regNum = regNum(~cellfun('isempty',regNum));
    reg = dir(fullfile(patientPath,'*.mat'));
    %fileNames = dir(sprintf('./Dataset/%s/*.edf',patient));
     %for j = 3:numel(regNum)
     for j = 1:numel(reg)    
     %for j =1:3    
        fileName = reg(j).name;
        %fileName = strcat(patient,'_',regNum{j});
        load(fileName)
        %[st_Header, m_SignalsMat] = edfread(fileName);
        mean_per_channel = mean(m_SignalsMat,2);
        m_SignalsMat = bsxfun(@minus, m_SignalsMat, mean_per_channel);
        %ignore channel 24
        %m_SignalsMat = m_SignalsMat(1:23,:);
        s_SRate = 256;
        
        s_SizeWidthSec = 4;
        shift = 1/8; %corrimiento en segundos de la ventana
        s_SzeWidthSam = round(s_SizeWidthSec * s_SRate);
        t_startSam = 1;
        t_endSam = s_SzeWidthSam;
        
        %verificar que no haya eventos mixtos
        if dataTimesPatient{j}(1) == 0 %no hay crisis en ese registro
            
            while t_endSam <= size(m_SignalsMat,2)
                neg_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_label{end+1} = 0;
                neg_rec_info(end+1).patient_num = patient;
                neg_rec_info(end).record = fileName(1:end-4);
                neg_rec_info(end).t_start = t_startSam;
                neg_rec_info(end).InstanceNum = neg_cont;
                %update variables
                neg_cont = neg_cont + 1;
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
            end
            
        else %sí hay crisis en el registro
            timesPatient = dataTimesPatient{j}*s_SRate;
            %crear señal binaria con 1's y 0's
            m_SignalsBinary = m_SignalsMat;
            for i =1:2:numel(timesPatient)
                m_SignalsBinary(:,timesPatient(i):timesPatient(i+1))=1; %marcar periodos de crisis con 1's
            end
            m_SignalsBinary(m_SignalsBinary~=1)=0;
            
            %verificar que no haya eventos mixtos en un intervalo de T
            %dentro del while
            while t_endSam <= size(m_SignalsMat,2)
            if sum(m_SignalsBinary(1,t_startSam:t_endSam)) == 0 %no hay crisis
                tempSignal = m_SignalsMat(:,t_startSam:t_endSam);
                neg_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_label{end+1} = 0;
                neg_rec_info(end+1).patient_num = patient;
                neg_rec_info(end).record = fileName(1:end-4);
                neg_rec_info(end).t_start = t_startSam;
                neg_rec_info(end).InstanceNum = neg_cont;
                %update variables
                neg_cont = neg_cont + 1;
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
                
            elseif sum(m_SignalsBinary(1,t_startSam:t_endSam)) == s_SzeWidthSam %solo crisis
                pos_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                %pos_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                pos_rec_info(end+1).patient_num = patient;
                pos_rec_info(end).record = fileName(1:end-4);
                pos_rec_info(end).t_start = t_startSam;
                pos_rec_info(end).InstanceNum = pos_cont;
                %update varibales
                pos_cont = pos_cont + 1;
                t_startSam = t_startSam + floor(shift*s_SRate);
                t_endSam = t_startSam + s_SzeWidthSam -1;
                
            else %no cuenta para ninguna instancia
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
                
            end
            end             
                 
        end
    fprintf('done register %s \n', fileName);    
     end        
     %submuestrear al final de cada paciente
     %ratio = round(numel(neg_instances)/numel(pos_instances));
     %ratio = round(size(neg_instances,3)/size(pos_instances,3));
     %vect = 1:ratio:numel(neg_instances);
     %select the negative instances randomly
     ratio{id-2} = numel(neg_instances)/numel(pos_instances);
     vectRand = randi(numel(neg_instances),1,numel(pos_instances));
     neg_recs{id-2} = neg_rec_info(vectRand);
     neg_instances_pat{id-2} = neg_instances(vectRand);
     pos_instances_pat{id-2} = pos_instances;
     pos_recs{id-2} = pos_rec_info;
%      save('pos_instances_pat2.mat','pos_instances_pat');
%      save('neg_instances_pat2.mat','neg_instances_pat');
    clear neg_rec_info
    clear pos_rec_info
    clear neg_instances
    clear pos_instances    
end 


save('redense_pos_instances_pat.mat','pos_instances_pat');
save('redense_neg_instances_pat.mat','neg_instances_pat');
save('redense_pos_recs.mat', 'pos_recs');
save('redense_neg_recs.mat', 'neg_recs');