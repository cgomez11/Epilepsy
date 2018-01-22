%Code for generating images
%datasetPath = 'C:\Users\catag\Documents\Maestria\Dataset';
%datasetPath = '/hpcfs/home/c.gomez10/inv/Dataset';
datasetPath = '/media/user_home2/EEG/Epilepsy';
dirDataset = dir(datasetPath);
%load('C:\Users\catag\Documents\Maestria\Epilepsy Project\201720\AllTimes_cut.mat');
load('/home/cgomez11/Project/Epilepsy/AllTimes_cut.mat');
%load('C:\Users\catag\Documents\Maestria\Epilepsy Project\201720\regNumber_cut.mat');
load('/home/cgomez11/Project/Epilepsy/regNumber_cut.mat');
rng('shuffle')
ww = 4;
srate = 256;
% pos_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
% neg_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
% pos_cont = 1;
% neg_cont = 1;

%%%%%%%%%%%%%%%%%%% NEW %%%%%%%%%%%%%%%%%%% 
N_seizures = 1;

for id = 3:numel(dirDataset) %recorrer pacientes
%for id = 8:8
    %neg_instances = {};
    neg_instances(:,:,1) = zeros(23, ww * srate);
    neg_instances(:,:,1) = [];
    %pos_instances = {};
    pos_instances(:,:,1) = zeros(23, ww * srate);
    pos_instances(:,:,1) = [];
    pos_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
    neg_rec_info = struct('patient_num', {''}, 'record',{''}, 't_start', {0}, 'InstanceNum', {0});
    pos_cont = 1;
    neg_cont = 1;
    
    patient = dirDataset(id).name;
    patientPath = fullfile(datasetPath,patient);
    %addpath(sprintf('C:\\Users\\catag\\Documents\\Maestria\\Dataset\\%s',patient))
    %addpath(sprintf('/hpcfs/home/c.gomez10/inv/Dataset/%s',patient))
    addpath(sprintf('/media/user_home2/EEG/Epilepsy/%s',patient))
    
    dataTimesPatient = allTimes{id-2}; %seizure times of each patient
    regNum = regNumber(:,id-2);
    regNum = regNum(~cellfun('isempty',regNum));
    reg = dir(fullfile(patientPath,'*.mat'));
    %fileNames = dir(sprintf('./Dataset/%s/*.edf',patient));
     %for j = 3:numel(regNum)
     for j = 1:numel(reg)    
     %for j =1:3    
        if dataTimesPatient{j} == 0 %no seizures
            continue
        elseif numel(dataTimesPatient{j}) <= 2*N_seizures %less than N seizures
            continue
        else %there are more than N seizures, then extract training instances
            %open the record
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
            %define ending time
            s_endingTime = dataTimesPatient{j}(2*N_seizures);
            t_endingTime = s_endingTime * s_SRate;
            
            %solo llegan los registros con mas de N crisis
            timesPatient = dataTimesPatient{j}*s_SRate;
            %crear señal binaria con 1's y 0's: solo extraer instancias
            %hasta la N crisis
            m_SignalsBinary = m_SignalsMat(1:t_endingTime);
            for i =1:2:2*N_seizures
                m_SignalsBinary(:,timesPatient(i):timesPatient(i+1))=1; %marcar periodos de crisis con 1's
            end
            m_SignalsBinary(m_SignalsBinary~=1)=0;
                
            %verificar que no haya eventos mixtos en un intervalo de T
            %dentro del while
            while t_endSam <= t_endingTime
                if sum(m_SignalsBinary(1,t_startSam:t_endSam)) == 0 %no hay crisis
                    tempSignal = m_SignalsMat(:,t_startSam:t_endSam);
                    %neg_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                    neg_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
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
                    %pos_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                    pos_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                    pos_rec_info(end+1).patient_num = patient;
                    pos_rec_info(end).record = fileName(1:end-4);
                    pos_rec_info(end).t_start = t_startSam;
                    pos_rec_info(end).InstanceNum = pos_cont;
                    %update varibales
                    pos_cont = pos_cont + 1;
                    t_startSam = t_startSam + floor(shift*s_SRate);
                    t_endSam = t_startSam + s_SzeWidthSam -1;
                    
                else %no cuenta para ninguna instancia:evento mixto
                    t_startSam = t_startSam + s_SzeWidthSam;
                    t_endSam = t_endSam + s_SzeWidthSam;
                    
                end
            end %ende del while
                
            %end
            fprintf('done register %s \n', fileName);
        end
     end %done with all the records of the patient
     
     %si hay instancias por guardar
     if isempty(neg_instances) && isempty(pos_instances) == 1
         continue

     else
         %submuestrear negativos
         ratio{id-2} = size(neg_instances,3)/size(pos_instances,3);
         vectRand = randi(size(neg_instances,3),1,size(pos_instances,3));
         neg_rec_info = neg_rec_info(vectRand);
         neg_instances = neg_instances(:,:,vectRand);
         train_instances = cat(3, pos_instances, neg_instances);
         pos_labels = ones(1, size(pos_instances,3));
         neg_labels = zeros(1, size(neg_instances,3));
         train_labels = cat(2, pos_labels, neg_labels);
         basePath = '/home/cgomez11/Project/Epilepsy/Dataset/DenseData/DataW4_025s_Adj/NewExpPerPatient';
         %basePath = 'C:\Users\catag\Documents\Maestria\Epilepsy Project\201810';
         %save(strcat(basePath,'redense_pos_instances_',patient,'.mat'),pos_instances, '-v7.3');
         %save(strcat(basePath,'redense_neg_instances_',patient,'.mat'),neg_instances, '-v7.3');
         save(strcat(basePath,'/train_instances_',patient,'.mat'), 'train_instances', '-v7.3');
         save(strcat(basePath,'/train_labels_',patient,'.mat'), 'train_labels', '-v7.3');
         save(strcat(basePath,'/redense_pos_info_',patient,'.mat'), 'pos_rec_info', '-v7.3');
         save(strcat(basePath,'/redense_neg_info_',patient,'.mat'), 'neg_rec_info', '-v7.3');
         clear neg_rec_info
         clear pos_rec_info
         clear neg_instances
         clear pos_instances
         clear train_labels pos_labels neg_labels train_instances
     end
     
end
