%extraer registros completos de pacientes de test y sus etiquetas
datasetPath = 'C:\Users\catag\Documents\Maestria\Dataset';
%datasetPath = '/hpcfs/home/c.gomez10/inv/Dataset';
dirDataset = dir(datasetPath);
load('AllTimes_cut.mat');
load('regNumber_cut.mat');
test_patients = [16, 14, 19, 17, 13, 1, 24, 15];

%for id = 1:numel(test_patients) %recorrer pacientes
%for id = 3:numel(dirDataset) %recorrer pacientes   
for id = 3:3
    %pat_records = {};
    patient = dirDataset(id).name;
    patientPath = fullfile(datasetPath,patient);
    addpath(sprintf('C:\\Users\\catag\\Documents\\Maestria\\Dataset\\%s',patient))
    %addpath(sprintf('/hpcfs/home/c.gomez10/inv/Dataset/%s',patient))
    
    dataTimesPatient = allTimes{id-2};
    regNum = regNumber(:,id-2);
    regNum = regNum(~cellfun('isempty',regNum));
    reg = dir(fullfile(patientPath,'*.mat'));
    
    for j = 1:numel(reg) %recorrer todos los registros
        fileName = reg(j).name;
        load(fileName)
        mean_per_channel = mean(m_SignalsMat,2);
        m_SignalsMat = bsxfun(@minus, m_SignalsMat, mean_per_channel);
        s_SRate = 256;
        s_Step = 64;
        t_startSam = 1;
        t_endSam = s_Step;
        sub_sampled_labels ={};
        if j == 1
            labels_pat(:,:,1) = zeros(1, size(m_SignalsMat,2)/s_Step);
            labels_pat(:,:,1) = [];
            records_pat(:,:,1) =  zeros(23, size(m_SignalsMat,2));
            records_pat(:,:,1) = [];
        end 
        
        %generate labels for all the record
        if dataTimesPatient{j}(1) == 0 %no hay crisis en el registro: llenar con 0s
            m_SignalsBinary = zeros(1, size(m_SignalsMat,2));
            sub_sampled_labels{end+1} = zeros(1,size(m_SignalsMat,2)/s_Step);
           
        else
            timesPatient = dataTimesPatient{j}*s_SRate; 
            %crear señal binaria con 1's y 0's
            m_SignalsBinary = m_SignalsMat;
            for i =1:2:numel(timesPatient)
                m_SignalsBinary(:,timesPatient(i):timesPatient(i+1))=1; %marcar periodos de crisis con 1's
            end
            m_SignalsBinary(m_SignalsBinary~=1)=0;
            m_SignalsBinary = m_SignalsBinary(1,:);
        
            while t_endSam <= size(m_SignalsBinary,2)
                sub_segment = m_SignalsBinary(t_startSam:t_endSam);
                suma_ptos = sum(sub_segment);
                if suma_ptos == 0 
                    sub_sampled_labels{end+1} = 0;
                elseif suma_ptos == s_Step
                    sub_sampled_labels{end+1} = 1;
                elseif suma_ptos/s_Step > 0.7
                    sub_sampled_labels{end+1} = 1;
                else
                    sub_sampled_labels{end+1} = 0;
                end
                t_startSam = t_startSam + s_Step;
                t_endSam = t_endSam + s_Step;
                clear sub_segment   
            end 
        end 
        %done with the record
        sub_sampled_labels = cell2mat(sub_sampled_labels);
        labels_pat(:,:,end+1) = sub_sampled_labels;
        records_pat(:,:,end+1) = m_SignalsMat;
        fprintf('done register %s \n', fileName);
        clear m_SignalsBinary
        clear sub_sampled_labels
    end 
    save(strcat('labels_pat_',patient,'.mat'), 'labels_pat');
    save(strcat('records_pat_',patient,'.mat'), 'records_pat');
    %all_pat_records{id-2} = pat_records;
    %clear records_pat
    %clear labels_pat
end 


%hacer escalamiento: con respecto a solo los de test? o hay que generar los
%registros de todos los pacientes para escalar