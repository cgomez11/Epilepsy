%extraer registros completos de pacientes de test y sus etiquetas
%datasetPath = '/hpcfs/home/c.gomez10/inv/Dataset';
datasetPath = '/media/user_home2/EEG/Epilepsy'
dirDataset = dir(datasetPath);
load('AllTimes_cut.mat');
load('regNumber_cut.mat');
%test_patients = [16, 14, 19, 17, 13, 1, 24, 15];
%test_patients = [1];

%for id = 1:numel(test_patients) %recorrer pacientes
%for id = 3:numel(dirDataset)%recorrer pacientes   
for id =3:3
    pat_labels = {};
    pat_records = {};
    patient = dirDataset(id).name;
    patientPath = fullfile(datasetPath,patient);
    %addpath(sprintf('/hpcfs/home/c.gomez10/inv/Dataset/%s',patient))
    addpath(sprintf('/media/user_home2/EEG/Epilepsy/%s',patient))
    
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
        
        if dataTimesPatient{j}(1) == 0 %no hay crisis en el registro: llenar con 0s
            m_SignalsBinary = zeros(1, size(m_SignalsMat,2));
           
        else
            timesPatient = dataTimesPatient{j}*s_SRate; 
            %crear señal binaria con 1's y 0's
            m_SignalsBinary = m_SignalsMat;
            for i =1:2:numel(timesPatient)
                m_SignalsBinary(:,timesPatient(i):timesPatient(i+1))=1; %marcar periodos de crisis con 1's
            end
            m_SignalsBinary(m_SignalsBinary~=1)=0;
            m_SignalsBinary = m_SignalsBinary(1,:);
        end
        pat_labels{end+1} = m_SignalsBinary; %para cada registro
        pat_records{end+1} = m_SignalsMat;
        fprintf('done register %s \n', fileName);
    end 
    all_pat_labels{id-2} = pat_labels;
    all_pat_records{id-2} = pat_records;
    clear pat_records
    clear pat_labels
    
end 


%hacer escalamiento: con respecto a solo los de test? o hay que generar los
%registros de todos los pacientes para escalar
