%Code for generating images
datasetPath = 'C:\Users\catag\Documents\Maestria\Dataset';
%datasetPath = '/hpcfs/home/c.gomez10/inv/Dataset';
dirDataset = dir(datasetPath);
load('AllTimes_cut.mat');
load('regNumber_cut.mat');
ww = 4;
srate = 256;
%neg_instances = {};
% neg_instances(:,:,1) = zeros(23, ww * srate);
% neg_instances(:,:,1) = [];

%pos_instances = {};
% pos_instances(:,:,1) = zeros(23, ww*srate);
% pos_instances(:,:,1) = [];

for id = 3:numel(dirDataset) %recorrer pacientes
%for id = 3:3
%     neg_instances = {};
%     pos_instances = {};
    
    patient = dirDataset(id).name;
    patientPath = fullfile(datasetPath,patient);
    addpath(sprintf('C:\\Users\\catag\\Documents\\Maestria\\Dataset\\%s',patient))
    %addpath(sprintf('/hpcfs/home/c.gomez10/inv/Dataset/%s',patient))
   
    dataTimesPatient = allTimes{id-2};
    regNum = regNumber(:,id-2);
    regNum = regNum(~cellfun('isempty',regNum));
    reg = dir(fullfile(patientPath,'*.edf'));
    %fileNames = dir(sprintf('./Dataset/%s/*.edf',patient));
     %for j = 3:numel(regNum)
     %for j = 1:numel(reg) %recorrer registros     
     for j =3:5    
        cont_neg = 0;
        cont_pos = 0;
        fileName = reg(j).name;
        %fileName = strcat(patient,'_',regNum{j});
        %load(fileName)
        [st_Header, m_SignalsMat] = edfread(fileName);
        mean_per_channel = mean(m_SignalsMat,2);
        m_SignalsMat = bsxfun(@minus, m_SignalsMat, mean_per_channel);
        %ignore channel 24
        %m_SignalsMat = m_SignalsMat(1:23,:);
        s_SRate = st_Header.samples(1);
        
        %m_SignalsMat = m_SignalsMat';
        v_Time = (0:size(m_SignalsMat, 2) - 1)./ s_SRate;
        
        s_SizeWidthSec = 4;
        s_SzeWidthSam = round(s_SizeWidthSec * s_SRate);
        t_startSam = 1;
        t_endSam = s_SzeWidthSam;
        
        %verificar que no haya eventos mixtos
        if dataTimesPatient{j}(1) == 0 %no hay crisis en ese registro
            
            while t_endSam <= size(m_SignalsMat,2)
                %neg_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                cont_neg = cont_neg + 1;
                m_SignalsMat_Temp = m_SignalsMat(:,t_startSam:t_endSam);
                m_SignalsMat_Temp = m_SignalsMat_Temp';
                v_Time_Temp = v_Time(t_startSam:t_endSam);
                
                [s_FigHdl, s_AxesHdl, s_PlotHdl] = f_PlotMultiCenterSigs( ...
                    m_SignalsMat_Temp, 1, 1, v_Time_Temp, st_Header.label);
                save_Name = strcat(fileName(1:end-4),'neg',num2str(cont_neg));
                print(s_FigHdl,sprintf('C:\\Users\\catag\\Documents\\Maestria\\NewDataset\\Negatives\\%s' ,save_Name),'-dpng')
                %neg_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_label{end+1} = 0;
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
            end
            
        else
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
                %neg_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                %neg_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                cont_neg = cont_neg + 1;
                m_SignalsMat_Temp = m_SignalsMat(:,t_startSam:t_endSam);
                m_SignalsMat_Temp = m_SignalsMat_Temp';
                v_Time_Temp = v_Time(t_startSam:t_endSam);
                
                [s_FigHdl, s_AxesHdl, s_PlotHdl] = f_PlotMultiCenterSigs( ...
                    m_SignalsMat_Temp, 1, 1, v_Time_Temp, st_Header.label);
                save_Name = strcat(fileName(1:end-4),'neg',num2str(cont_neg));
                print(s_FigHdl,sprintf('C:\\Users\\catag\\Documents\\Maestria\\NewDataset\\Negatives\\%s' ,save_Name),'-dpng')
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
                
            elseif sum(m_SignalsBinary(1,t_startSam:t_endSam)) == s_SzeWidthSam %solo crisis
                %pos_instances{end+1} = m_SignalsMat(:,t_startSam:t_endSam);
                %pos_instances(:,:,end+1) = m_SignalsMat(:,t_startSam:t_endSam);
                cont_pos = cont_pos + 1;
                m_SignalsMat_Temp = m_SignalsMat(:,t_startSam:t_endSam);
                m_SignalsMat_Temp = m_SignalsMat_Temp';
                v_Time_Temp = v_Time(t_startSam:t_endSam);
                
                [s_FigHdl, s_AxesHdl, s_PlotHdl] = f_PlotMultiCenterSigs( ...
                    m_SignalsMat_Temp, 1, 1, v_Time_Temp, st_Header.label);
                save_Name = strcat(fileName(1:end-4),'pos',num2str(cont_pos));
                print(s_FigHdl,sprintf('C:\\Users\\catag\\Documents\\Maestria\\NewDataset\\Positives\\%s' ,save_Name),'-dpng')
                
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
                
            else %no cuenta para ninguna instancia
                t_startSam = t_startSam + s_SzeWidthSam;
                t_endSam = t_endSam + s_SzeWidthSam;
                
            end
            end             
                 
        end
    fprintf('done register %s \n', fileName); 
    
     end        
     %submuestrear al final de cada paciente
     %select the negative instances randomly
%      vectRand = randi(numel(neg_instances),1,numel(pos_instances));
%      neg_instances_pat{id-2} = neg_instances(vectRand);
%      pos_instances_pat{id-2} = pos_instances;
%      save('pos_instances_pat2.mat','pos_instances_pat');
%      save('neg_instances_pat2.mat','neg_instances_pat');
% 
%     clear neg_instances
%     clear pos_instances
end 


% save('pos_instances_pat.mat','pos_instances_pat');
% save('neg_instances_pat.mat','neg_instances_pat');