load('allTimes.mat');

[a, numPatients] = size(allTimes);

Times = zeros(numPatients);

for i=1:numPatients
    
    [a, numRecs] = size(allTimes{i});
    
    CrisisCumulative = 0;
    
    for j=1:numRecs
        
        crisisTimes = AllTimes{i}{j};
        
        crisisCumulative = crisisCumulative + CrisisTimes(2) - CrisisTimes(1);
        
    end
    
    Times(i) = crisisCumulative;
    
end

fileID = fopen('crisisCumulativeTimes.txt','w');
formatSpec = '%i';
fprintf(fileID,formatSpec,Times);


