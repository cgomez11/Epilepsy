ORIGIN =  '/home/cgomez11/Project/Epilepsy/Dataset/Data/DataW4/DataExp2';

test = load('test_mat2.mat');
images = test(1).test_mat;
labels = load('test_label2.mat');
testSetLabels = labels(1).test_label;

clear test;
clear labels;

[size1, size2, numInstances] = size(images);

for i=1:numInstances
	if isequal(testSetLabels(i), 0)

		im  = images(:,:,i);
		imwrite(im, strcat(ORIGIN, '/interIctal/', num2str(i), '.jpg') );

	elseif isequal(testSetLabels(i),1)

		im  = images(:,:,i);
		imwrite(im, strcat(ORIGIN, '/ictal/', num2str(i), '.jpg') );

	end

end
