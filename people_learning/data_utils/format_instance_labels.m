
list_file = 'ImageSets/Segmentation/train.txt';

fid = fopen(list_file);
tline = fgets(fid);
counter = 0;
while ischar(tline)
    counter = counter+1;
    object_im = imread(['SegmentationObject/', tline(1:end-1), '.png']);
    class_im = imread(['SegmentationClass/', tline(1:end-1), '.png']);
    elements = unique(object_im);
    if elements(1)==0
        elements = elements(2:end);
    end
    if elements(end)==255
        elements = elements(1:end);
    end
    class_elements = [];
    for j=1:length(elements)
        class_elements = [class_elements, class_im(find(object_im==elements(j), 1))];
    end
    dlmwrite('classes.txt',class_elements, '-append')
    disp(class_elements)
    tline = fgets(fid);
end    
fclose(fid);
    