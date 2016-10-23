
list = dir('*label.png');

for i=1:length(list);
    disp(i)
    image = imread(list(i).name);
    imwrite(image, list(i).name)
end