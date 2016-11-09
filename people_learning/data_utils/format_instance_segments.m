list = dir('SegmentationObject/*.png');
for i=1:length(list)
    im = imread(['SegmentationObject/', list(i).name]);
    imwrite(im, [list(i).name]);
end
    