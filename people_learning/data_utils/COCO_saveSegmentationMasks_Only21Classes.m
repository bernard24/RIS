
targetpath ='/media/data/coco/valImagesInPascal/';
targetJPEGpath = [targetpath 'JPEGImages/'];
targetSegMaskpath = [targetpath 'SegmentationClass/'];
targetImageSetPath= [targetpath 'ImageSets/Segmentation/'];

dataDir='/media/data/coco';
dataType='val2014';
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);

if(~exist('coco','var')),
    coco=CocoApi(annFile);
end


tmpMapping = dlmread('labels_maped_ids_only.txt', '', 1, 0);
mapTable = zeros(90, 1); % For some strange reason coco's max class id is 90
mapTable(tmpMapping(:,1)) = tmpMapping(:,2);


cats = coco.loadCats(coco.getCatIds());
nms={cats.name};
fprintf('COCO categories: ');
fprintf('%s, ',nms{:}); fprintf('\n');
nms=unique({cats.supercategory});
fprintf('COCO supercategories: ');
fprintf('%s, ',nms{:}); fprintf('\n');


catIds = coco.getCatIds();

imgIds = coco.getImgIds();
numofimages = length(imgIds);


fid = fopen([targetImageSetPath 'val.txt'], 'w');
vocmap = VOClabelcolormap(255);

for iimage = 1 : numofimages
    
    imgId = imgIds(iimage);
    
    %load the image
    img = coco.loadImgs(imgId);
    I = imread(sprintf('%s/images/%s/%s',dataDir,dataType, img.file_name));
    
    
    [~,~,num_channels] = size(I);
    
    if (num_channels ~= 3)
       fprintf('skipping %s\n', sprintf('%s/images/%s/%s',dataDir,dataType, img.file_name));
       continue;
    end
        
    %load and display instance annotations
    annIds = coco.getAnnIds('imgIds',imgId);%,'catIds',catIds,'iscrowd',[]);
    curAnnotations = coco.loadAnns(annIds);
    
    gtSegmentation = zeros(size(I,1), size(I,2));
    
    for annoInd = 1 : length(curAnnotations)
        
        if (curAnnotations(annoInd).iscrowd)
            mask = logical(coco.decodeMask(curAnnotations(annoInd).segmentation));
        else
            mask = coco.segToMask(curAnnotations(annoInd).segmentation,img.height,img.width);
        end
        
        pascal_cat_id = mapTable(curAnnotations(annoInd).category_id);
        if (pascal_cat_id == 0)
            error('Something is seriously wrong\n');
        end
        gtSegmentation(mask) = pascal_cat_id;
    end
        
    if numel(gtSegmentation(gtSegmentation > 0 & gtSegmentation < 21)) < 200
        continue;
    end
    
    
    gtSegmentation = gtSegmentation + 1;
    gtSegmentation(gtSegmentation > 21) = 1;
    
    curFileTitle = img.file_name(1:end-4);
    imageFilename = [targetJPEGpath curFileTitle '.jpg'];
    gtFilename = [targetSegMaskpath curFileTitle '.png'];
    
    
    [im_h, im_w, ~] = size(I);
    im = I;
    
    if (im_h > 500)
        hStart = floor((im_h - 500) / 2) + 1;
        im = im(hStart:hStart + 499,:,:);
        gtSegmentation = gtSegmentation(hStart:hStart + 499,:);
    end
    
    if (im_w > 500)
        wStart = floor((im_w - 500) / 2) + 1;
        im = im(:,wStart:wStart+499,:);
        gtSegmentation = gtSegmentation(:,wStart:wStart+499);
    end
    
    imwrite(im, imageFilename, 'jpg');
    imwrite(gtSegmentation, vocmap, gtFilename, 'png');
    
    fprintf(fid, '%s\n', curFileTitle);
end
fclose(fid);
