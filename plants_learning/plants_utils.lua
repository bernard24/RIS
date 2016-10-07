require 'torch'
local plants_utils = {}
plants_utils.list_file = nil
plants_utils.labels_file = nil
function plants_utils.create_instance (input_path, gt_path, width, height, gt_width, gt_height, scaling, flipping, rotating)
    gt_height =  gt_height or height
    gt_width =  gt_width or width    

    scaling = scaling or 0
    rotating = rotating or 1
    flipping = flipping or 1

    original_height = 530
    original_width = 500

    list_filename = input_path .. 'data_list.txt'
    labels_filename = gt_path .. 'label_list.txt'

    if not plants_utils.list_file then
	plants_utils.list_file = io.open(list_filename, "r")
	plants_utils.labels_file = io.open(labels_filename, "r")
    end
    io.input(plants_utils.list_file)
    local input_file = io.read()
    io.input(plants_utils.labels_file)
    local gt_file = io.read()
    if gt_file==nil then
        print('New epoch.')
        -- closes the open file
        io.close(plants_utils.list_file)
        io.close(plants_utils.labels_file)
        -- Opens a file in read
	plants_utils.list_file = io.open(list_filename, "r")
	plants_utils.labels_file = io.open(labels_filename, "r")

        io.input(plants_utils.list_file)
        input_file = io.read()
        io.input(plants_utils.labels_file)
        gt_file = io.read()
        --radians = radians+0.5
    end
    local input_image = image.load(input_path .. input_file):sub(1,3)
    local gt_image = image.load(gt_path .. gt_file)*255
    
    input_image = image.scale(input_image, original_width, original_height, 'bilinear')
    gt_image = image.scale(gt_image, original_width, original_height, 'simple')

    if scaling==1 then
	    local x_times = torch.uniform()*1.5+0.5
    --print(x_times)
   	 local new_width = torch.floor(original_width*x_times)
    	local new_height = torch.floor(original_height*x_times)
    	input_image = image.scale(input_image, new_width, new_height, 'bilinear')
    	gt_image = image.scale(gt_image, new_width, new_height, 'simple')
    
	if x_times>1 then
        	input_image = image.crop(input_image, torch.floor(new_width/2)-torch.floor(original_width/2),torch.floor(new_height/2)-torch.floor(original_height/2), torch.floor(new_width/2)+torch.floor(original_width/2),torch.floor(new_height/2)+torch.floor(original_height/2))
       	 	gt_image = image.crop(gt_image, torch.floor(new_width/2)-torch.floor(original_width/2),torch.floor(new_height/2)-torch.floor(original_height/2), torch.floor(new_width/2)+torch.floor(original_width/2),torch.floor(new_height/2)+torch.floor(original_height/2))
    	else
        	local aux_input = torch.zeros(3,original_height,original_width)
        	aux_input[{{},{1,new_height},{1,new_width}}] = input_image
        	input_image = aux_input
        	local aux_gt = torch.zeros(1,original_height,original_width)
        	aux_gt[{{},{1,new_height},{1,new_width}}] = gt_image
        	gt_image = aux_gt
    	end
    end

    if flipping==1 then
	local rand_flip = torch.uniform()
   	if rand_flip>0.5 then
        	input_image = image.hflip(input_image)
        	gt_image = image.hflip(gt_image)
    	end
    end
    
    if rotating==1 then
	local radians = torch.uniform()*2*3.14159
    	input_image =  image.rotate(input_image, radians, 'bilinear')
    	gt_image =  image.rotate(gt_image, radians, 'simple')
    end

    input_image = torch.reshape(image.scale(input_image, width, height):float(), 3, height, width)
    gt_image = torch.reshape(image.scale(gt_image, gt_width, gt_height, 'simple'):float(), 1, 1, gt_height, gt_width)
    
    local n_instances = gt_image:max()
    nInstancesPerImage = n_instances
    
    nPixels = height*width
    local gt_tensor = torch.Tensor(n_instances, gt_height, gt_width):fill(0)

    counter = 1
    for i = n_instances,1,-1 do
        local max_val = gt_image:max()
        local current_mask = gt_tensor:sub(counter,counter,1,gt_height,1,gt_width)
        current_mask[torch.ge(gt_image,max_val)] = 1
        counter = counter+1
        gt_image[torch.ge(gt_image,max_val)] = 0
    end
    
    if gpumode==1 then
        input_image = input_image:cuda()
        gt_tensor = gt_tensor:cuda()
    end
    
    return input_image, gt_tensor
end
return plants_utils
