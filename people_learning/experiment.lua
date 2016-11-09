require 'load_everything'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training plants network.')
cmd:text()
cmd:text('Options')
cmd:option('-name', 'experiment', 'Name of the experiment')
cmd:option('-gpumode', 1, 'Run on gpu')
cmd:option('-gpu_setDevice', 1, 'Which gpu use')
cmd:option('-fcn8_1_model', '', 'FCN8 model, part 1')
cmd:option('-fcn8_2_model', '', 'FCN8 model, part 2')
cmd:option('-fcn8_3_model', '', 'FCN8 model, part 3')
cmd:option('-lstm_model', '', 'LSTM model')
cmd:option('-learn_fcn8_1', 1, 'Whether learning FCN8 model, part 1')
cmd:option('-learn_fcn8_2', 1, 'Whether learning FCN8 model, part 2')
cmd:option('-learn_fcn8_3', 0, 'Whether learning FCN8 model, part 3')
cmd:option('-learn_lstm', 1, 'Whether learning LSTM model')
cmd:option('-learn_post_lstm', 1, 'Whether learning postLSTM model')
cmd:option('-learn_shall_we_stop', 1, 'Whether learning stop condition classifier')
cmd:option('-learning_rate', 10^-6, 'Learning rate')
cmd:option('-lambda', 1, 'lambda parameter for the IoUMultiClass loss function')
cmd:option('-seq_length', 5, 'Maximum number of RNN iterations at training stage')
cmd:option('-class', 0, 'Class of objects to be detected. If 0, then all pascal classes')
cmd:option('-non_object_iterations', 0, 'Number of iterations over the number of objects apearing in an image')
cmd:option('-images_dir', '../../../Data/images/', 'Data directory')
cmd:option('-labels_dir', '../../../Data/MatlabAPI_1/', 'Data directory')
cmd:option('-rnn_channels', 100, 'Number of channels of the rnn state')
cmd:option('-rnn_layers', 2, 'Number of layers of the rnn')
cmd:option('-rnn_type', 'ConvLSTM', 'Type of rnn')
cmd:option('-rnn_filter_size', 3, 'Size of the filter of the rnn')
cmd:option('-train_or_val', 'train', 'Select between training and validation instances')
cmd:option('-it', 10000, 'Training iterations')
cmd:option('-summary_after', 1000, 'Doing a summary of performance after this number of iterations')

-- parse input params
cmd_params = cmd:parse(arg)
input_path = cmd_params.data_dir
gt_path = cmd_params.data_dir
--cmd_params.rundir = cmd:string('experiment', cmd_params, {dir=true})
cmd_params.rundir = cmd_params.name
paths.mkdir(cmd_params.rundir)
-- create log file
cmd:log(cmd_params.rundir .. '/log', cmd_params)
gpumode = cmd_params.gpumode
if gpumode==1 then
	cutorch.setDevice(cmd_params.gpu_setDevice)
end

rnn_type = cmd_params.rnn_type

criterion = nn.MatchCriterion(cmd_params.lambda)
if gpumode==1 then
    criterion = criterion:cuda()
end

train_or_val = cmd_params.train_or_val
image_path = cmd_params.images_dir
gt_path = cmd_params.labels_dir

local seed = 1234567890
torch.manualSeed(seed)

-- Opens a file in read
list_file = io.open(gt_path .. train_or_val .. '_GT_list.txt', "r")
class_file = io.open(gt_path .. train_or_val .. '_GT_class.txt', "r")

-------------------------------------------------------------------------------

side_size = 500
nPixels = side_size*side_size

n_classes = 1
focus_class = {} --{[15] = 1}
if cmd_params.class==0 then
	for i = 1,20 do
	    	table.insert(focus_class,i+1)
	end
	n_classes = 21
else
	focus_class = {[cmd_params.class] = 1} --table.insert(focus_class, cmd_params.class)
end

function createInstance ()
    
	rotate = 1
    io.input(list_file)
    local image_name = io.read()
    io.input(class_file)
    local class_set = io.read()
    if class_set==nil then
        print('New epoc.')
        -- closes the open file
        io.close(list_file)
        io.close(class_file)
        -- Opens a file in read
	list_file = io.open(gt_path .. train_or_val .. '_GT_list.txt', "r")
	class_file = io.open(gt_path .. train_or_val .. '_GT_class.txt', "r")

        io.input(list_file)
        image_name = io.read()
        io.input(class_file)
        class_set = io.read()
    end
    local i=1
    local l={}
    local counter = 0


    for word in string.gmatch(class_set, '%d+') do l[i] = tonumber(word) if focus_class[l[i]] then counter = counter + 1 end i = i+1 end
    local instance_name = string.sub(image_name, 1, -5)
    local gt_name = instance_name .. '_gt.png'
    local gt_image = image.load(gt_path .. gt_name)
    local input = image.load(image_path .. image_name)*255
    
    if (#input)[1]==1 then
        return nil, nil
    end

    local gt_image_aux = gt_image:clone()*255
    local n_instances = gt_image_aux:max()
    local height = (#gt_image_aux)[2]
    local width = (#gt_image_aux)[3]

    if height>side_size and height>width then
        width = torch.floor(side_size/height*width)
        height = side_size
    end
    if width>side_size and width>=height then
        height = torch.floor(side_size/width*height)
        width = side_size
    end
    input = torch.reshape(image.scale(input, width, height):float(), 3, height, width)
    gt_image_aux = torch.reshape(image.scale(gt_image_aux, width, height, 'simple'):float(), 1, height, width)

    if scale then
	local rand_number = torch.uniform()/4  
	local vertical_pad = torch.floor(rand_number*height)
	local horizontal_pad = torch.floor(rand_number*width)
	local padder = nn.SpatialZeroPadding(horizontal_pad, horizontal_pad, vertical_pad, vertical_pad)
        local gt_padder = nn.SpatialZeroPadding(horizontal_pad, horizontal_pad, vertical_pad, vertical_pad)
	input = padder:forward(input)
	gt_image_aux = gt_padder:forward(gt_image_aux)
    end    

    local rand_flip = torch.uniform()
    if rand_flip>0.5 then
        input = image.hflip(input)
        gt_image_aux = image.hflip(gt_image_aux)
    end
    
    if rotate then
        local radians = torch.uniform()*3.14159/4 - 3.14159/8
        input =  image.rotate(input, radians, 'bilinear')
        gt_image_aux =  image.rotate(gt_image_aux, radians, 'simple')
    end

    input = torch.reshape(image.scale(input, width, height):float(), 3, height, width)
    gt_image_aux = torch.reshape(image.scale(gt_image_aux, width, height, 'simple'):float(), 1, 1, height, width)
    
    local actual_input = torch.Tensor(3, side_size, side_size):fill(0)
    actual_input[{ {1,3}, {1,height}, {1,width} }] = input

    
    local gt_tensor = torch.Tensor(counter, side_size, side_size):fill(0)

    counter = 1
    gt_labels = {}
    if n_instances>0 then
	    for i = n_instances,1,-1 do
		local max_val = gt_image_aux:max()
		if focus_class[l[i]] then
		    local current_mask = gt_tensor:sub(counter,counter,1,height,1,width)
		    current_mask[torch.ge(gt_image_aux,max_val)] = 1
		    if current_mask:max()>0 then
		        counter = counter+1
		        table.insert(gt_labels, l[i])
		    end
		end
		gt_image_aux[torch.ge(gt_image_aux,max_val)] = 0
	    end
    end
    if counter>1 then
        gt_tensor = gt_tensor:sub(1,counter-1)
        gt_tensor = gt_tensor:permute(1,3,2)
    else
        gt_tensor = torch.Tensor()    
    end
        
    local mean_pix = torch.Tensor({-103.939, -116.779, -123.68})
    local mean_pix_tensor = torch.repeatTensor(mean_pix, side_size,side_size,1)
    mean_pix_tensor = mean_pix_tensor:permute(3,1,2)

    local actual_input_aux = torch.Tensor(#actual_input)
    actual_input_aux[{{1},{},{}}] = actual_input[{{3},{},{}}]
    actual_input_aux[{{2},{},{}}] = actual_input[{{2},{},{}}]
    actual_input_aux[{{3},{},{}}] = actual_input[{{1},{},{}}]
    
    actual_input_aux:add(mean_pix_tensor)

    actual_input_aux = actual_input_aux:permute(1,3,2) 
    
    if gpumode==1 then
        actual_input_aux = actual_input_aux:cuda()
        gt_tensor = gt_tensor:cuda()
    end
    return actual_input_aux, gt_tensor, gt_labels
end

function create_instance_with_class()
    while true do
        input, tensorGt, gt_labels = createInstance()
        if input and input[1] then
		return input, tensorGt, gt_labels
	end
    end
end

input, tensorGt, gt_labels = create_instance_with_class()

-------------------------------------------------------------------------------
nChannels = cmd_params.rnn_channels
if #cmd_params.fcn8_1_model==0 then
	print('Creating Model8.1.')
	model8_1 = Fcn8.get_part1()
else
	print('Loading Model8.1.')
	model8_1 = torch.load(cmd_params.fcn8_1_model)
	model8_1 = Fcn8.format_part1(model8_1)
end

if #cmd_params.fcn8_2_model==0 then
	print('Creating Model8.2.')
	model8_2 = Fcn8.get_part2(nChannels)
else
	print('Loading Model8.2.')
	model8_2 = torch.load(cmd_params.fcn8_2_model)
	model8_2 = Fcn8.format_part2(model8_2, nChannels)
end
if #cmd_params.fcn8_3_model==0 then
	local input = nn.Identity()()
	local upsample = nn.Upsample(8, 1)(input):annotate{name = 'upsample'}
	upsample.data.module.name = 'upsample'
	local crop_dist = -6
	crop = nn.SpatialZeroPadding(crop_dist, crop_dist, crop_dist, crop_dist)(upsample):annotate{name = 'crop'}
	model8_3 = nn.gModule({input}, {crop})
else
	model8_3 = torch.load(cmd_params.fcn8_3_model)
end

if gpumode==1 then
    model8_1:cuda()
    model8_2:cuda()
    model8_3:cuda()
end

-------------------------------------------------------------------------------

ConvLSTM = {}
if rnn_type=='ConvLSTM' then
	ConvLSTM = require 'ConvLSTM'
else
    ConvLSTM = require 'ConvRNN'
end
protos = {}
rnn_size = nChannels
rnn_layers = cmd_params.rnn_layers
xSize = 64
ySize = 64

protos, init_state = create_lstm_protos(xSize, ySize, nChannels, rnn_size, rnn_layers, 3, n_classes)

if #cmd_params.lstm_model==0 then
        print('Creating LSTM.')
else
        print('Loading LSTM.')

	loaded_protos = torch.load(cmd_params.lstm_model)
	protos.rnn = ConvLSTM.format(loaded_protos.rnn, xSize, ySize, nChannels, rnn_size, rnn_layers, 3)

	if cmd_params.class==0 then
		local target_node = getParameter(protos.post_lstm, 'after_conv')
		local source_node = getParameter(loaded_protos.post_lstm, 'after_conv')

		if not source_node then 	-- In the old version this node does not have a name and we have to look for it
			loaded_protos.post_lstm:apply( function(m) if not m.name and m.bias and m.weight then source_node = m end end)
	  	end

		target_node.weight = source_node.weight:clone()
		target_node.bias = source_node.bias:clone()	

		local source_node = getParameter(loaded_protos.post_lstm, 'addition')
		local target_node = getParameter(protos.post_lstm, 'addition')
		target_node.bias = source_node.bias:clone()
	else
		local target_node = getParameter(protos.post_lstm, 'after_conv')
		local source_node = getParameter(loaded_protos.rnn, 'after_conv')

		if not source_node then 	-- In the old version this node does not have a name and we have to look for it
			loaded_protos.rnn:apply( function(m) if not m.name and m.bias and m.weight then source_node = m end end)
	  	end
		if not source_node then
			source_node = getParameter(loaded_protos.post_lstm, 'after_conv')
		end
		if not source_node then 	-- In the old version this node does not have a name and we have to look for it
			loaded_protos.post_lstm:apply( function(m) if not m.name and m.bias and m.weight then source_node = m end end)
	  	end

		target_node.weight = source_node.weight:clone()
		target_node.bias = source_node.bias:clone()	

		local source_node = getParameter(loaded_protos.rnn, 'addition')
		if not source_node then
			source_node = getParameter(loaded_protos.post_lstm, 'addition')
		end
		local target_node = getParameter(protos.post_lstm, 'addition')
		target_node.bias = source_node.bias:clone()

		local target_node = getParameter(protos.shall_we_stop, 'shall_we_stop_linear')
		local source_node = getParameter(loaded_protos.rnn, 'shall_we_stop_linear')
		if not source_node then
			source_node = getParameter(loaded_protos.post_lstm, 'shall_we_stop_linear')
		end
		if not source_node then
			source_node = getParameter(loaded_protos.shall_we_stop, 'shall_we_stop_linear')
		end
		target_node.weight = source_node.weight:clone()
		target_node.bias = source_node.bias:clone()	
	end
end
if gpumode==1 then
    protos.rnn:cuda()
end

-------------------------------------------------------------------------------

optimState = {
  learningRate = cmd_params.learning_rate,
   momentum = 0.99
}

-------------------------------------------------------------------------------

learning_networks = {}
if cmd_params.learn_fcn8_1==1 then
    table.insert(learning_networks, model8_1)
end
if cmd_params.learn_fcn8_2==1 then
    table.insert(learning_networks, model8_2)
end
if cmd_params.learn_lstm==1 then
    table.insert(learning_networks, protos.rnn)
end
if cmd_params.learn_post_lstm==1 then
    table.insert(learning_networks, protos.post_lstm)
end
if cmd_params.learn_shall_we_stop==1 then
    table.insert(learning_networks, protos.shall_we_stop)
end
if cmd_params.learn_fcn8_3==1 then
    table.insert(learning_networks, model8_3)
end

params, grad_params = model_utils.combine_all_parameters(learning_networks)
-- make a bunch of clones after flattening, as that reallocates memory

seq_length = cmd_params.seq_length --nInstancesPerImage
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, seq_length, not proto.parameters)
end

fork_input = nn.Identity()()
local fork_outputs = {}
for i=1,seq_length do
    table.insert(fork_outputs, nn.Identity()(fork_input))
end
fork = nn.gModule({fork_input}, fork_outputs)
if gpumode==1 then
    fork = fork:cuda()
end

init_state = {}
for L=1,rnn_layers do
    local h_init = torch.zeros(rnn_size,xSize, ySize)
    if gpumode==1 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if rnn_type=='ConvLSTM' then
        table.insert(init_state, h_init:clone())
    end
end

-------------------------------------------------------------------------------

print('Starting training...')
--ProFi = require 'ProFi'
--ProFi:start()
grad_clip = 5

n_training_instances = cmd_params.it
summarize_results_after = cmd_params.summary_after
testo=0
hist = torch.Tensor(torch.floor(n_training_instances/summarize_results_after))
minihist = torch.Tensor(summarize_results_after):fill(0)
optimMethod = optim.adam --rmsprop --sgd --
local time = sys.clock()

print('number of parameters in the model: ' .. params:nElement())

for i=1,n_training_instances do
          -- disp progress
    xlua.progress(i, n_training_instances)
    inputAux, tensorGt, gt_labels = create_instance_with_class()

    nInstancesPerImage = #gt_labels
    if nInstancesPerImage>0 then
	    target = tensorGt:resize(nInstancesPerImage,nPixels):t()
    else
	    target = tensorGt
    end
    -- do fwd/bwd and return loss, grad_params
    local init_state_global = clone_list(init_state)
    
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= params then
           params:copy(x)
        end

        -- reset gradients
        grad_params:zero()
        local loss = 0

        ------------------- forward pass -------------------
        model8_1:training()
        med_x = model8_1:forward(inputAux)
        med2_x = model8_2:forward(med_x)  
        x = fork:forward(med2_x) 
        
        local rnn_state = {[0] = init_state_global}
        local predictions_small = {}
	    local middle_predictions = {}
        local predictions = {}
        local rnn_iterations = math.min(seq_length, nInstancesPerImage+cmd_params.non_object_iterations)
        local dictions = torch.Tensor(500*500, rnn_iterations)
    	local scores = torch.Tensor(rnn_iterations)           -- softmax outputs
    	if gpumode==1 then
    		dictions = dictions:cuda()
    		scores = scores:cuda()
    	end
        for t=1,rnn_iterations do
            --clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
            predictions_small[t] = lst[#lst] -- last element is the prediction
            --if cmd_params.class==0 then
		    middle_predictions[t] = clones.post_lstm[t]:forward(predictions_small[t])
		    predictions[t] = clones.shall_we_stop[t]:forward(middle_predictions[t])
		    actual_prediction = predictions[t][1]

			
			local AA = predictions[t][1]:resize(predictions[t][1]:nElement())
		    dictions[{{},t}] = AA
		    scores[t] = predictions[t][2]
        end
    	local aux = {[1] = dictions, [2] = scores}
    	loss = criterion:forward({[1] = dictions, [2] = scores}, target) / rnn_iterations
            
    	------------------ backward pass -------------------
    	local doutput = criterion:backward({[1] = dictions, [2] = scores}, target)

        -- initialize gradient at time t to be zeros (there's no influence from future)
        local drnn_state = {[rnn_iterations] = clone_list(init_state, true)} -- true also zeros the clones
        local dx = {}
        for t=rnn_iterations,1,-1 do
            -- backprop through loss, and softmax/linear
            
        local ddiction_t = doutput[1]:sub(1,-1,t,t)
        local dscore_t = doutput[2]:sub(t,t)

        local dmiddle_output_t = clones.shall_we_stop[t]:backward(middle_predictions[t], {[1] = ddiction_t, [2] = dscore_t})
        local doutput_t_small = clones.post_lstm[t]:backward(predictions[t], dmiddle_output_t)

        table.insert(drnn_state[t], doutput_t_small)

        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k == 1 then
                -- gradient on x
                dx[t] = v
            else -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end   
        
    for t=rnn_iterations+1,seq_length do
        dx[t] = dx[rnn_iterations]:clone():fill(0)
    end      
	if cmd_params.learn_fcn8_2==1 or cmd_params.learn_fcn8_1==1 then
		grad_params:clamp(-grad_clip, grad_clip)
		med2_dx = fork:backward(med2_x,dx) 
		med_dx = model8_2:backward(med_x,med2_dx)
	end
    if cmd_params.learn_fcn8_1==1 then
	    model8_1:backward(inputAux,med_dx)
	end    
    
    err = loss
    minihist[i%summarize_results_after+1]=err
    if testo==1 then
        hist[i/summarize_results_after] = torch.mean(minihist)
        print(hist[i/summarize_results_after])
        minihist:fill(0)
        testo = 0
	    if cmd_params.learn_fcn8_1==1 then
	            torch.save(cmd_params.rundir .. '/coco_fcn8_1.model', model8_1)
	    end
	    if cmd_params.learn_fcn8_2==1 then
	            torch.save(cmd_params.rundir .. '/coco_fcn8_2.model', model8_2)
	    end
	    if cmd_params.learn_lstm==1 or cmd_params.learn_post_lstm==1 or cmd_params.learn_shall_we_stop==1 then
	            torch.save(cmd_params.rundir .. '/coco_convlstm.model', protos)
	    end
            torch.save(cmd_params.rundir .. '/hist.t7', hist:sub(1,i/summarize_results_after))
        end
        return err,grad_params
    end

    if i%summarize_results_after==0 then
        testo = 1
    end
    optimMethod(feval, params, optimState)
end
time = sys.clock() - time
time = time / n_training_instances
print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
--ProFi:stop()
--ProFi:writeReport( 'profile.txt' )
