require 'load_everything'

original_height = 530
original_width = 500

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training plants network.')
cmd:text()
cmd:text('Options')
cmd:option('-name', 'experiment', 'Name of the experiment')
cmd:option('-gpumode', 1, 'Run on gpu')
cmd:option('-gpu_setDevice', 1, 'Which gpu use')
cmd:option('-pre_model', '', 'Pre-CNN model')
cmd:option('-cnn_model', '', 'CNN model')
cmd:option('-lstm_model', '', 'LSTM model')

cmd:option('-seq_length', 40, 'Maximum number of RNN iterations at training stage')
cmd:option('-lambda', 1, 'Hyperparameter pondering the classification accuracy term.')
cmd:option('-pass_output', 0, 'Pass previous output as input')
cmd:option('-non_object_iterations', 0, 'Number of iterations over the number of objects apearing in an image')

--cmd:option('-height', 106, 'Height of the resized images')
--cmd:option('-width', 100, 'Width of the resized images')

cmd:option('-height', 530, 'Height of the resized images')
cmd:option('-width', 500, 'Width of the resized images')


cmd:option('-learn_pre', 0, 'Whether learning Pre-CNN model')
cmd:option('-learn_cnn', 1, 'Whether learning CNN model')
cmd:option('-learn_lstm', 1, 'Whether learning LSTM model')
cmd:option('-learn_post_lstm', 1, 'Whether learning post-LSTM model')
cmd:option('-learning_rate', 10^-4, 'Learning rate')
cmd:option('-data_dir', '../Data/LSCData/A1/', 'Data directory')
cmd:option('-rnn_channels', 30, 'Number of channels of the rnn state')
cmd:option('-rnn_layers', 2, 'Number of layers of the rnn')
cmd:option('-rnn_filter_size', 3, 'Size of the filter of the rnn')
cmd:option('-cnn_filter_size', 3, 'Size of the filter of the cnn')
cmd:option('-it', 100000, 'Training iterations')
cmd:option('-summary_after', 128, 'Doing a summary of performance after this number of iterations')

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

criterion = nn.MatchCriterion(cmd_params.lambda)
if gpumode==1 then
    criterion = criterion:cuda()
end

rnn_size = cmd_params.rnn_channels
rnn_layers = cmd_params.rnn_layers
nChannels = cmd_params.rnn_channels
filter_size = cmd_params.rnn_filter_size

xSize = 106 --cmd_params.height
ySize = 100 --cmd_params.width
height = cmd_params.height
width = cmd_params.width

is_big = 0
if height==530 then
	is_big = 1
end

if #cmd_params.cnn_model==0 then
	print('Creating CNN model from scratch.')
	model = create_cnn(nChannels, cmd_params.cnn_filter_size)
else
	print('Loading CNN ...')
	model = torch.load(cmd_params.cnn_model)
end

if is_big==1 then
	if #cmd_params.pre_model==0 then
		factor = 5
		bilinear_down_kernelSize = 2*factor-factor%2
	 	bilinear_down_kernelStride = factor
	 	bilinear_down_pad = math.ceil((factor-1)/2)

		f = math.ceil(bilinear_down_kernelSize/2)
		c = (2*f-1-f%2)/(2*f)
		input_channels = 3
		bilinear_filter = torch.zeros(input_channels, input_channels, bilinear_down_kernelSize, bilinear_down_kernelSize)
		for cha=1,input_channels do
			for x=1,bilinear_down_kernelSize do
			    for y=1,bilinear_down_kernelSize do
				bilinear_filter[cha][cha][x][y] = (1-math.abs((x-1)/f-c))*(1-math.abs((y-1)/f-c))
			    end
			end
			bilinear_filter[cha][cha] = bilinear_filter[cha][cha]/bilinear_filter[cha][cha]:sum()
		end
		local input = nn.Identity()()
		local output = nn.SpatialConvolution( 
		      3, 3, 
		      bilinear_down_kernelSize, bilinear_down_kernelSize, 
		      bilinear_down_kernelStride, bilinear_down_kernelStride,
		      bilinear_down_pad
		)(input)
	
		output.data.module.weight = bilinear_filter
		output.data.module.bias:fill(0)

		premodel = nn.gModule({input}, {output})

	else
		premodel = torch.load(cmd_params.pre_model)
	end
	if gpumode==1 then
		premodel:cuda()
	end
end


local protos = {}
if cmd_params.pass_output==1 then
	protos = create_lstm_protos (xSize, ySize, nChannels+1, rnn_size+1, rnn_layers, filter_size, cmd_params.lambda, 0)
else
	protos = create_lstm_protos (xSize, ySize, nChannels, rnn_size, rnn_layers, filter_size, cmd_params.lambda, 0)
end
if #cmd_params.lstm_model==0 then
	print('Creating ConvLSTM model from scratch.')
	print('Is it big? ' .. is_big)
else
	print('Loading LSTM.')
    loaded_protos = torch.load(cmd_params.lstm_model)
    protos.rnn = ConvLSTM.format(loaded_protos.rnn, xSize, ySize, nChannels, rnn_size, rnn_layers, 3)

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

	local source_node = getParameter(loaded_protos.post_lstm, 'shall_we_stop_linear')
	local target_node = getParameter(protos.post_lstm, 'shall_we_stop_linear')

	target_node.weight = source_node.weight:clone()
	target_node.bias = source_node.bias:clone()	

    if gpumode==1 then
        protos.rnn:cuda()
    end
end

optimState = {
  learningRate = cmd_params.learning_rate
}

learning_networks = {}
if cmd_params.learn_pre==1 then
    table.insert(learning_networks, premodel)
end
if cmd_params.learn_cnn==1 then
    table.insert(learning_networks, model)
end
if cmd_params.learn_lstm==1 then
    table.insert(learning_networks, protos.rnn)
end
if cmd_params.learn_post_lstm==1 then
    table.insert(learning_networks, protos.post_lstm)
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

-- the initial state of the cell/hidden states
init_state = {}
for L=1,rnn_layers do
    local h_init = {}
    if cmd_params.pass_output==1 then
	h_init = torch.zeros(rnn_size+1,xSize, ySize)
    else
	h_init = torch.zeros(rnn_size,xSize, ySize)
    end
    if gpumode==1 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end


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

    tensorGt = torch.Tensor(0)
    while tensorGt:nElement()==0 do
        inputAux, tensorGt, gt = plants.create_instance (input_path, gt_path, width, height, ySize, xSize)
    end

    local nInstancesPerImage = (#tensorGt)[1]
    local nPixels = (#tensorGt)[2]*(#tensorGt)[3]
    target = tensorGt:resize(nInstancesPerImage,nPixels):t()
    
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
		if is_big==1 then
			premed_x = premodel:forward(inputAux)
			med_x = model:forward(premed_x)
		else
			med_x = model:forward(inputAux)
		end

		if cmd_params.pass_output==1 then
		    local previous_output = torch.Tensor(1,xSize,ySize):fill(0)
		    if gpumode==1 then
			previous_output = previous_output:cuda()
		    end
		    med_x = torch.cat(med_x, previous_output, 1)
		end

	    x = fork:forward(med_x) 
	        
		local rnn_iterations = math.min(seq_length, nInstancesPerImage+cmd_params.non_object_iterations)
	        local rnn_state = {[0] = init_state_global}
	        local predictions = {}           -- softmax outputs
	        local dictions = torch.Tensor(ySize*xSize, rnn_iterations)           -- softmax outputs
	        local scores = torch.Tensor(rnn_iterations)           -- softmax outputs
		if gpumode==1 then
			dictions = dictions:cuda()
			scores = scores:cuda()
		end   

	        for t=1,rnn_iterations do
	            clones.rnn[t]:training()
	            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
	            rnn_state[t] = {}

	            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
	            predictions[t] = lst[#lst] -- last element is the prediction
		    local postlst = clones.post_lstm[t]:forward(predictions[t])
		
		    ori_size = #postlst[1]
	            dictions[{{},t}] = postlst[1]:resize(postlst[1]:nElement()) 
	            scores[t] = postlst[2]

		    if cmd_params.pass_output==1 and t<rnn_iterations then
			    x[t+1][{{-1},{},{}}] = postlst[1]
		    end
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
	    local predoutput_t = clones.post_lstm[t]:backward(predictions[t], {[1] = ddiction_t, [2] = dscore_t})
	       
	        table.insert(drnn_state[t], predoutput_t)

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
	    
	    grad_params:clamp(-grad_clip, grad_clip)
	        
		if cmd_params.learn_cnn==1 then
			med_dx = fork:backward(med_x,dx)
			if is_big==1 then
				premed_dx = model:backward(premed_x, med_dx)
				if cmd_params.learn_pre==1 then
					premodel:backward(inputAux, premed_dx)
				end
			else
				model:backward(inputAux,med_dx)
			end
		end
	    err = loss
	    minihist[i%summarize_results_after+1]=err
	    if testo==1 then
	        hist[i/summarize_results_after] = torch.mean(minihist)
	        minihist:fill(0)
	        print(hist[i/summarize_results_after])

	    if cmd_params.learn_pre==1 then
			torch.save(cmd_params.rundir .. '/plants_pre_cnn.model', premodel)
	    end
	    if cmd_params.learn_cnn==1 then
	        	torch.save(cmd_params.rundir .. '/plants_pre_lstm.model', model)
	    end
	    if cmd_params.learn_lstm==1 or cmd_params.learn_post_lstm==1 then
	        torch.save(cmd_params.rundir .. '/plants_convlstm.model', protos)
	    end
	    torch.save(cmd_params.rundir .. '/hist.t7', hist:sub(1,i/summarize_results_after))
	        testo = 0
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

print(hist[#hist])
