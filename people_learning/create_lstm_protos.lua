function create_lstm_protos (height, width, nChannels, rnn_size, rnn_layers, filter_size, n_classes)
	n_classes = n_classes or 1
	local protos = {}
    protos.rnn = ConvLSTM.convlstm(height, width, nChannels, rnn_size, rnn_layers, filter_size)
	-- the initial state of the cell/hidden states
	init_state = {}
	for L=1,rnn_layers do
	    local h_init = torch.zeros(rnn_size,height, width)
	    if gpumode==1 then h_init = h_init:cuda() end
	    table.insert(init_state, h_init:clone())
	    table.insert(init_state, h_init:clone())
	end

	-- post_lstm:
	local top_h = nn.Identity()()
	local outputs = {}
	local after_conv = nn.SpatialConvolution(
	      rnn_size, 1,
	      1, 1, 
	      1, 1,
	      0
	   )(top_h)
	after_conv.data.module.name = 'after_conv'
	after_conv.data.module.weight:uniform(-0.08,0.08)
	local addition = nn.Add(1)(nn.LogSoftMax()(nn.Reshape(height*width,false)(after_conv)))
	local real_output = nn.Sigmoid()(nn.Reshape(1,height,width,false)(addition))  --Problematic line (order width and height)
	addition.data.module.bias:fill(1.0)
	addition.data.module.name = 'addition'


	local upsample = nn.Upsample(8, 1)(real_output):annotate{name = 'upsample'}
	upsample.data.module.name = 'upsample'
	local crop_dist = -6
	crop = nn.SpatialZeroPadding(crop_dist, crop_dist, crop_dist, crop_dist)(upsample):annotate{name = 'crop'}

	table.insert(outputs, crop)
	table.insert(outputs, top_h)
	protos.post_lstm = nn.gModule({top_h}, outputs)

	local top_h_copy1 = nn.Identity()()
	local crop_copy1 = nn.Identity()()
	local shall_we_stop_inputs = {}
	local shall_we_stop_outputs = {}
	table.insert(shall_we_stop_inputs, crop_copy1)
	table.insert(shall_we_stop_inputs, top_h_copy1)
	table.insert(shall_we_stop_outputs, crop_copy1)
	shall_we_stop_linear = nn.Linear(
	  4*nChannels, 1
	)(nn.Reshape(4*nChannels,false)(nn.SpatialMaxPooling(width/2, height/2)(top_h_copy1)))
	shall_we_stop_linear.data.module.name = 'shall_we_stop_linear'
	shall_we_stop_linear.data.module.bias:fill(10.0)
	shall_we_stop_linear.data.module.weight:uniform(-0.08,0.08)
	local shall_we_stop = nn.Sigmoid()(shall_we_stop_linear)
	table.insert(shall_we_stop_outputs, shall_we_stop)

	protos.shall_we_stop = nn.gModule(shall_we_stop_inputs, shall_we_stop_outputs)

	if gpumode==1 then
	    for k,v in pairs(protos) do v:cuda() end
	end

	return protos, init_state
end
