
local ConvLSTM = {}
ConvLSTM.typename = 'ConvLSTM'
function ConvLSTM.convlstm(height, width, input_channels, rnn_size, n, kernelSize, kernelStride, dropout)
  kernelStride = kernelStride or 1
  dropout = dropout or 0 

  nngraph.setDebug(true)

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}

  --local L = 1
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency


    local i2h, h2h
    if we_have_cudnn==1 then
	i2h = cudnn.SpatialConvolution(
      input_channels, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L}
    else
	i2h = nn.SpatialConvolution(
      input_channels, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L} 
    end

    if we_have_cudnn==1 then
	h2h = cudnn.SpatialConvolution(
      rnn_size, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    else
    h2h = nn.SpatialConvolution(
      rnn_size, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    end

    i2h.data.module.bias:uniform(-0.08,0.08)
    i2h.data.module.bias[{{1, 3*rnn_size}}]:fill(1.0)
    i2h.data.module.weight:uniform(-0.08,0.08)
    i2h.data.module.name = 'i2h_'..L
    h2h.data.module.bias:uniform(-0.08,0.08)
    h2h.data.module.weight:uniform(-0.08,0.08)
    h2h.data.module.name = 'h2h_'..L

    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size*height*width)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(1)(reshaped):split(4)
    -- decode the gates
    local in_gate_unshaped = nn.Sigmoid()(n1)
    local forget_gate_unshaped = nn.Sigmoid()(n2)
    local out_gate_unshaped = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform_unshaped = nn.Tanh()(n4)

    local in_gate = nn.Reshape(rnn_size,height,width)(in_gate_unshaped)
    local forget_gate = nn.Reshape(rnn_size,height,width)(forget_gate_unshaped)
    local out_gate = nn.Reshape(rnn_size,height,width)(out_gate_unshaped)
    local in_transform = nn.Reshape(rnn_size,height,width)(in_transform_unshaped)

    -- perform the LSTM update
    local next_c       = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local next_c = nn.Reshape(rnn_size, height, width)(next_c)
    local next_h = nn.Reshape(rnn_size, height, width)(next_h)
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local out = nn.Identity()(top_h)
  table.insert(outputs, out)

  return nn.gModule(inputs, outputs)
end

function ConvLSTM.format(net, input_x, input_y, input_channels, rnn_size, n, kernelSize, kernelStride, dropout)
  kernelStride = kernelStride or 1
  dropout = dropout or 0 

  nngraph.setDebug(true)

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}

  --local L = 1
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency


    local i2h, h2h
   if we_have_cudnn==1 then
	i2h = cudnn.SpatialConvolution(
      input_channels, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L}
    else
	i2h = nn.SpatialConvolution(
      input_channels, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L}
    end
    node = getParameter(net, 'i2h_'..L)
    i2h.data.module.bias = node.bias:clone()
    i2h.data.module.weight = node.weight:clone()
    i2h.data.module.name = 'i2h_'..L
   if we_have_cudnn==1 then
  	h2h = cudnn.SpatialConvolution(
      rnn_size, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    else
    h2h = nn.SpatialConvolution(
      rnn_size, 4*rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    end
    node = getParameter(net, 'h2h_'..L)
    h2h.data.module.bias = node.bias:clone()
    h2h.data.module.weight = node.weight:clone()
    h2h.data.module.name = 'h2h_'..L

    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size*input_x*input_y)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(1)(reshaped):split(4)
    -- decode the gates
    local in_gate_unshaped = nn.Sigmoid()(n1)
    local forget_gate_unshaped = nn.Sigmoid()(n2)
    local out_gate_unshaped = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform_unshaped = nn.Tanh()(n4)

    local in_gate = nn.Reshape(rnn_size,input_x,input_y)(in_gate_unshaped)
    local forget_gate = nn.Reshape(rnn_size,input_x,input_y)(forget_gate_unshaped)
    local out_gate = nn.Reshape(rnn_size,input_x,input_y)(out_gate_unshaped)
    local in_transform = nn.Reshape(rnn_size,input_x,input_y)(in_transform_unshaped)

    -- perform the LSTM update
    local next_c       = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local next_c = nn.Reshape(rnn_size, input_x, input_y)(next_c)
    local next_h = nn.Reshape(rnn_size, input_x, input_y)(next_h)
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local out = nn.Identity()(top_h)
  table.insert(outputs, out)

  collectgarbage()
  return nn.gModule(inputs, outputs)
end

return ConvLSTM

