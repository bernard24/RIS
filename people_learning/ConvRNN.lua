local ConvRNN = {}
ConvRNN.typename = 'ConvRNN'
function ConvRNN.convrnn(height, width, input_channels, rnn_size, n, kernelSize, kernelStride, dropout)
  kernelStride = kernelStride or 1
  dropout = dropout or 0 

  nngraph.setDebug(true)

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}

  --local L = 1
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency


    local i2h, h2h
    if we_have_cudnn==1 then
	i2h = cudnn.SpatialConvolution(
      input_channels, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L}
    else
	i2h = nn.SpatialConvolution(
      input_channels, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L} 
    end

    if we_have_cudnn==1 then
	h2h = cudnn.SpatialConvolution(
      rnn_size, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    else
    h2h = nn.SpatialConvolution(
      rnn_size, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    end

    i2h.data.module.bias:uniform(-0.08,0.08)
    i2h.data.module.weight:uniform(-0.08,0.08)
    i2h.data.module.name = 'i2h_'..L
    h2h.data.module.bias:uniform(-0.08,0.08)
    h2h.data.module.weight:uniform(-0.08,0.08)
    h2h.data.module.name = 'h2h_'..L

    local next_h = nn.Tanh()(nn.CAddTable()({i2h, h2h}))
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local out = nn.Identity()(top_h)
  table.insert(outputs, out)

  return nn.gModule(inputs, outputs)
end

function ConvRNN.format(net, input_x, input_y, input_channels, rnn_size, n, kernelSize, kernelStride, dropout)
  kernelStride = kernelStride or 1
  dropout = dropout or 0 

  nngraph.setDebug(true)

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}

  --local L = 1
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency


    local i2h, h2h
   if we_have_cudnn==1 then
	i2h = cudnn.SpatialConvolution(
      input_channels, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(x):annotate{name = 'i2h_'..L}
    else
	i2h = nn.SpatialConvolution(
      input_channels, rnn_size, 
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
      rnn_size, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    else
    h2h = nn.SpatialConvolution(
      rnn_size, rnn_size, 
      kernelSize, kernelSize, 
      kernelStride, kernelStride,
      math.floor(kernelSize/2)
    )(prev_h):annotate{name = 'h2h_'..L}
    end
    node = getParameter(net, 'h2h_'..L)
    h2h.data.module.bias = node.bias:clone()
    h2h.data.module.weight = node.weight:clone()
    h2h.data.module.name = 'h2h_'..L
    
    local next_h = nn.Tanh()(nn.CAddTable()({i2h, h2h}))
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


return ConvRNN
