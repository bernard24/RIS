function create_cnn (nChannels, kernelSize, inputSize, kernelStride)

	inputSize = inputSize or 3
	kernelStride = kernelStride or 1

	input = nn.Identity()()

	layer1 = nn.ReLU()(nn.SpatialConvolution( 
	      inputSize, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(input))
	layer1.data.mapindex[1].module.name = 'layer1'

	layer2 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(layer1))
	layer2.data.mapindex[1].module.name = 'layer2'

	layer3 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(layer2))
	layer3.data.mapindex[1].module.name = 'layer3'

	layer5 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(layer3))
	layer5.data.mapindex[1].module.name = 'last_layer'

	model = nn.gModule({input}, {layer5})
	if gpumode==1 then
		model:cuda()
	end
	return model
end

function create_big_cnn (nChannels, kernelSize, inputSize, kernelStride)

	inputSize = inputSize or 3
	kernelStride = kernelStride or 1

	input = nn.Identity()()

	factor = 5
	bilinear_down_kernelSize = 2*factor-factor%2
 	bilinear_down_kernelStride = factor
 	bilinear_down_pad = math.ceil((factor-1)/2)

	f = math.ceil(bilinear_down_kernelSize/2)
	c = (2*f-1-f%2)/(2*f)
	bilinear_filter = torch.zeros(nChannels, nChannels, bilinear_down_kernelSize, bilinear_down_kernelSize)
	for cha=1,nChannels do
		for x=1,bilinear_down_kernelSize do
		    for y=1,bilinear_down_kernelSize do
			bilinear_filter[cha][cha][x][y] = (1-math.abs((x-1)/f-c))*(1-math.abs((y-1)/f-c))
		    end
		end
	end
	
	layer1 = nn.ReLU()(nn.SpatialConvolution( 
	      inputSize, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(input))

	layer2 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      bilinear_down_kernelSize, bilinear_down_kernelSize, 
	      bilinear_down_kernelStride, bilinear_down_kernelStride,
	      bilinear_down_pad
	   )(layer1))

	layer2.data.mapindex[1].module.weight = bilinear_filter

	layer3 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(layer2))

	layer5 = nn.ReLU()(nn.SpatialConvolution( 
	      nChannels, nChannels, 
	      kernelSize, kernelSize, 
	      kernelStride, kernelStride,
	      math.floor(kernelSize/2)
	   )(layer3))

	model = nn.gModule({input}, {layer5})
	if gpumode==1 then
		model:cuda()
	end
	return model
end
