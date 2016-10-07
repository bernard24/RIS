local Upsample, parent = torch.class('nn.Upsample', 'nn.Module')


function Upsample:__init(factor, n_channels)
	parent.__init(self)
	self.factor = factor
	self.n_channels = n_channels

	kernel_size = 2*factor - factor%2
	kernel_stride = factor

	self.kernel_size = kernel_size

	a_conv = nn.SpatialConvolution(
	      n_channels, n_channels,
	      kernel_size, kernel_size, 
	      kernel_stride, kernel_stride,
	      math.ceil((factor-1)/2)
	)
	a_conv.bias:fill(0)

	f = math.ceil(kernel_size/2)
	c = (2*f-1-f%2)/(2*f)
	bilateral_filter = torch.zeros(n_channels, n_channels, kernel_size, kernel_size)
	for cha=1,n_channels do
		for x=1,kernel_size do
		    for y=1,kernel_size do
			bilateral_filter[cha][cha][x][y] = (1-math.abs((x-1)/f-c))*(1-math.abs((y-1)/f-c))
		    end
		end
	end
	a_conv.weight = bilateral_filter

	if gpumode==1 then
		a_conv:cuda()
	end
	self.conv = a_conv
	self.bias = a_conv.bias
	self.weight = a_conv.weight
	self.gradWeight = a_conv.gradWeight
	self.gradBias = a_conv.gradBias
end


function Upsample:updateOutput(input)
	self.conv.bias = self.bias
	self.conv.weight = self.weight
	local output_aux = torch.zeros((#input)[1], (#input)[2]*self.factor, (#input)[3]*self.factor)
	local finput_aux = torch.zeros(self.kernel_size*self.kernel_size*self.n_channels, (#input)[2]*(#input)[3])
	if gpumode==1 then
		output_aux = output_aux:cuda()
		finput_aux = finput_aux:cuda()
	end
	self.conv.finput = finput_aux	
	self.finput = self.conv.finput
	self.output = self.conv:updateGradInput(output_aux, input)
	return self.output
end


function Upsample:updateGradInput(input, gradOutput)
	self.conv.bias = self.bias
	self.conv.weight = self.weight
	local finput_aux = torch.zeros(self.kernel_size*self.kernel_size*self.n_channels, (#input)[2]*(#input)[3])
	if gpumode==1 then
		finput_aux = finput_aux:cuda()
	end
	self.conv.finput = finput_aux
	self.finput = self.conv.finput
	
	self.gradInput = self.conv:updateOutput(gradOutput)
	return self.gradInput
end

function accGradParameters(input, gradOutput, scale)
	self.gradWeight:fill(0)
   	self.gradBias:fill(0)
end


