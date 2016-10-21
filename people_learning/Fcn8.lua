require 'nn'
require 'image'                                                           
require 'nnx'
require 'nngraph'
require 'Upsample'

local Fcn8 = {}
Fcn8.net = nil
nngraph.setDebug(true)
function Fcn8.get_part1()

	input = nn.Identity()()

	local conv1_1 = nn.ReLU()(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 100, 100)(input)):annotate{name = 'conv1_1'}
	conv1_1.data.mapindex[1].module.bias = Fcn8.net.modules[1].bias
	conv1_1.data.mapindex[1].module.weight = Fcn8.net.modules[1].weight
	conv1_1.data.mapindex[1].module.name = 'conv1_1'
	local conv1_2 = nn.ReLU()(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1)(conv1_1)):annotate{name = 'conv1_2'}
	conv1_2.data.mapindex[1].module.bias = Fcn8.net.modules[3].bias
	conv1_2.data.mapindex[1].module.weight = Fcn8.net.modules[3].weight
	conv1_2.data.mapindex[1].module.name = 'conv1_2'
	local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv1_2):annotate{name = 'pool1'}
	local conv2_1 = nn.ReLU()(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1, 1)(pool1)):annotate{name = 'conv2_1'}
	conv2_1.data.mapindex[1].module.bias = Fcn8.net.modules[6].bias
	conv2_1.data.mapindex[1].module.weight = Fcn8.net.modules[6].weight
	conv2_1.data.mapindex[1].module.name = 'conv2_1'
	local conv2_2 = nn.ReLU()(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 1, 1)(conv2_1)):annotate{name = 'conv2_2'}
	conv2_2.data.mapindex[1].module.bias = Fcn8.net.modules[8].bias
	conv2_2.data.mapindex[1].module.weight = Fcn8.net.modules[8].weight
	conv2_2.data.mapindex[1].module.name = 'conv2_2'
	local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv2_2):annotate{name = 'pool2'}
	local conv3_1 = nn.ReLU()(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1, 1, 1)(pool2)):annotate{name = 'conv3_1'}
	conv3_1.data.mapindex[1].module.bias = Fcn8.net.modules[11].bias
	conv3_1.data.mapindex[1].module.weight = Fcn8.net.modules[11].weight
	conv3_1.data.mapindex[1].module.name = 'conv3_1'
	local conv3_2 = nn.ReLU()(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1, 1)(conv3_1)):annotate{name = 'conv3_2'}
	conv3_2.data.mapindex[1].module.bias = Fcn8.net.modules[13].bias
	conv3_2.data.mapindex[1].module.weight = Fcn8.net.modules[13].weight
	conv3_2.data.mapindex[1].module.name = 'conv3_2'
	local conv3_3 = nn.ReLU()(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1, 1)(conv3_2)):annotate{name = 'conv3_3'}
	conv3_3.data.mapindex[1].module.bias = Fcn8.net.modules[15].bias
	conv3_3.data.mapindex[1].module.weight = Fcn8.net.modules[15].weight
	conv3_3.data.mapindex[1].module.name = 'conv3_3'
	local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv3_3):annotate{name = 'pool3'}
	local conv4_1 = nn.ReLU()(nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1)(pool3)):annotate{name = 'conv4_1'}
	conv4_1.data.mapindex[1].module.bias = Fcn8.net.modules[18].bias
	conv4_1.data.mapindex[1].module.weight = Fcn8.net.modules[18].weight
	conv4_1.data.mapindex[1].module.name = 'conv4_1'
	local conv4_2 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv4_1)):annotate{name = 'conv4_2'}
	conv4_2.data.mapindex[1].module.bias = Fcn8.net.modules[20].bias
	conv4_2.data.mapindex[1].module.weight = Fcn8.net.modules[20].weight
	conv4_2.data.mapindex[1].module.name = 'conv4_2'
	local conv4_3 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv4_2)):annotate{name = 'conv4_3'}
	conv4_3.data.mapindex[1].module.bias = Fcn8.net.modules[22].bias
	conv4_3.data.mapindex[1].module.weight = Fcn8.net.modules[22].weight
	conv4_3.data.mapindex[1].module.name = 'conv4_3'
	local pool4 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv4_3):annotate{name = 'pool4'}

	local conv5_1 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(pool4)):annotate{name = 'conv5_1'}
	conv5_1.data.mapindex[1].module.bias = Fcn8.net.modules[25].bias
	conv5_1.data.mapindex[1].module.weight = Fcn8.net.modules[25].weight
	conv5_1.data.mapindex[1].module.name = 'conv5_1'
	local conv5_2 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv5_1)):annotate{name = 'conv5_2'}
	conv5_2.data.mapindex[1].module.bias = Fcn8.net.modules[27].bias
	conv5_2.data.mapindex[1].module.weight = Fcn8.net.modules[27].weight
	conv5_2.data.mapindex[1].module.name = 'conv5_2'
	local conv5_3 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv5_2)):annotate{name = 'conv5_3'}
	conv5_3.data.mapindex[1].module.bias = Fcn8.net.modules[29].bias
	conv5_3.data.mapindex[1].module.weight = Fcn8.net.modules[29].weight
	conv5_3.data.mapindex[1].module.name = 'conv5_3'
	local pool5 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv5_3):annotate{name = 'pool5'}
	local fc6 = nn.Dropout(0.5)(nn.ReLU()(nn.SpatialConvolutionMM(512, 4096, 7, 7, 1, 1, 0, 0)(pool5))):annotate{name = 'fc6'}
	fc6.data.mapindex[1].mapindex[1].module.bias = Fcn8.net.modules[32].bias
	fc6.data.mapindex[1].mapindex[1].module.weight = Fcn8.net.modules[32].weight
	fc6.data.mapindex[1].mapindex[1].module.name = 'fc6'
	local fc7 = nn.Dropout(0.5)(nn.ReLU()(nn.SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1, 0, 0)(fc6))):annotate{name = 'fc7'}
	fc7.data.mapindex[1].mapindex[1].module.bias = Fcn8.net.modules[35].bias
	fc7.data.mapindex[1].mapindex[1].module.weight = Fcn8.net.modules[35].weight
	fc7.data.mapindex[1].mapindex[1].module.name = 'fc7'

	return nn.gModule({input}, {pool3,pool4,fc7})
end

function Fcn8.get_part2(nChannels)
	testo = 0
	if nChannels then
		testo = 1
	else
		nChannels = 21
	end

	local pool3 = nn.Identity()()
	local pool4 = nn.Identity()()
	local fc7 = nn.Identity()()
	
	local score_fr = nn.SpatialConvolutionMM(4096, nChannels, 1, 1, 1, 1, 0, 0)(fc7):annotate{name = 'score_fr'}
	if testo==0 then
		score_fr.data.module.bias = Fcn8.net.modules[38].bias
		score_fr.data.module.weight = Fcn8.net.modules[38].weight
	end
	score_fr.data.module.name = 'score_fr'

	local score2 = nn.Upsample(2, nChannels)(score_fr):annotate{name = 'score2'}
	score2.data.module.name = 'score2'

	local score_pool4 = nn.SpatialConvolutionMM(512, nChannels, 1, 1, 1, 1, 0, 0)(pool4):annotate{name = 'score_pool4'}
	if testo==0 then
		score_pool4.data.module.bias = Fcn8.net.modules[40].bias
		score_pool4.data.module.weight = Fcn8.net.modules[40].weight
	end
	score_pool4.data.module.name = 'score_pool4'

	local pool4_dist = -6
	local score_pool4c = nn.SpatialZeroPadding(pool4_dist, pool4_dist, pool4_dist, pool4_dist)(score_pool4):annotate{name = 'score_pool4c'}
	local score_fused = nn.CAddTable(){score2, score_pool4c}:annotate{name = 'score_fused'}
	
	local score4 = nn.Upsample(2, nChannels)(score_fused):annotate{name = 'score4'}
	score4.data.module.name = 'score4'

	local score_pool3 = nn.SpatialConvolutionMM(256, nChannels, 1, 1, 1, 1, 0, 0)(pool3):annotate{name = 'score_pool3'}
	if testo==0 then
		score_pool3.data.module.bias = Fcn8.net.modules[42].bias
		score_pool3.data.module.weight = Fcn8.net.modules[42].weight
	end
	score_pool3.data.module.name = 'score_pool3'

	local pool3_dist = -12
	local score_pool3c = nn.SpatialZeroPadding(pool3_dist, pool3_dist, pool3_dist, pool3_dist)(score_pool3):annotate{name = 'score_pool3c'}
	local fuse = nn.CAddTable(){score4, score_pool3c}:annotate{name = 'fuse'}

	return nn.gModule({pool3,pool4,fc7}, {fuse})
end

function Fcn8.get_part3(nChannels)
	if not nChannels then
		nChannels = 21
	end

	local input = nn.Identity()()

	local upsample = nn.Upsample(8, nChannels)(input):annotate{name = 'upsample'}
	upsample.data.module.name = 'upsample'

	local crop_dist = -6
	crop = nn.SpatialZeroPadding(crop_dist, crop_dist, crop_dist, crop_dist)(upsample):annotate{name = 'crop'}
	
	return nn.gModule({input}, {crop})
end


function Fcn8.format_part1(net)
	input = nn.Identity()()

	local conv1_1 = nn.ReLU()(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 100, 100)(input)):annotate{name = 'conv1_1'}
	if we_have_cudnn==1 then 	
		conv1_1 = cudnn.ReLU()(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 100, 100)(input)):annotate{name = 'conv1_1'}
	end
	node = getParameter(net, 'conv1_1')
	conv1_1.data.mapindex[1].module.bias = node.bias:clone()
	conv1_1.data.mapindex[1].module.weight = node.weight:clone()
	conv1_1.data.mapindex[1].module.name = 'conv1_1'
	local conv1_2 = nn.ReLU()(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1)(conv1_1)):annotate{name = 'conv1_2'}
	if we_have_cudnn==1 then 	
		conv1_2 = cudnn.ReLU()(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(conv1_1)):annotate{name = 'conv1_2'}
	end
	node = getParameter(net, 'conv1_2')
	conv1_2.data.mapindex[1].module.bias = node.bias:clone()
	conv1_2.data.mapindex[1].module.weight = node.weight:clone()
	conv1_2.data.mapindex[1].module.name = 'conv1_2'
	local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv1_2):annotate{name = 'pool1'}
	if we_have_cudnn==1 then 	
		pool1 = cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv1_2):annotate{name = 'pool1'}
	end
	local conv2_1 = nn.ReLU()(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1, 1)(pool1)):annotate{name = 'conv2_1'}
	if we_have_cudnn==1 then 	
		conv2_1 = cudnn.ReLU()(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(pool1)):annotate{name = 'conv2_1'}
	end
	node = getParameter(net, 'conv2_1')
	conv2_1.data.mapindex[1].module.bias = node.bias:clone()
	conv2_1.data.mapindex[1].module.weight = node.weight:clone()
	conv2_1.data.mapindex[1].module.name = 'conv2_1'
	local conv2_2 = nn.ReLU()(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 1, 1)(conv2_1)):annotate{name = 'conv2_2'}
	if we_have_cudnn==1 then 	
		conv2_2 = cudnn.ReLU()(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(conv2_1)):annotate{name = 'conv2_2'}
	end
	node = getParameter(net, 'conv2_2')
	conv2_2.data.mapindex[1].module.bias = node.bias:clone()
	conv2_2.data.mapindex[1].module.weight = node.weight:clone()
	conv2_2.data.mapindex[1].module.name = 'conv2_2'
	local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv2_2):annotate{name = 'pool2'}
	if we_have_cudnn==1 then 	
		pool2 = cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv2_2):annotate{name = 'pool2'}
	end
	local conv3_1 = nn.ReLU()(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1, 1, 1)(pool2)):annotate{name = 'conv3_1'}
	if we_have_cudnn==1 then 	
		conv3_1 = cudnn.ReLU()(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(pool2)):annotate{name = 'conv3_1'}
	end
	node = getParameter(net, 'conv3_1')
	conv3_1.data.mapindex[1].module.bias = node.bias:clone()
	conv3_1.data.mapindex[1].module.weight = node.weight:clone()
	conv3_1.data.mapindex[1].module.name = 'conv3_1'
	local conv3_2 = nn.ReLU()(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1, 1)(conv3_1)):annotate{name = 'conv3_2'}
	if we_have_cudnn==1 then 	
		conv3_2 = cudnn.ReLU()(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(conv3_1)):annotate{name = 'conv3_2'}
	end
	node = getParameter(net, 'conv3_2')
	conv3_2.data.mapindex[1].module.bias = node.bias:clone()
	conv3_2.data.mapindex[1].module.weight = node.weight:clone()
	conv3_2.data.mapindex[1].module.name = 'conv3_2'
	local conv3_3 = nn.ReLU()(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1, 1)(conv3_2)):annotate{name = 'conv3_3'}
	if we_have_cudnn==1 then 	
		conv3_3 = cudnn.ReLU()(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(conv3_2)):annotate{name = 'conv3_3'}
	end
	node = getParameter(net, 'conv3_3')
	conv3_3.data.mapindex[1].module.bias = node.bias:clone()
	conv3_3.data.mapindex[1].module.weight = node.weight:clone()
	conv3_3.data.mapindex[1].module.name = 'conv3_3'
	local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv3_3):annotate{name = 'pool3'}
	if we_have_cudnn==1 then 	
		pool3 = cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv3_3):annotate{name = 'pool3'}
	end
	local conv4_1 = nn.ReLU()(nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1)(pool3)):annotate{name = 'conv4_1'}
	if we_have_cudnn==1 then 	
		conv4_1 = cudnn.ReLU()(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(pool3)):annotate{name = 'conv4_1'}
	end
	node = getParameter(net, 'conv4_1')
	conv4_1.data.mapindex[1].module.bias = node.bias:clone()
	conv4_1.data.mapindex[1].module.weight = node.weight:clone()
	conv4_1.data.mapindex[1].module.name = 'conv4_1'
	local conv4_2 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv4_1)):annotate{name = 'conv4_2'}
	if we_have_cudnn==1 then 	
		conv4_2 = cudnn.ReLU()(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(conv4_1)):annotate{name = 'conv4_2'}
	end
	node = getParameter(net, 'conv4_2')
	conv4_2.data.mapindex[1].module.bias = node.bias:clone()
	conv4_2.data.mapindex[1].module.weight = node.weight:clone()
	conv4_2.data.mapindex[1].module.name = 'conv4_2'
	local conv4_3 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv4_2)):annotate{name = 'conv4_3'}
	if we_have_cudnn==1 then 	
		conv4_3 = cudnn.ReLU()(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(conv4_2)):annotate{name = 'conv4_3'}
	end	
	node = getParameter(net, 'conv4_3')
	conv4_3.data.mapindex[1].module.bias = node.bias:clone()
	conv4_3.data.mapindex[1].module.weight = node.weight:clone()
	conv4_3.data.mapindex[1].module.name = 'conv4_3'
	local pool4 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv4_3):annotate{name = 'pool4'}
	if we_have_cudnn==1 then 	
		pool4 = cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv4_3):annotate{name = 'pool4'}
	end

	local conv5_1 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(pool4)):annotate{name = 'conv5_1'}
	if we_have_cudnn==1 then 	
		conv5_1 = cudnn.ReLU()(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(pool4)):annotate{name = 'conv5_1'}
	end
	node = getParameter(net, 'conv5_1')
	conv5_1.data.mapindex[1].module.bias = node.bias:clone()
	conv5_1.data.mapindex[1].module.weight = node.weight:clone()
	conv5_1.data.mapindex[1].module.name = 'conv5_1'
	local conv5_2 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv5_1)):annotate{name = 'conv5_2'}
	if we_have_cudnn==1 then 	
		conv5_2 = cudnn.ReLU()(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(pool4)):annotate{name = 'conv5_2'}
	end
	node = getParameter(net, 'conv5_2')
	conv5_2.data.mapindex[1].module.bias = node.bias:clone()
	conv5_2.data.mapindex[1].module.weight = node.weight:clone()
	conv5_2.data.mapindex[1].module.name = 'conv5_2'
	local conv5_3 = nn.ReLU()(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1)(conv5_2)):annotate{name = 'conv5_3'}
	if we_have_cudnn==1 then 	
		conv5_3 = cudnn.ReLU()(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(conv5_2)):annotate{name = 'conv5_3'}
	end	
	node = getParameter(net, 'conv5_3')
	conv5_3.data.mapindex[1].module.bias = node.bias:clone()
	conv5_3.data.mapindex[1].module.weight = node.weight:clone()
	conv5_3.data.mapindex[1].module.name = 'conv5_3'
	local pool5 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv5_3):annotate{name = 'pool5'}
	if we_have_cudnn==1 then 	
		pool5 = cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(conv5_3):annotate{name = 'pool5'}
	end
	local fc6 = nn.Dropout(0.5)(nn.ReLU()(nn.SpatialConvolutionMM(512, 4096, 7, 7, 1, 1, 0, 0)(pool5))):annotate{name = 'fc6'}
	if we_have_cudnn==1 then 	
		fc6 = nn.Dropout(0.5)(cudnn.ReLU()(cudnn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 0, 0)(pool5))):annotate{name = 'fc6'}
	end
	node = getParameter(net, 'fc6')
	fc6.data.mapindex[1].mapindex[1].module.bias = node.bias:clone()
	fc6.data.mapindex[1].mapindex[1].module.weight = node.weight:clone()
	fc6.data.mapindex[1].mapindex[1].module.name = 'fc6'
	local fc7 = nn.Dropout(0.5)(nn.ReLU()(nn.SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1, 0, 0)(fc6))):annotate{name = 'fc7'}
	if we_have_cudnn==1 then 	
		fc7 = nn.Dropout(0.5)(cudnn.ReLU()(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0)(fc6))):annotate{name = 'fc7'}
	end
	node = getParameter(net, 'fc7')
	fc7.data.mapindex[1].mapindex[1].module.bias = node.bias:clone()
	fc7.data.mapindex[1].mapindex[1].module.weight = node.weight:clone()
	fc7.data.mapindex[1].mapindex[1].module.name = 'fc7'

	return nn.gModule({input}, {pool3,pool4,fc7})
end

function Fcn8.format_part2(net, nChannels)

	local pool3 = nn.Identity()()
	local pool4 = nn.Identity()()
	local fc7 = nn.Identity()()
	
	local score_fr = nn.SpatialConvolutionMM(4096, nChannels, 1, 1, 1, 1, 0, 0)(fc7):annotate{name = 'score_fr'}
	if we_have_cudnn==1 then 	
		score_fr = cudnn.SpatialConvolution(4096, nChannels, 1, 1, 1, 1, 0, 0)(fc7):annotate{name = 'score_fr'}
	end	
	node = getParameter(net, 'score_fr')
	score_fr.data.module.bias = node.bias:clone()
	score_fr.data.module.weight = node.weight:clone()
	score_fr.data.module.name = 'score_fr'

	local score2 = nn.Upsample(2, nChannels)(score_fr):annotate{name = 'score2'}
	score2.data.module.name = 'score2'

	local score_pool4 = nn.SpatialConvolutionMM(512, nChannels, 1, 1, 1, 1, 0, 0)(pool4):annotate{name = 'score_pool4'}
	if we_have_cudnn==1 then 	
		score_pool4 = cudnn.SpatialConvolution(512, nChannels, 1, 1, 1, 1, 0, 0)(pool4):annotate{name = 'score_pool4'}
	end	
	node = getParameter(net, 'score_pool4')
	score_pool4.data.module.bias = node.bias:clone()
	score_pool4.data.module.weight = node.weight:clone()
	score_pool4.data.module.name = 'score_pool4'
	
	local pool4_dist = -6
	local score_pool4c = nn.SpatialZeroPadding(pool4_dist, pool4_dist, pool4_dist, pool4_dist)(score_pool4):annotate{name = 'score_pool4c'}
	local score_fused = nn.CAddTable(){score2, score_pool4c}:annotate{name = 'score_fused'}
	
	local score4 = nn.Upsample(2, nChannels)(score_fused):annotate{name = 'score4'}
	score4.data.module.name = 'score4'

	local score_pool3 = nn.SpatialConvolutionMM(256, nChannels, 1, 1, 1, 1, 0, 0)(pool3):annotate{name = 'score_pool3'}
	if we_have_cudnn==1 then 	
		score_pool3 = cudnn.SpatialConvolution(256, nChannels, 1, 1, 1, 1, 0, 0)(pool3):annotate{name = 'score_pool3'}
	end	
	node = getParameter(net, 'score_pool3')
	score_pool3.data.module.bias = node.bias:clone()
	score_pool3.data.module.weight = node.weight:clone()
	score_pool3.data.module.name = 'score_pool3'

	local pool3_dist = -12
	local score_pool3c = nn.SpatialZeroPadding(pool3_dist, pool3_dist, pool3_dist, pool3_dist)(score_pool3):annotate{name = 'score_pool3c'}
	local fuse = nn.CAddTable(){score4, score_pool3c}:annotate{name = 'fuse'}

	return nn.gModule({pool3,pool4,fc7}, {fuse})
end


return Fcn8

