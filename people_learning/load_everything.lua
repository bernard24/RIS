require 'nn'
require 'cunn'
--
--require 'cutorch'
--if we_have_cudnn==1 then
	require 'cudnn'
	cudnn.benchmark = true
	cudnn.fastest = true
	--cudnn.verbose = true
--else
--	require 'cunn'
--end
--require 'inn'

torch.setdefaulttensortype('torch.FloatTensor')

require 'image'                                                           
require 'nnx'
require 'nngraph'
require 'Upsample'
model_utils = require 'util.model_utils'
Fcn8 = require 'Fcn8'
require 'optim'
require 'misc'
require 'create_lstm_protos'
require 'IoU3Criterion'
require 'IoU5Criterion'
require 'IoUMultiClassCriterion'
require 'MatchCriterion2'

require 'MatchClassesCriterion'
require 'create_classes_lstm_protos'
require 'create_classes_lstm_protos_2'

require 'get_instance'

function getParameter(nngraph_model, name)
    local params
    nngraph_model:apply( function(m) if m.name==name then params = m end end)
    return params
end
