require 'nn'
require 'cunn'

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
require 'MatchCriterion'

function getParameter(nngraph_model, name)
    local params
    nngraph_model:apply( function(m) if m.name==name then params = m end end)
    return params
end
