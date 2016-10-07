require 'torch'   -- torch
require 'image'
require 'nn'      -- provides all sorts of trainable modules/layers
require 'nngraph'
require 'cunn'
require 'util.misc'
require 'Upsample'
require 'MatchCriterion'
ConvLSTM = require 'ConvLSTM'
require 'create_lstm_protos'
require 'create_cnn'

model_utils = require 'util.model_utils'
plants = require 'plants_utils'

require 'optim'

function getParameter(nngraph_model, name)
    local params
    nngraph_model:apply( function(m) if m.name==name then params = m end end)
    return params
end
