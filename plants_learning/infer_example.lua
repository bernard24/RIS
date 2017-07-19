require 'torch'   -- torch
require 'image'
require 'nn'	  -- provides all sorts of trainable modules/layers
require 'cunn'
require 'nngraph'
require 'util.misc'
ConvLSTM = require 'ConvLSTM'
model_utils = require 'util.model_utils'
require 'IoU4Criterion'  -- It is MatchCriterion with a different name

input_path = 'LSCData/A1/' -- Change the path to where your data are

seq_length = 30
xSize = 106
ySize = 100
rnn_layers = 2
rnn_size = 30

list_file = io.open(input_path .. 'data_list.txt', "r")

gpumode = 1

function createInstance ()
    io.input(list_file)
    input_file = io.read()
    if input_file==nil then
        return nil
    end
    print(input_file)
    local input_image = image.load(input_path .. input_file):sub(1,3)

    input_image = torch.reshape(image.scale(input_image, ySize, xSize):float(), 3, xSize, ySize)

    if gpumode==1 then
        input_image = input_image:cuda()
    end

    return input_image
end

input= createInstance()
-- itorch.image(input:sub(1,3))


model = torch.load('plants_pre_lstm.model')
protos = torch.load('plants_convlstm.model')

x = model:forward(input)

-- the initial state of the cell/hidden states
local init_state_global = {}
for L=1,rnn_layers do
    local h_init = torch.zeros(rnn_size,xSize, ySize)
    if gpumode==1 then h_init = h_init:cuda() end
    table.insert(init_state_global, h_init:clone())
    table.insert(init_state_global, h_init:clone())
end

local current_state = {}
current_state = init_state_global
prediction = {}           -- softmax outputs
solutions = torch.zeros(seq_length, xSize, ySize)
local counter = 0
for t=1,seq_length do
    local lst = protos.rnn:forward{x, unpack(current_state)}

    current_state = {}
    for i=1,#init_state_global do table.insert(current_state, lst[i]) end -- extract the state, without output
    local prediction = lst[#lst] -- last element is the prediction
    local postlst = protos.post_lstm:forward(prediction)
    output = postlst[1]:clone()
    output:resize(1,xSize,ySize)
    degree = postlst[2]

--    itorch.image(output)
    print(degree[1])

    if degree[1]<0.5 then
       break
    end
    counter = counter+1

    solutions[{t,{},{}}] = output:double()
end
canvas = torch.zeros(xSize,ySize)
for t=counter, 1, -1 do
    canvas[solutions:sub(t,t):gt(0.9)] = t
end
print(counter)

-- itorch.image(canvas)
