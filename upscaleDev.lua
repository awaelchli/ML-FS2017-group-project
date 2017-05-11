
require 'nn'

if not package.loaded['nn.PixelShuffle'] then
    require 'PixelShuffle'
end

--require 'nngraph'
require 'load_images'
require 'prepare_data'
require 'build_network'
--require 'build_network_no_graph'
require 'split_data'
require 'torch'
require 'optim'
require 'gnuplot'
require 'rnn'
require 'dpnn'
require 'paths'


-- Setup environment
paths.mkdir('logs')
paths.mkdir('out')
--nngraph.setDebug(true)

-- Set up Logger
loss_logger = optim.Logger('logs/loss.log')
loss_logger:style{'+-', '+-'}
loss_logger:setNames{'Training loss', 'Validation loss'}
--loss_logger:display(false) -- only save, but not display

grad_logger = optim.Logger('logs/grad_norm.log')
grad_logger:setNames{'Gradient norm'}
grad_logger:style{'-'}
--grad_logger:display(false) -- only save, but not display

-- Load dataset
test_separate = true
images = load_images.load('datasets/Set14/image_SRF_4/', 'png', false)

if test_separate then
    test_images = load_images.load('datasets/Set5/image_SRF_4/', 'png', false)
end


-- Preprocess dataset
data = prepare_data(images)
if test_separate then
    train, validation = split_data(data, 0.8, 0.2)
    test = prepare_data(test_images)
else
    train, validation, test = split_data(data, 0.8, 0.1)
end

-- Make network
upscaleFactor = 4
num_recursions = 5

if(true) then
    -- Build network from scratch
    net = build_network(data.channels(), upscaleFactor, num_recursions)
else
    -- Load network from disk
    net = torch.load("out/upscaleDeConv.model")
end

saveNet = net:clone('weight','bias','gradWeight','gradBias')
saveNet:clearState() --if it wasn't clean, clean it
netUnion = nn.Container()
netUnion:add(net)
netUnion:add(saveNet)
x, dl_dx = netUnion:getParameters()

-- Train network
criterion = nn.MSECriterion()

feval = function(x_new)

    if x ~= x_new then
        x:copy(x_new)
    end

    -- select a new training sample
    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > train.size() then _nidx_ = 1 end

    local target = train.HR[_nidx_]
    local input = train.LR[_nidx_]

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x = criterion:forward(net:forward(input), target)
    net:backward(input, criterion:backward(net.output, target))

    -- return loss(x) and dloss/dx
    return loss_x, dl_dx
end

sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

epochs = 10000

for i = 1, epochs do

    train_loss = 0

    -- Loop over all training samples (one epoch)
    -- Run optimization and compute training loss
    for i = 1, train.size() do
        _, fs = optim.sgd(feval, x, sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / train.size()

    -- Compute gradient norm
    current_abs_grad = torch.norm(dl_dx)
    grad_logger:add{current_abs_grad}
    if i % 10 == 0 then
        grad_logger:plot()
    end

    -- Compute validation error
    validation_loss = 0
    for i = 1, validation.size() do
        local target = validation.HR[i]
        local input = validation.LR[i]
        validation_loss = validation_loss + criterion:forward(net:forward(input), target)
    end
    validation_loss = validation_loss / validation.size()
    
    -- Report training and validation loss
    loss_logger:add{train_loss, validation_loss}
    if i % 10 == 0 then
        loss_logger:plot()  
    end

    -- Print to console
    print('i'..i..' loss = ' .. train_loss .. ' grad norm = ' .. current_abs_grad)
end


torch.save("out/upscaleDeConv.model", saveNet)
print("Model saved")
print("finished")

-- Test example

test_loss = 0
for i = 1, test.size() do
    local target = test.HR[i]
    local input = test.LR[i]
    test_loss = test_loss + criterion:forward(net:forward(input), target)
end
test_loss = test_loss / test.size()

print('Loss on Testset: ', test_loss)

imgselector = 1

local origin = test.LR[imgselector]
local gt = test.HR[imgselector]
local test_forward = net:forward(origin)
--[[
local scaled = image.scale(origin, gt:size(3), gt:size(2), 'simple') -- upscaled(nearest neighbor) LR
local scaledbl = image.scale(origin, gt:size(3), gt:size(2), 'bilinear') -- upscaled(nearest neighbor) LR
local scaledbc = image.scale(origin, gt:size(3), gt:size(2), 'bicubic') -- upscaled(nearest neighbor) LR
local diff = torch.add(gt, -1, test)
local diff2 = torch.add(scaledbc, -1, test)
--]]
image.save("out/test.png", test_forward)
