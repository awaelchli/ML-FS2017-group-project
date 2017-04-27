
require 'nn'
--require 'nngraph'
require 'load_images'
require 'build_network'
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

-- Set up Logger

--nngraph.setDebug(true)

logger = optim.Logger('logs/loss_log.txt')
loggerGrad = optim.Logger('logs/grad_norm_log.txt')
loggerGrad:setNames{'Gradient norm'}
loggerGrad:style{'-'}

-- Load data
images = load_images.load('datasets/Set14/image_SRF_4/', 'png', false)
inputChannels = 3

-- Convert greyscale images to RGB
for i = 1, #images do
        images[i] = images[i]:expand(inputChannels, images[i]:size(2), images[i]:size(3))
end

n = #images / 2

imagesLR = {}
imagesHR = {}

for i = 1, n do
	imagesLR[i] = images[2 * i]
	imagesHR[i] = images[2 * i - 1]
end

data = {}
data.HR = imagesHR
data.LR = imagesLR
data.size = function()
    return #data.LR
end

train, validation, test = split_data(data, 0.8, 0.1)

-- Make network
upscaleFactor = 4
num_recursions = 5

if(true) then
    net = build_network(inputChannels, upscaleFactor, num_recursions)

    out = net:forward(imagesLR[1])
    print(out:size())
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
   if _nidx_ > #imagesHR then _nidx_ = 1 end

   --local sample = data[_nidx_]
   local target = imagesHR[_nidx_]      -- this funny looking syntax allows
   local inputs = imagesLR[_nidx_]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accommodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(net:forward(inputs), target)
   net:backward(inputs, criterion:backward(net.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-2,
   weightDecay = 0,
   momentum = 0
}

for i = 1, 1000 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1, #imagesHR do
      
      _,fs = optim.sgd(feval,x,sgd_params)


      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / #imagesHR
   
   current_abs_grad = torch.norm( dl_dx )
   
    --if i%10 == 0 then
        print('i'..i..' loss = ' .. current_loss .. ' grad norm = ' .. current_abs_grad)
    --end
   
   logger:add{['training error'] = current_loss}
   logger:style{['training error'] = '-'}
   if i%10 == 0 then
   logger:plot()  
   end
   
   loggerGrad:add{current_abs_grad}
   if i%10 == 0 then
   	loggerGrad:plot()
   end
end

-- [[
torch.save("out/upscaleDeConv.model", saveNet)
print("Model saved")
--]]
print("finished")

-- Test example

imgselector = 7

local origin = imagesLR[imgselector]
local gt = imagesHR[imgselector]
local test = net:forward(origin)
--[[
local scaled = image.scale(origin, gt:size(3), gt:size(2), 'simple') -- upscaled(nearest neighbor) LR
local scaledbl = image.scale(origin, gt:size(3), gt:size(2), 'bilinear') -- upscaled(nearest neighbor) LR
local scaledbc = image.scale(origin, gt:size(3), gt:size(2), 'bicubic') -- upscaled(nearest neighbor) LR
local diff = torch.add(gt, -1, test)
local diff2 = torch.add(scaledbc, -1, test)
--]]
image.save("out/test.png", test)
