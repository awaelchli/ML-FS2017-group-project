
require 'nn'
require 'nngraph'
require 'load_images'
require 'torch'
require 'optim'

-- Set up Logger

nngraph.setDebug(true)
logger = optim.Logger('loss_log.txt')

-- Load data

images = load_images.load('datasets/Set14/image_SRF_4/', 'png', false)
inputChannels = 3

-- Convert greyscale images
for i = 1, #images do
        images[i] = images[i]:expand(inputChannels, images[i]:size(2), images[i]:size(3))
end

n = #images / 2

imagesLR = {}--images:select()
imagesHR = {}--images[2]

for i = 1, n do
	imagesLR[i] = images[2 * i]
	imagesHR[i] = images[2 * i - 1]
end

-- Make network
--[[
upscaleFactor = 4

net = nn.Sequential()
net:add(nn.SpatialConvolution(inputChannels, 6, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6, 6, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6, 32, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1))
net:add(nn.PixelShuffle(upscaleFactor))

--net:add(nn.SpatialFullConvolution(32, 3, 9, 9, upscaleFactor, upscaleFactor, 3, 3, 1, 1))

--graph.dot(net.fg, 'cnn1', 'cnn1')
]]

-- Load network from disk
net = torch.load("upscaleDeConv.model")

-- Train network

criterion = nn.MSECriterion()

x, dl_dx = net:getParameters()

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
   learningRate = 1e-3,
   learningRateDecay = 0, --1e-4,
   weightDecay = 0,
   momentum = 0
}

for i = 1,200 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1, #imagesHR do
      
      _,fs = optim.sgd(feval,x,sgd_params)


      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / #imagesHR
   
    --if i%10 == 0 then
        print('i'..i..' loss = ' .. current_loss)
    --end
   
   logger:add{['training error'] = current_loss}
   logger:style{['training error'] = '-'}
   if i%100 == 0 then
   logger:plot()  
   end
end

-- [[
torch.save("upscaleDeConv.model", net)
print("Model saved")
--]]
print("finished")

-- Test example

imgselector = 7

local origin = imagesLR[imgselector]
local gt = imagesHR[imgselector]
local test = net:forward(origin)
local scaled = image.scale(origin, gt:size(3), gt:size(2), 'simple') -- upscaled(nearest neighbor) LR
local scaledbl = image.scale(origin, gt:size(3), gt:size(2), 'bilinear') -- upscaled(nearest neighbor) LR
local scaledbc = image.scale(origin, gt:size(3), gt:size(2), 'bicubic') -- upscaled(nearest neighbor) LR
local diff = torch.add(gt, -1, test)
local diff2 = torch.add(scaledbc, -1, test)

image.save("test.png", test)

-- [[
--itorch.image(.....) -- difference between output and LR
itorch.image(scaled) -- upscaled LR
--itorch.image(scaledbl) -- upscaled(bilinear) LR
--itorch.image(scaledbc) -- upscaled(bicubic) LR
itorch.image(test) -- output 
itorch.image(gt) -- HR
--itorch.image(diff) -- difference between output and HR
--]]

--[[
itorch.image({image.crop(scaled, 'bl', gt:size(3)/2, gt:size(2)/2),image.crop(scaledbc, 'bl', gt:size(3)/2, gt:size(2)/2)})
itorch.image({image.crop(gt, 'bl', gt:size(3)/2, gt:size(2)/2),image.crop(test, 'bl', gt:size(3)/2, gt:size(2)/2)})
itorch.image(diff) -- difference between output and HR
itorch.image(diff2) -- difference between output and upscaleBC
--]]
