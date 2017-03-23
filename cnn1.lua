require 'nn'
require 'nngraph'
require 'load_images'
require 'torch'
require 'optim'

images = load_images.load('datasets/Set14/image_SRF_4/', 'png')
n = #images / 2

imagesLR = {}--images:select()
imagesHR = {}--images[2]

even = 2 * torch.range(1, n)
odd = 2 * torch.range(1, n) - torch.ones(n)
for i = 1, n do
	imagesLR[i] = images[odd[i]]
	imagesHR[i] = images[even[i]]
end

inputChannels = 3
upscaleFactor = 4

net = nn.Sequential()
net:add(nn.SpatialConvolution(inputChannels, 8, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())

net:add(nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1))
net:add(nn.PixelShuffle(upscaleFactor))


criterion = nn.MSECriterion()

-- 4. Train the net

-- To minimize the loss defined above, using the linear net defined
-- in 'net', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex net, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:

x, dl_dx = net:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our net, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > #imagesHR then _nidx_ = 1 end

   --local sample = data[_nidx_]
   local target = imagesHR[_nidx_]      -- this funny looking syntax allows
   local inputs = imagesLR[_nidx_]    -- slicing of arrays.

   print(#inputs)
   print(#target)

   -- reset gradients (gradients are always accumulated, to accommodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(net:forward(inputs), target)
   net:backward(inputs, criterion:backward(net.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the net using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

-- we cycle 1e4 times over our training data
for i = 1,1e4 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1, #imagesHR do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)
   
   logger:add{['training error'] = current_loss}
   logger:style{['training error'] = '-'}
   if i%200 == 0 then
   logger:plot()  
   end
end