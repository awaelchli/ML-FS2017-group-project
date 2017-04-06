--require 'nngraph'
require 'torch'
require 'nn'
require 'rnn'
require 'dpnn'
require 'optim'
require 'load_images'

-- define inputs
--local x1 = torch.rand(3,5,5)

images = load_images.load('datasets/Set14/image_SRF_4/', 'png', false)
inputChannels = 3

-- Convert greyscale images
for i = 1, #images do
        images[i] = images[i]:expand(inputChannels, images[i]:size(2), images[i]:size(3))
end
x1 = images[1]


-- model implementation, define an MLP


local input = nn.Identity()
inputChannels = 3 -- RGB
kernelSize = 3
step = 1
pad = 1 --math.ceil((kernelSize-1)/2)

inner = nn.Sequential()
inner:add(nn.SpatialConvolution(inputChannels, inputChannels, kernelSize, kernelSize, step, step, 1, 1))
inner:add(nn.ReLU())

--print(input)
--print(inner)

recurrent = nn.Recurrent(inner, input, nn.Identity(),nn.Identity(), 1)

out = nn.Identity()
recurrent:add(out)




--local mlp = nn.Sequential()
--mlp:add(h)





-- print the size of your outputs
outputs = recurrent:forward(x1)
--graph.dot(mlp.fg, 'Graph','test_recurrent')
print(outputs:size())
image.save('rec_test.png', outputs)
