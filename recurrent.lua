require 'nngraph'
require 'torch'
require 'nn'
require 'rnn'
require 'dpnn'
require 'optim'

-- define inputs
local x1 = torch.rand(5,5,3)


-- model implementation, define an MLP


local input = nn.Identity()():annotate{name = 'Input'}
inputChannels = 3 -- RGB
kernelSize = 3
step = 1
pad = 1 --math.ceil((kernelSize-1)/2)

inner = nn.Sequential()
inner:add(nn.SpatialConvolution(inputChannels, inputChannels, kernelSize, kernelSize, step, step, 1, 1))
inner:add(nn.ReLU())

c = input:sharedClones()

recurrent = nn.Recurrent(inner, input, nn.Identity()())

out = identity()()
recurrent:add(out)




--local mlp = nn.Sequential()
--mlp:add(h)
mlp = nn.gModule({input},{out})




-- print the size of your outputs
outputs = mlp:forward({x1})
graph.dot(mlp.fg, 'Graph','test_recurrent')
print(outputs:size())
