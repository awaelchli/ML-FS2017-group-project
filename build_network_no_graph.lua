require 'torch'
require 'nn'
require 'rnn'

function build_network(inputChannels, upscaleFactor, numRecursions)

    -- the upscale network
    local net1 = nn.Sequential()
    net1:add(nn.SpatialConvolution(inputChannels, 6, 3, 3, 1, 1, 1, 1))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(6, 6, 3, 3, 1, 1, 1, 1))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(6, 32, 5, 5, 1, 1, 2, 2))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1))
    net1:add(nn.PixelShuffle(upscaleFactor))

    -- inner part of the residual network
    local innerNet = nn.Sequential()
    innerNet:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    innerNet:add(nn.ReLU())
    innerNet:add(nn.SpatialConvolution(32, 3, 5, 5, 1, 1, 2, 2))
    innerNet:add(nn.ReLU())

    -- forward input to the end where the residual will be added
    local resNet = nn.ConcatTable()
    resNet:add(nn.Identity())
    resNet.add(innerNet)

    -- inner part of recurrent net is the residual net
    local inside_recurrent = nn.Sequential()
    inside_recurrent:add(resNet)
    inside_recurrent:add(nn.CAddTable())

    local recurrent = nn.Recurrent(
        nn.Identity(),      -- start
        nn.Identity(),      -- input transform 
        inside_recurrent,   -- hidden network 
        nn.Identity(),      -- feedback
        numRecursions,      -- rho
        nn.SelectTable(2)
    )

    -- decorator, recursively apply to the same input (not multiple inputs)
    rnn = nn.Repeater(recurrent, numRecursions)

    local net = nn.Sequential()
    net:add(net1)
    net:add(rnn)
    net:add(nn.SelectTable(numRecursions)) -- select last output from RNN

    return net
end