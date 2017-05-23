require 'torch'
require 'nn'
require 'rnn'

channels = actionParam.numHiddenChannelsInRecursion

function build_network(inputChannels, upscaleFactor, numRecursions)

    -- the upscale network
    local net1 = nn.Sequential()
    net1:add(nn.SpatialConvolution(inputChannels, 6, 11, 11, 1, 1, 5, 5))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(6, 6, 7, 7, 1, 1, 3, 3))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(6, 12, 3, 3, 1, 1, 1, 1))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(12, 16, 5, 5, 1, 1, 2, 2))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
    net1:add(nn.ReLU())
    net1:add(nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1))
    net1:add(nn.PixelShuffle(upscaleFactor))

    -- inner part of the residual network
    local innerNet = nn.Sequential()
    innerNet:add(nn.SpatialConvolution(3, channels, 9, 9, 1, 1, 4, 4))
    innerNet:add(nn.ReLU())
    innerNet:add(nn.SpatialConvolution(channels, channels, 5, 5, 1, 1, 2, 2))
    innerNet:add(nn.ReLU())
    innerNet:add(nn.SpatialConvolution(channels, 3, 5, 5, 1, 1, 2, 2))
    innerNet:add(nn.ReLU())

    -- forward input to the end where the residual will be added
    local resNet = nn.ConcatTable()
    resNet:add(nn.Identity())
    resNet:add(innerNet)

    -- inner part of recurrent net is the residual net
    local inside_recurrent = nn.Sequential()
    inside_recurrent:add(resNet)
    inside_recurrent:add(nn.CAddTable())

    local recurrent = nn.Recurrent(
        nn.Identity(),      -- start
        nn.Identity(),      -- input transform 
        inside_recurrent,   -- feedback
        nn.Identity(),      -- transfer
        numRecursions,      -- rho
        nn.SelectTable(2)   -- merge
    )

    -- decorator, recursively apply to the same input (not multiple inputs)
    local rnn = nn.Sequential()
    rnn:add(nn.Repeater(recurrent, numRecursions))
    rnn:add(nn.SelectTable(numRecursions)) -- select last output from Repeater

    local net = nn.Sequential()
    net:add(net1)
    net:add(rnn)

    return net
end