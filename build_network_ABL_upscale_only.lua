require 'torch'
require 'nn'
require 'rnn'

-- Ablation Study 2

function build_network(inputChannels, upscaleFactor, numRecursions)

    -- the upscale network
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(inputChannels, 6, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(6, 6, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(6, 32, 5, 5, 1, 1, 2, 2))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1))
    net:add(nn.PixelShuffle(upscaleFactor))

    return net
end