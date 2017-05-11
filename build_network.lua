require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'

function build_network(inputChannels, upscaleFactor, numRecursions)

    -- the upscale network
    local netStart = nn.SpatialConvolution(inputChannels, 6, 3, 3, 1, 1, 1, 1)()
    local netEnd = netStart
    netEnd = nn.ReLU()(netEnd)
    netEnd = nn.SpatialConvolution(6, 6, 3, 3, 1, 1, 1, 1)(netEnd)
    netEnd = nn.ReLU()(netEnd)
    netEnd = nn.SpatialConvolution(6, 32, 5, 5, 1, 1, 2, 2)(netEnd)
    netEnd = nn.ReLU()(netEnd)
    netEnd = nn.SpatialConvolution(32, inputChannels * upscaleFactor * upscaleFactor, 3, 3, 1, 1, 1, 1)(netEnd)
    netEnd = nn.PixelShuffle(upscaleFactor)(netEnd)

    -- inner part of the residual network
    local innerNet = nn.Sequential()
    innerNet:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    innerNet:add(nn.ReLU())
    innerNet:add(nn.SpatialConvolution(32, 3, 5, 5, 1, 1, 2, 2))
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
        nn.SelectTable(2)	-- merge
    )

    -- decorator, recursively apply to the same input (not multiple inputs)
    netEnd = nn.Repeater(recurrent, numRecursions)(netEnd)
    netEnd = nn.SelectTable(numRecursions)(netEnd) -- select last output from RNN

    local net = nn.gModule({netStart},{netEnd})

    --graph.graphvizFile(innerNet.fg, "dot", "out/innerNet.svg")
    graph.graphvizFile(net.fg, "dot", "out/net.svg")

    return net--, innerNet
end