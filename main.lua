require 'torch'
require 'nn'

if not package.loaded['nn.PixelShuffle'] then
    require 'PixelShuffle'
end

require 'optim'
require 'rnn'
require 'dpnn'
require 'nngraph'
require 'gnuplot'
require 'environment'

cmd = torch.CmdLine()
cmd:option('-param', 'full_network', 'pass name of parameter file to be used (without ".lua")')
cmd:option('-type', 'train', 'select what to do (create/train/test)')
cmd:option('-name', '0', 'name of the save file for the network (default is same as -param)')
cmd:option('-epochs', 0, 'maximum number of epochs to train (default taken from param file)')

local argv = cmd:parse(arg)

-- Load the parameters
require('parameters.'..argv.param)

-- Setup environment (create folders for results etc.)
environment.setup(actionParam)

if argv.name == '0' then
	actionParam.name = argv.param
else
	actionParam.name = argv.name
	actionParam.folders.testResults = actionParam.folders.output .. 'results/' .. argv.name .. '/'
	actionParam.folders.logs = 'logs/' .. argv.name .. '/'
end

if argv.epochs ~= 0 then
	actionParam.epochs = argv.epochs
end

print('Parameter set:')
print(actionParam)

if argv.type == 'create' then
	dofile('mod_'..actionParam.create..'.lua')
elseif argv.type == 'train' then
	dofile('mod_'..actionParam.loadTrainData..'.lua')
	dofile('mod_'..actionParam.train..'.lua')
elseif argv.type == 'test' then
	dofile('mod_'..actionParam.loadTestData..'.lua')
	dofile('mod_'..actionParam.test..'.lua')
else
	assert(false, "type '"..actionType.."' unknown")
end