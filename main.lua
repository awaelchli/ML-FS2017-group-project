local toload = arg[1] or 'full_network'
local actionType = arg[2] or 'train'

require 'parameters.'..toload..'.lua'

if actionType == 'create' do
	dofile(actionParam.create..'.lua')
elseif actionType == 'train' do
	dofile(actionParam.train..'.lua')
elseif actionType == 'test' do
	dofile(actionParam.test..'.lua')
else
	assert(false, "Action type '"..actionType.."' unknown")
end