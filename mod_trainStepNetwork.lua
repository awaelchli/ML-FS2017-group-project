require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


upscaleFactor = actionParam.upscaleFactor
num_recursions = actionParam.numRecursions
sgd_params = actionParam.sgd_params
epochs = actionParam.epochs

saveInterval = actionParam.saveInterval
analysisInterval = actionParam.analysisInterval


-- Set up Logger
loss_logger = optim.Logger('logs/loss.log')
loss_logger:style{'+-', '+-','+-', '+-'}
loss_logger:setNames{'Training loss1', 'Training loss2', 'Validation loss1', 'Validation loss2'}
--loss_logger:display(false) -- only save, but not display

grad_logger = optim.Logger('logs/grad_norm.log')
grad_logger:setNames{'Gradient1 norm', 'Gradient1 norm'}
grad_logger:style{'-','-'}
--grad_logger:display(false) -- only save, but not display


-- Load Network
net = torch.load("out/"..actionParam.name..".model")
net1 = net:get(1)
net2 = net:get(2)
local net1paramsize, net1gradsize = net1:getParameters()
net1paramsize = net1paramsize:nElement()
net1gradsize = net1gradsize:nElement()

saveNet = net:clone('weight','bias','gradWeight','gradBias')
saveNet:clearState() --if it wasn't clean, clean it
netUnion = nn.Container()
netUnion:add(net)
netUnion:add(saveNet)
x, dl_dx = netUnion:getParameters()
x1 = x[{{1,net1paramsize}}]
x2 = x[{{net1paramsize+1, x:nElement()}}]
dl_dx1 = dl_dx[{{1,net1gradsize}}]
dl_dx2 = dl_dx[{{net1gradsize+1, dl_dx:nElement()}}]


-- Train network
criterion = nn.MSECriterion()

intermediatePic = {}

feval1 = function(x1_new)

    if x1 ~= x1_new then
        x1:copy(x1_new)
    end

    -- select a new training sample
    _nidx1_ = (_nidx1_ or 0) + 1
    if _nidx1_ > train.size() then _nidx1_ = 1 end

    local target = train.HR[_nidx1_]
    local input = train.LR[_nidx1_]

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    dl_dx1:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x1 = criterion:forward(net1:forward(input), target)
    intermediatePic[_nidx1_] = net1.output
    net1:backward(input, criterion:backward(net1.output, target))

    -- return loss(x) and dloss/dx
    return loss_x1, dl_dx1
end

feval2 = function(x2_new)

    if x2 ~= x2_new then
        x2:copy(x2_new)
    end

    -- select a new training sample
    _nidx2_ = (_nidx2_ or 0) + 1
    if _nidx2_ > train.size() then _nidx2_ = 1 end

    local target = train.HR[_nidx2_]
    local input = intermediatePic[_nidx2_]

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    dl_dx2:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x2 = criterion:forward(net2:forward(input), target)
    net2:backward(input, criterion:backward(net2.output, target))

    -- return loss(x) and dloss/dx
    return loss_x2, dl_dx2
end

print("Start training (up to "..epochs.." epochs)")

for i = 1, epochs do

    train_loss1 = 0
    train_loss2 = 0

    -- Loop over all training samples (one epoch)
    -- Run optimization and compute training loss
    for j = 1, train.size() do
        _, fs = optim.sgd(feval1, x1, sgd_params)
        train_loss1 = train_loss1 + fs[1]
        _, fs = optim.sgd(feval2, x2, sgd_params)
        train_loss2 = train_loss2 + fs[1]
    end
    train_loss1 = train_loss1 / train.size()
    train_loss2 = train_loss2 / train.size()

    -- Compute gradient norm
    current_abs_grad1 = torch.norm(dl_dx1)
    current_abs_grad2 = torch.norm(dl_dx2)
    grad_logger:add{current_abs_grad1, current_abs_grad2}
    if saveInterval and i % saveInterval == 0 then
        grad_logger:plot()
    end

    -- Compute validation error
    validation_loss1 = 0
    psnr1 = 0
	ssim1 = 0
    validation_loss2 = 0
    psnr2 = 0
	ssim2 = 0
    for j = 1, validation.size() do
        local target = validation.HR[j]
        local input = validation.LR[j]
        local approx1 = net1:forward(input)
        local approx2 = net2:forward(approx1)
        validation_loss1 = validation_loss1 + criterion:forward(approx1, target)
        validation_loss2 = validation_loss2 + criterion:forward(approx2, target)
        if analysisInterval and i % analysisInterval == 0 then
			psnr1 = psnr1 + PSNR(target, approx1)
			ssim1 = ssim1 + SSIM(target, approx1)
			psnr2 = psnr2 + PSNR(target, approx2)
			ssim2 = ssim2 + SSIM(target, approx2) 
		end
    end
    validation_loss1 = validation_loss1 / validation.size()
    validation_loss2 = validation_loss2 / validation.size()
    if analysisInterval and i % analysisInterval == 0 then
    	psnr1 = psnr1 / validation.size()
		ssim1 = ssim1 / validation.size()
    	psnr2 = psnr2 / validation.size()
		ssim2 = ssim2 / validation.size()
	end
    
    -- Report training and validation loss
    loss_logger:add{train_loss1, train_loss2, validation_loss1, validation_loss2}
    if saveInterval and i % saveInterval == 0 then
        loss_logger:plot()  
    end

    -- Print to console
		print('i'..i..'\tloss1 = ' .. train_loss1 .. '\tgrad1 norm = ' .. current_abs_grad1)
		print('  \tloss2 = ' .. train_loss2 .. '\tgrad2 norm = ' .. current_abs_grad2)
    if analysisInterval and i % analysisInterval == 0 then
		print('  \tpsnr1 = ' .. psnr1 .. '   \tssim1 = ' .. ssim1)
		print('  \tpsnr2 = ' .. psnr2 .. '   \tssim2 = ' .. ssim2)
	end
    
    -- Periodically save network
    if saveInterval and i % saveInterval == 0 then
        torch.save("out/"..actionParam.name..".model", saveNet)
        print("Model saved")
    end
end


torch.save("out/"..actionParam.name..".model", saveNet)
print("Model saved")
print("Finished")