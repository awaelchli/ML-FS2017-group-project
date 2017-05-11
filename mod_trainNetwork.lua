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
loss_logger:style{'+-', '+-'}
loss_logger:setNames{'Training loss', 'Validation loss'}
loss_logger:display(false) -- only save, but not display

grad_logger = optim.Logger('logs/grad_norm.log')
grad_logger:setNames{'Gradient norm'}
grad_logger:style{'-'}
grad_logger:display(false) -- only save, but not display


-- Load Network
net = torch.load("out/"..actionParam.name..".model")

saveNet = net:clone('weight','bias','gradWeight','gradBias')
saveNet:clearState() --if it wasn't clean, clean it
netUnion = nn.Container()
netUnion:add(net)
netUnion:add(saveNet)
x, dl_dx = netUnion:getParameters()


-- Train network
criterion = nn.MSECriterion()

feval = function(x_new)

    if x ~= x_new then
        x:copy(x_new)
    end

    -- select a new training sample
    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > train.size() then _nidx_ = 1 end

    local target = train.HR[_nidx_]
    local input = train.LR[_nidx_]

    -- reset gradients (gradients are always accumulated, to accommodate batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x = criterion:forward(net:forward(input), target)
    net:backward(input, criterion:backward(net.output, target))

    -- return loss(x) and dloss/dx
    return loss_x, dl_dx
end

print("Start training (up to "..epochs.." epochs)")

for i = 1, epochs do

    train_loss = 0

    -- Loop over all training samples (one epoch)
    -- Run optimization and compute training loss
    for j = 1, train.size() do
        _, fs = optim.sgd(feval, x, sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / train.size()

    -- Compute gradient norm
    current_abs_grad = torch.norm(dl_dx)
    grad_logger:add{current_abs_grad}
    if saveInterval and i % saveInterval == 0 then
        grad_logger:plot()
    end

    -- Compute validation error
    validation_loss = 0
    psnr = 0
	ssim = 0
    for j = 1, validation.size() do
        local target = validation.HR[j]
        local input = validation.LR[j]
        local approx = net:forward(input)
        validation_loss = validation_loss + criterion:forward(approx, target)
        if analysisInterval and i % analysisInterval == 0 then
			psnr = psnr + PSNR(target, approx)
			ssim = ssim + SSIM(target, approx) 
		end
    end
    validation_loss = validation_loss / validation.size()
    if analysisInterval and i % analysisInterval == 0 then
    	psnr = psnr / validation.size()
		ssim = ssim / validation.size()
	end
    
    -- Report training and validation loss
    loss_logger:add{train_loss, validation_loss}
    if saveInterval and i % saveInterval == 0 then
        loss_logger:plot()  
    end

    -- Print to console
		print('epoch = '..i..'\tloss = ' .. train_loss .. '\tgradient norm = ' .. current_abs_grad)
    if analysisInterval and i % analysisInterval == 0 then
		print('\tpsnr = ' .. psnr .. '   \tssim = ' .. ssim)
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