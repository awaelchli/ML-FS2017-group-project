require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


-- Load Network
net = torch.load(actionParam.folders.output .. actionParam.name .. ".model")

-- TODO: remove duplicate code (see metrics.lua)

-- Test example
criterion = nn.MSECriterion()
test_loss = 0
psnr = 0
ssim = 0
for i = 1, test.size() do
    local target = test.HR[i]
    local input = test.LR[i]
    local result = net:forward(input)
    test_loss = test_loss + criterion:forward(result, target)
    psnr = psnr + PSNR(target, result)
    ssim = ssim + SSIM(target, result)
    image.save(actionParam.folders.testResults .. "img_" .. string.format("%03d", i) .. ".png", result)
end
test_loss = test_loss / test.size()
psnr = psnr / test.size()
ssim = ssim / test.size()

-- Print results
print('Loss on Testset: ', test_loss)
print('SSIM on Testset: ', ssim)
print('PSNR on Testset: ', psnr)

io.open(actionParam.folders.testResults .. 'metrics.txt', 'w')
    :write('Statistics on testset:', '\n')
    :write('Average PSNR: ' .. psnr .. 'dB', '\n')
    :write('Average SSIM: ' .. ssim, '\n')
    :write('Loss: ' .. test_loss, '\n')
    :close()