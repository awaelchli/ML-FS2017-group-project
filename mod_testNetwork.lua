require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


-- Load Network
net = torch.load("out/"..actionParam.name..".model")


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
    image.save("out/results/img_"..string.format("%03d", i)..".png", result)
end
test_loss = test_loss / test.size()
psnr = psnr / test.size()
ssim = ssim / test.size()

print('Loss on Testset: ', test_loss)
print('SSIM on Testset: ', ssim)
print('PSNR on Testset: ', psnr)