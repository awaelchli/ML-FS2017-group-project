require 'torch'
require 'image'
require 'metrics'
require 'paths'
require 'load_images'
require 'prepare_data'

--------------------------------------------------------------------------------------------
-- This script loads low resolution images, upscales them via naive interpolation methods --
-- and compares them with the ground truth by means of PSNR and SSIM metrics.             --
--------------------------------------------------------------------------------------------

local set5_4 = 'datasets/Set5/image_SRF_4/'
local save_interpolated_images = true
local output = 'out/naive_upscaling/'

local test_images = load_images.load(set5_4, 'png', false)
local dataset = prepare_data(test_images)

-- Upscale 
local nearest = {}
local bilinear = {}
local bicubic = {}

for i = 1, dataset.size() do
    local low_res = dataset.LR[i]
    local ground_truth = dataset.HR[i]
    
    nearest[i] = image.scale(low_res, ground_truth:size(3), ground_truth:size(2), 'simple')
    bilinear[i] = image.scale(low_res, ground_truth:size(3), ground_truth:size(2), 'bilinear')
    bicubic[i] = image.scale(low_res, ground_truth:size(3), ground_truth:size(2), 'bicubic')
end

-- Compute average PSNR and SSIM
local avgPSNR_nearest = averagePSNR(nearest, dataset.HR)
local avgSSIM_nearest = averageSSIM(nearest, dataset.HR)

local avgPSNR_bilinear = averagePSNR(bilinear, dataset.HR)
local avgSSIM_bilinear = averageSSIM(bilinear, dataset.HR)

local avgPSNR_bicubic = averagePSNR(bicubic, dataset.HR)
local avgSSIM_bicubic = averageSSIM(bicubic, dataset.HR)


-- Print results
io.open(output .. 'avgerage-PSNR.txt', 'w')
    :write('Average PSNR:', '\n')
    :write('nearest:\t' .. avgPSNR_nearest, '\n')
    :write('bilinear:\t' .. avgPSNR_bilinear, '\n')
    :write('bicubic:\t' .. avgPSNR_bicubic, '\n')
    :close()

io.open(output .. 'avgerage-SSIM.txt', 'w')
    :write('Average SSIM:', '\n')
    :write('nearest:\t' .. avgSSIM_nearest, '\n')
    :write('bilinear:\t' .. avgSSIM_bilinear, '\n')
    :write('bicubic:\t' .. avgSSIM_bicubic, '\n')
    :close()


-- Save images
if(save_interpolated_images) then
    paths.mkdir(output)
    
    for i = 1, #nearest do
        image.save(output .. string.format('%03d', i) .. '_nearest.png', nearest[i])
        image.save(output .. string.format('%03d', i) .. '_bilinear.png', bilinear[i])
        image.save(output .. string.format('%03d', i) .. '_bicubic.png', bicubic[i])
    end

end