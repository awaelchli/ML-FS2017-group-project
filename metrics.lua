require 'image'

function averagePSNR(set1, set2)

    -- TODO: test sizes

    local avgPSNR = 0
    for i = 1, #set1 do
        avgPSNR = avgPSNR + PSNR(set1[i], set2[i])
    end

    return avgPSNR / #set1
end

function averageSSIM(set1, set2)

    -- TODO: test sizes

    local avgSSIM = 0
    for i = 1, #set1 do
        avgSSIM = avgSSIM + SSIM(set1[i], set2[i])
    end

    return avgSSIM / #set1
end


function PSNR(img1, img2)

   local eps = 0
   local max = 1
   
   -- MSE (mean over element-wise squared differences)
   mse = torch.csub(img1, img2)
   mse:cmul(mse)
   mse = mse:sum() / mse:nElement()

   -- PSNR
   psnr = 10 * torch.log(max * max / mse) / torch.log(10)
   
   return psnr
end

--------------------------------------------------------------------------------
function SSIM(img1, img2)

   if img1:size(1) > 2 then
    img1 = image.rgb2y(img1)
    img1 = img1[1]
    img2 = image.rgb2y(img2)
    img2 = img2[1]
   end

   -- place images between 0 and 255.
   img1:mul(255)
   img2:mul(255)

   local K1 = 0.01;
   local K2 = 0.03;
   local L = 255;

   local C1 = (K1*L)^2;
   local C2 = (K2*L)^2;
   local window = image.gaussian(11, 1.5/11,0.0708);

   local window = window:div(torch.sum(window));

   local mu1 = image.convolve(img1, window, 'full')
   local mu2 = image.convolve(img2, window, 'full')

   local mu1_sq = torch.cmul(mu1,mu1);
   local mu2_sq = torch.cmul(mu2,mu2);
   local mu1_mu2 = torch.cmul(mu1,mu2);

   local sigma1_sq = image.convolve(torch.cmul(img1,img1),window,'full') - mu1_sq
   local sigma2_sq = image.convolve(torch.cmul(img2,img2),window,'full') - mu2_sq
   local sigma12 =  image.convolve(torch.cmul(img1,img2),window,'full') - mu1_mu2

   local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)) ,
     torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
   local mssim = torch.mean(ssim_map);
   
   return mssim
end