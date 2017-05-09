function PSNR(true_frame, pred)

   local eps = 0.0001

   local prediction_error = 0
   for i = 1, pred:size(2) do
          for j = 1, pred:size(3) do
            for c = 1, pred:size(1) do
            -- put image from -1 to 1 to 0 and 255
            prediction_error = prediction_error +
              (pred[c][i][j] - true_frame[c][i][j])^2
            end
          end
   end
   
   --MSE
   prediction_error=128*128*prediction_error/(pred:size(1)*pred:size(2)*pred:size(3))

   --PSNR
   if prediction_error>eps then
      prediction_error = 10*torch.log((255*255)/ prediction_error)/torch.log(10)
   else
      prediction_error = 10*torch.log((255*255)/ eps)/torch.log(10)
   end
   
   return prediction_error
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
   img1:add(1):div(2):mul(255)
   img2:add(1):div(2):mul(255)

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