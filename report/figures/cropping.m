img_name = 'img_001_SRF_4';

img1 = imread([img_name '_LR.png']);
img1 = im2double(img1);
img1 = imresize(img1, 4, 'bilinear'); 
[cropped1, rect] = imcrop(img1);

img2 = imread([img_name '_HR.png']);
img2 = im2double(img2);
[cropped2] = imcrop(img2, rect);

img3 = imread(['img_001.png']);
img3 = im2double(img3);

[cropped3] = imcrop(img3, rect);

%%
figure;
imshow(cropped1);
figure;
imshow(cropped2);
figure;
imshow(cropped3);

%%

img_name = 'urban_001';
imwrite(cropped1, [img_name '_bilinear_cropped.png']);
imwrite(cropped2, [img_name '_ground_truth_cropped.png']);
imwrite(cropped3, [img_name '_ours_cropped.png']);