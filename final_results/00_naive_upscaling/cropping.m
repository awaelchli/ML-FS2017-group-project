img_name = '001';

img1 = imread([img_name '_nearest.png']);
img1 = im2double(img1);
[cropped1, rect] = imcrop(img1);


img2 = imread([img_name '_bilinear.png']);
img2 = im2double(img2);
[cropped2] = imcrop(img2, rect);

img3 = imread([img_name '_bicubic.png']);
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
imwrite(cropped1, [img_name '_nearest_cropped.png']);
imwrite(cropped2, [img_name '_bilinear_cropped.png']);
imwrite(cropped3, [img_name '_bicubic_cropped.png']);