require 'nn'
require 'nngraph'
require 'load_images'
require 'torch'
require 'optim'
require 'image'

--nngraph.setDebug(true)
logger = optim.Logger('loss_log.txt')

images = load_images.load('datasets/Set14/image_SRF_4/', 'png')
inputChannels = 3

-- Convert greyscale images
for i = 1, #images do
        images[i] = images[i]:expand(inputChannels, images[i]:size(2), images[i]:size(3))
end

n = #images / 2

imagesLR = {}--images:select()
imagesHR = {}--images[2]

print(images)

for i = 1, n do
	imagesLR[i] = images[2 * i]
	imagesHR[i] = images[2 * i - 1]
end


input = - nn.Identity()
output1 = input - nn.SpatialConvolution(3, 3, 5, 5, 1, 1, 2, 2)
output2 = {input, output1} - nn.CAddTable()

net = nn.gModule({input}, {output2})

graph.dot(net.fg, 'res_net', 'res_net')


out = net:forward(imagesLR[1])
print(out:size())

image.save('res_test.png', out)
