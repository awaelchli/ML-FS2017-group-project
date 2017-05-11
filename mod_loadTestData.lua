require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


test_separate = actionParam.testeSeparate


-- Load dataset
if test_separate then
	test_images = load_images.load(actionParam.testData[1], 'png', false)
    test = prepare_data(test_images)
else
	images = load_images.load(actionParam.mainData[1], 'png', false)
	data = prepare_data(images)
    train, validation, test = split_data(data, 0.8, 0.1)
end