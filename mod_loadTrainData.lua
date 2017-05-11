require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


test_separate = actionParam.testeSeparate


-- Load dataset
images = load_images.load(actionParam.mainData[1], 'png', false)

if test_separate then
    test_images = load_images.load(actionParam.testData[1], 'png', false)
end

-- Preprocess dataset
data = prepare_data(images)
if test_separate then
    train, validation = split_data(data, 0.8, 0.2)
    test = prepare_data(test_images)
else
    train, validation, test = split_data(data, 0.8, 0.1)
end