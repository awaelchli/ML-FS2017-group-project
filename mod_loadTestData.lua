require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


-- Load dataset
test_images = load_images.multi_load(actionParam.testData, 'png')
test = prepare_data(test_images)