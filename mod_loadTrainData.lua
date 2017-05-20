require 'load_images'
require 'prepare_data'
require 'split_data'
require 'metrics'


-- Load dataset
train_images = load_images.multi_load(actionParam.trainingData, 'png')
validation_images = load_images.multi_load(actionParam.validationData, 'png')

-- Preprocess dataset
train = prepare_data(train_images)
validation = prepare_data(validation_images)