require 'parameters.TEMPLATE'

actionParam.networkFile = 'build_network_padding_2'

-----------------------------------
-- Parameters to be tuned freely --
-----------------------------------
actionParam.upscaleFactor = 4
actionParam.numRecursions = 5
actionParam.numHiddenChannelsInRecursion = 6

actionParam.epochs = 500
actionParam.sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-8,
   weightDecay = 0,
   momentum = 0--0.9
}

actionParam.trainingData = {'datasets/BSD100_SR/image_SRF_4/'}
actionParam.validationData = {'datasets/Set14/image_SRF_4/'}
actionParam.testData = {'datasets/Set5/image_SRF_4/'}

actionParam.folders.testResults = actionParam.folders.output .. 'results/full_network_deep/'
actionParam.folders.logs = actionParam.folders.testResults .. 'logs/'