require 'parameters.TEMPLATE'

actionParam.networkFile = 'build_network_padding'

-----------------------------------
-- Parameters to be tuned freely --
-----------------------------------
actionParam.upscaleFactor = 4
actionParam.numRecursions = 20
actionParam.numHiddenChannelsInRecursion = 12

actionParam.epochs = 500
actionParam.sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-8,
   weightDecay = 0,
   momentum = 0--0.9
}

actionParam.mainData = {'datasets/BSD100_SR/image_SRF_4/','datasets/Set14/image_SRF_4/'}
actionParam.testData = {'datasets/Set5/image_SRF_4/'}

actionParam.folders.testResults = actionParam.folders.output .. 'results/full_network_padding_deep/'
actionParam.folders.logs = actionParam.folders.testResults .. 'logs/'