actionParam = {}

-- Environment
actionParam.folders = {}
actionParam.folders.logs = 'logs/'
actionParam.folders.output = 'out/'
actionParam.folders.testResults = actionParam.folders.output .. 'results/'

-- Network params
actionParam.networkFile = 'build_network'
actionParam.inputChannels = 3
actionParam.upscaleFactor = 4
actionParam.numRecursions = 5
actionParam.numHiddenChannelsInRecursion = 6 -- 32  -- approximately +700MB RAM usage per additional channel

-- Training params
actionParam.epochs = 100
actionParam.sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}
actionParam.saveInterval = 10			-- set to false to disable
actionParam.analysisInterval = 5		-- set to false to disable

-- Data set params
actionParam.trainingData = {'datasets/BSD100_SR/image_SRF_4/'}
actionParam.validationData = {'datasets/Set14/image_SRF_4/'}
actionParam.testData = {'datasets/Set5/image_SRF_4/'}

-- Code modules
actionParam.create = 'createNetwork'
actionParam.loadTrainData = 'loadTrainData'
actionParam.train = 'trainNetwork'
actionParam.loadTestData = 'loadTestData'
actionParam.test = 'testNetwork'