function split_data(data, trainRatio, validationRatio)

    local numTrainSamples = torch.floor(data.size() * trainRatio)
    local numValidationSamples = torch.floor(data.size() * validationRatio)
    local numTestSamples = data.size() - numTrainSamples - numValidationSamples

    local train = {}
    local validation = {}
    local test = {}
    
    local j = 1

    local fill_split = function(data, split, range) 
        split.HR = {} 
        split.LR = {}
        split.size = function() return range end

        for i = 1, range do
            split.HR[i] = data.HR[j]
            split.LR[i] = data.LR[j]
            j = j + 1
        end
    end

    fill_split(data, train, numTrainSamples)
    fill_split(data, validation, numValidationSamples)
    fill_split(data, test, numTestSamples)

    return train, validation, test

end