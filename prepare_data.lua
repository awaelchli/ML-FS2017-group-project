function prepare_data(images) 

    local inputChannels = 3

    -- Convert greyscale images to RGB
    for i = 1, #images do
        images[i] = images[i]:expand(inputChannels, images[i]:size(2), images[i]:size(3))
    end

    local n = #images / 2

    local data = {}
    data.HR = {}
    data.LR = {}
    data.size = function() return #data.LR end
    data.channels = function() return inputChannels end
    
    for i = 1, n do
        data.LR[i] = images[2 * i]
        data.HR[i] = images[2 * i - 1]
    end

    return data

end