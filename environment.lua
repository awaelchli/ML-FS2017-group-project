require 'paths'

environment = {}

function environment.setup(parameters) 

    paths.mkdir(parameters.folders.logs)
    paths.mkdir(parameters.folders.output)
    paths.mkdir(parameters.folders.testResults)

end

return environment