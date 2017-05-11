require(actionParam.networkFile)

print("Building network")
net = build_network(actionParam.inputChannels, actionParam.upscaleFactor, actionParam.numRecursions)

print("Cleaning up")
saveNet = net:clone('weight','bias','gradWeight','gradBias')
saveNet:clearState() --if it wasn't clean, clean it
netUnion = nn.Container()
netUnion:add(net)
netUnion:add(saveNet)
x, dl_dx = netUnion:getParameters()

print("Saving model")
torch.save("out/"..actionParam.name..".model", saveNet)

print("Done")