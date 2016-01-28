--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 18/01/16
-- Time: 14:41
-- To change this template use File | Settings | File Templates.
--


function dirLookup(dir)
    local listOfFiles = {}
    local p = io.popen('find "' .. dir .. '" -type f') --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.
    for file in p:lines() do --Loop through all files
    listOfFiles[#listOfFiles + 1] = file
    print(file)
    end
    return listOfFiles
end

function preTrain()

    -- create temp model for pretraining
    model = nn.Sequential()
    model:add(glimpse)
    model:add(classifier)

    parameters, gradParameters = model:getParameters()

    maxWShift = (DATA_WIDTH / 2 - glimpseSize / 2) / (WIDTH / 2)
    maxHShift = (DATA_HEIGHT / 2 - glimpseSize / 2) / (HEIGHT / 2)

    while epoch < opt.preTrainEpochs do
        preTrainOptim()
    end

    -- PRE-TRAINED
    opt.preTrain = false

    -- re-load data with random shift
    dofile '1_data_lol_shifted.lua'

    model = agent
    parameters, gradParameters = model:getParameters()

    -- add reinforce loss
    dofile '3_loss.lua'
    epoch = 0
end

