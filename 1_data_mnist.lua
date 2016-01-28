--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

-- classes
classes = { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }
WIDTH = 28
HEIGHT = 28
DATA_N_CHANNEL = 1
ninputs = WIDTH * HEIGHT

---------------------------------------------------------------------------------
print("==> Loading data")

local dataset = (opt ~= nil and opt.dataset or 'mnist')

trainData, testData = {}, {}


if dataset == 'mnist' then

    -- train data
    local temp = torch.load('data/mnist/train.th7', 'ascii')
    trsize = 1000 --temp[1]:size()[1]
    trainData.data = torch.DoubleTensor(trsize, HEIGHT, WIDTH * opt.digits, 1)
    trainData.labels = {}
    for rec = 1, trsize do
        local tempData
        for digit = 1, opt.digits do
            if digit == 1 then
                tempData = temp[1][rec]
                trainData.labels[rec] = {}
                trainData.labels[rec][digit] = (temp[2][rec] == 0 and 10 or temp[2][rec])
            else
                local rand = math.floor(math.random() * trsize) + 1
                tempData = tempData:cat(temp[1][rand], 2)
                trainData.labels[rec][digit] = (temp[2][rand] == 0 and 10 or temp[2][rand])
            end
        end
        trainData.data[rec] = tempData
    end

    -- test data
    local temp = torch.load('data/mnist/test.th7', 'ascii')
    tesize = 100 --temp[1]:size()[1]
    testData.data = torch.DoubleTensor(tesize, HEIGHT, WIDTH * opt.digits, 1)
    testData.labels = {}
    for rec = 1, tesize do
        local tempData
        for digit = 1, opt.digits do
            if digit == 1 then
                tempData = temp[1][rec]
                testData.labels[rec] = {}
                testData.labels[rec][digit] = (temp[2][rec] == 0 and 10 or temp[2][rec])
            else
                local rand = math.floor(math.random() * tesize) + 1
                tempData = tempData:cat(temp[1][rand], 2)
                testData.labels[rec][digit] = (temp[2][rand] == 0 and 10 or temp[2][rand])
            end
        end
        testData.data[rec] = tempData
    end
end

---------------------------------------------------------------------------------
print("==> Preprocessing data")

trainData.data = trainData.data:transpose(2, 3):transpose(2, 4)
testData.data = testData.data:transpose(2, 3):transpose(2, 4)


print("==> Preprocessing normalization")

local mean = trainData.data:mean()
local std = trainData.data:std()

trainData.data = trainData.data:add(-mean):div(std)
testData.data = testData.data:add(-mean):div(std)


--print(trainData.data[trsize])
--print(testData.data[tesize])





