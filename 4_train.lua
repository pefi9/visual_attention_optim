--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--


----------------------------------------------------------------------
print '==> defining some tools'

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
    parameters, gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = 1e-7,
    nesterov = true,
    dampening = 0
}
optimMethod = optim.sgd

lr = opt.learningRate
wd = opt.weightDecay
momentum = opt.momentum
lrDecay = 1e-3

----------------------------------------------------------------------
print '==> defining training procedure'

-- custom training method with nesterov momentum
function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(trsize)

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local output
    for t = 1, trsize, opt.batchSize do

        -- disp progress
        xlua.progress(t, trsize)

        -- create mini batch
        local inputs = {}
        local targets = {}

        for i = t, math.min(t + opt.batchSize - 1, trsize) do
            -- load new sample
            local input = trainData.data[shuffle[i]]
            local target = trainData.labels[shuffle[i]]

            table.insert(inputs, input)
            table.insert(targets, target)
        end


        -- Nesterov momentum
        -- (1) evaluate f(x) and df/dx
        -- first step in the direction of the momentum vector
        if not prev_parameters then
            prev_parameters = parameters:clone()
        else
            prev_parameters:resizeAs(parameters):copy(parameters)
        end

        if prev_dfdx then
            parameters:add(momentum, prev_dfdx)
        end

        -- reset gradients
        gradParameters:zero()

        -- f is the average of all criterions
        local f = 0

        -- evaluate function for complete mini batch
        for i = 1, #inputs do
            -- estimate f
            output = model:forward(inputs[i])
            local err = criterion:forward(output, torch.Tensor(targets[i]))
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, torch.Tensor(targets[i]))
            model:backward(inputs[i], df_do)

            -- update confusion
            if (opt.model == 'va') then
                confusion:add(output[1][1], targets[i][1])
            else
                for d = 1, opt.digits do
                    confusion:add(output[d], targets[i][d])
                end
            end
        end

        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f / #inputs

        -- weight decay
        if (wd ~= 0) then
            gradParameters:add(wd, parameters)
        end

        -- (4) apply momentum
        if not prev_dfdx then
            prev_dfdx = torch.Tensor():typeAs(gradParameters):resizeAs(gradParameters):fill(0)
        else
            prev_dfdx:mul(momentum)
        end

        prev_dfdx:add(-lr, gradParameters)
        prev_parameters:add(prev_dfdx)
        parameters:copy(prev_parameters)
    end

    -- time taken
    time = sys.clock() - time
    time = time / trsize
    print("\n==> time to learn 1 sample = " .. (time * 1000) .. 'ms')
    print("Learning rate: " .. lr)

    -- print confusion matrix
    print(confusion)

    -- update logger/plot
    trainLogger:add { ['% mean class accuracy (train set)'] = confusion.totalValid * 100 }
    if opt.plot then
        trainLogger:style { ['% mean class accuracy (train set)'] = '-' }
        trainLogger:plot()
    end

    -- save/log current net
    --    local filename = paths.concat(opt.save, 'model.net')
    --    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    print('==> saving model to ' .. filename)
    --    torch.save(filename, model)

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
    lr = (lrDecay == 0 and lr or lr / (1 + lrDecay))
end



function trainOptim()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(trsize)

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local output
    for t = 1, trsize, opt.batchSize do
        -- disp progress
        xlua.progress(t, trsize)

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t, math.min(t + opt.batchSize - 1, trsize) do
            -- load new sample
            local input = trainData.data[shuffle[i]]
            local target = trainData.labels[shuffle[i]]

            table.insert(inputs, input)
            table.insert(targets, target)
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1, #inputs do
                -- estimate f
                output = model:forward(inputs[i])
                local err = criterion:forward(output, torch.Tensor(targets[i]))
                f = f + err

                -- estimate df/dW
                local df_do = criterion:backward(output, torch.Tensor(targets[i]))
                model:backward(inputs[i], df_do)

                -- update confusion
                if (opt.model == 'va') then
                    confusion:add(output[1][1], targets[i][1])
                else
                    for d = 1, opt.digits do
                        confusion:add(output[d], targets[i][d])
                    end
                end
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f / #inputs

            -- return f and df/dX
            return f, gradParameters
        end

        -- optimize on current mini-batch

        optimMethod(feval, parameters, optimState)
    end


    -- time taken
    time = sys.clock() - time
    time = time / trsize
    print("\n==> time to learn 1 sample = " .. (time * 1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update logger/plot
    trainLogger:add { ['% mean class accuracy (train set)'] = confusion.totalValid * 100 }
    if opt.plot then
        trainLogger:style { ['% mean class accuracy (train set)'] = '-' }
        trainLogger:plot()
    end


    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end


function preTrainOptim()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(trsize)

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local output
    for t = 1, trsize, opt.batchSize do
        -- disp progress
        xlua.progress(t, trsize)

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t, math.min(t + opt.batchSize - 1, trsize) do
            -- load new sample
            local loc_x = (math.random() * 2 - 1) * maxWShift
            local loc_y = (math.random() * 2 - 1) * maxHShift

            local input = { trainData.data[shuffle[i]], torch.Tensor { loc_y, loc_x } }
            local target = trainData.labels[shuffle[i]]

            table.insert(inputs, input)
            table.insert(targets, target)
        end

        -- reset gradients
        gradParameters:zero()

        -- f is the average of all criterions
        local f = 0

        -- evaluate function for complete mini batch
        for i = 1, #inputs do
            -- estimate f
            output = model:forward(inputs[i])
            local err = criterion:forward(output, torch.Tensor { targets[i] })
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, torch.Tensor { targets[i] })
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])
        end

        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        model:updateParameters(lr)
    end

    -- time taken
    time = sys.clock() - time
    time = time / trsize
    print("\n==> time to learn 1 sample = " .. (time * 1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update logger/plot
    trainLogger:add { ['% mean class accuracy (train set)'] = confusion.totalValid * 100 }
    if opt.plot then
        trainLogger:style { ['% mean class accuracy (train set)'] = '-' }
        trainLogger:plot()
    end


    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end