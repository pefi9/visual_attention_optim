--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

    -- This loss takes a vector of classes, and the index of
    -- the grountruth class as arguments. It is an SVM-like loss
    -- with a default margin of 1.

    criterion = nn.MultiMarginCriterion()

elseif (opt.loss == 'nll' or opt.preTrain) then

    -- The loss works like the MultiMarginCriterion: it takes
    -- a vector of classes, and the index of the grountruth class
    -- as arguments.

    criterion = nn.ClassNLLCriterion()

elseif opt.loss == 'mse' then

    -- for MSE, we add a tanh, to restrict the model's output
    model:add(nn.Tanh())

    -- The mean-square error is not recommended for classification
    -- tasks, as it typically tries to do too much, by exactly modeling
    -- the 1-of-N distribution. For the sake of showing more examples,
    -- we still provide it here:

    criterion = nn.MSECriterion()
    criterion.sizeAverage = false

elseif opt.loss == 'reinforce' then

    criterion = nn.ParallelCriterion(true)
    criterion:add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
    criterion:add(nn.VRClassReward(model, 1), nil, nn.Convert()) -- REINFORCE

elseif opt.loss == 'multi_nll' then

    criterion = nn.ParallelCriterion()
    for idx = 1, opt.digits do
        criterion:add(nn.ClassNLLCriterion())
    end
else

    error('unknown -loss')
end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)