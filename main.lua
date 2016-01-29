--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- C. http://torch.ch/blog/2015/09/21/rmva.html

print("==> Loading required libraries")

require 'dp'
--require 'dpnn'
require 'rnn'
require 'torch'
require 'xlua'
require 'optim'

version = 1

---------------------------------------------------------------------------------
print("==> Processing options")

--[[command line arguments]] --
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')

cmd:option('--threads', 1, 'set number of threads')
cmd:option('--seed', 123)

--[[ data ]] --
cmd:option('--dataset', 'mnist', 'which data to use: mnist')
cmd:option('--digits', 1, 'how many digits has the number')

--[[ model ]] --
cmd:option('--model', 'va', 'which model to use: cnn | va (visual_attention)')

--[[ loss ]] --
cmd:option('--loss', 'reinforce', 'type of loss function to minimize: nll | mse | margin | reinforce | multi_nll')

--[[ train ]] --
cmd:option('--save', 'testing', 'selecet subfolder where to store loggers')
cmd:option('--batchSize', 20)
cmd:option('--learningRate', 1e-2, 'setup the learning rate')
cmd:option('--momentum', 9e-1, 'setup the momentum')
cmd:option('--weightDecay', 0, 'weight decay')
cmd:option('--plot', true)
cmd:option('--epochs', 100, 'define max number of epochs')
cmd:option('--preTrain', false, 'pretrain the glimpse sensor')
cmd:option('--preTrainEpochs', 50, 'pretrain the glimpse sensor')
cmd:option('--uniform', -0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')


cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)
cmd:log('logger.log', opt)

--torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

---------------------------------------------------------------------------------
print("==> Loading scripts")

dofile 'utils.lua'

if (opt.dataset == 'mnist') then

    dofile '1_data_mnist.lua'

end

-- create modules
if (opt.model == 'cnn') then
    dofile '2_model_cnn.lua'
elseif (opt.model == 'va') then
    dofile '2_model_VA.lua'
else
    print('Not implemented model selected!')
end

dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

---------------------------------------------------------------------------------
print("==> Training")

epoch = 0

if opt.uniform > 0 then
    for k, param in ipairs(model:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

if opt.preTrain then
    preTrain()
end

while epoch < opt.epochs do

    trainOptim()
    test()
end






