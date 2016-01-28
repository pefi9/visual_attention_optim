--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

----------------------------------------------------------------------
print '==> define parameters of the model'

-- 10-class problem
noutputs = #classes

-- input dimensions
nfeats = 1
width = WIDTH * opt.digits
height = HEIGHT
ninputs = nfeats * width * height

-- hidden units, filter sizes (for ConvNet only):
nstates = { 8, 16, 128 * opt.digits }
filtsize = { 5, 5 }
poolsize = { 2, 2 }
remainHeight = 5
remainWidth = { 5, 12, 19}
normkernel = image.gaussian1D(7)


----------------------------------------------------------------------
print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialZeroPadding(2, 2, 2, 2))
model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.SpatialLPPooling(nstates[1], 2, poolsize[1], poolsize[1]))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.SpatialLPPooling(nstates[2], 2, poolsize[2], poolsize[2]))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[2] * remainHeight * remainWidth[opt.digits]))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2] * remainHeight * remainWidth[opt.digits], nstates[3]))
model:add(nn.ReLU())

local prl = nn.ConcatTable()
for idx=1,opt.digits do
    prl:add(nn.Sequential():add(nn.Linear(nstates[3], noutputs)):add(nn.LogSoftMax()))
end

model:add(prl)





