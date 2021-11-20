# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlescience as psci
import numpy as np
import paddle

# Random Seed
paddle.seed(1234)


# Generate BC value
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 5) < 1e-4:
            bc_value[i][0] = 0.1
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value


# Generate BC weight
def GenBCWeight(xy, bc_index):
    bc_weight = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 5) < 1e-4:
            bc_weight[i][0] = 1.0 - 0.2 * abs(xy[id][0])
            bc_weight[i][1] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
    return bc_weight


# Geometry
geo = psci.geometry.Rectangular(space_origin=(-5, -5), space_extent=(5, 5))

# PDE Laplace
pdes = psci.pde.NavierStokes2D(nu=0.1, rho=1.0)

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_steps=(101, 101))

# bc value
bc_value = GenBC(geo.steps, geo.bc_index)
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation='tanh')
# load params from checkpoint
# net.set_state_dict(paddle.load('./checkpoint/net_params_30000'))

# Loss, TO rename
bc_weight = GenBCWeight(geo.steps, geo.bc_index)
# loss = psci.loss.L2(pdes=pdes, geo=geo, bc_weight=bc_weight, run_in_batch=True)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    bc_weight=bc_weight,
                    run_in_batch=False)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
# load params from checkpoint
# opt.set_state_dict(paddle.load('./checkpoint/opt_params_30000'))

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000, batch_size=None, checkpoint_freq=1000)

# Use solution
rslt = solution(geo).numpy()
u = rslt[:, 0]
v = rslt[:, 1]
u_and_v = np.sqrt(u * u + v * v)
psci.visu.Rectangular2D(geo, u, filename="rslt_u")
psci.visu.Rectangular2D(geo, v, filename="rslt_v")
psci.visu.Rectangular2D(geo, u_and_v, filename="u_and_v")
np.save('./rslt_ldc_2d.npy', rslt)
