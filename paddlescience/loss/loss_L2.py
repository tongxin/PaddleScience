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

import paddle
import paddle.nn.functional as F
from paddle.autograd import jacobian, hessian, batch_jacobian, batch_hessian
from ..pde import first_order_rslts, first_order_derivatives, second_order_derivatives
from .loss_base import LossBase


class L2(LossBase):
    def __init__(self,
                 pdes,
                 geo,
                 aux_func=None,
                 bc_weight=None,
                 run_in_batch=False):
        super(L2, self).__init__(pdes, geo)

        self.pdes = pdes
        self.geo = geo
        self.aux_func = aux_func
        self.bc_weight = bc_weight
        self.d_records = dict()
        self.run_in_batch = run_in_batch

    def set_batch_size(self, batch_size):
        self.pdes.set_batch_size(batch_size)
        self.geo.set_batch_size(batch_size)
        self.batch_size = batch_size
        self.num_batch = self.geo.get_num_batch()

    def cal_first_order_rslts(self, net, ins):
        outs = net.nn_func(ins)
        for i in range(net.num_outs):
            self.d_records[first_order_rslts[i]] = outs[i]

    def cal_first_order_derivatives(self, net, ins):
        d_values = jacobian(net.nn_func, ins, create_graph=True)
        for i in range(net.num_outs):
            for j in range(net.num_ins):
                if self.pdes.time_dependent:
                    self.d_records[first_order_derivatives[i][j]] = d_values[
                        i][j]
                else:
                    self.d_records[first_order_derivatives[i][
                        j + 1]] = d_values[i][j]

    def cal_second_order_derivatives(self, net, ins):
        for i in range(net.num_outs):

            def func(ins):
                return net.nn_func(ins)[i]

            d_values = hessian(func, ins, create_graph=True)
            for j in range(net.num_ins):
                for k in range(net.num_ins):
                    if self.pdes.time_dependent:
                        self.d_records[second_order_derivatives[i][j][
                            k]] = d_values[j][k]
                    else:
                        self.d_records[second_order_derivatives[i][j + 1][
                            k + 1]] = d_values[j][k]

    def batch_cal_first_order_rslts(self, net, ins):
        outs = net.nn_func(ins)
        for i in range(net.num_outs):
            self.d_records[first_order_rslts[i]] = outs[:, i]

    def batch_cal_first_order_derivatives(self, net, ins):
        d_values = batch_jacobian(net.nn_func, ins, create_graph=True)
        d_values = paddle.reshape(
            d_values, shape=[net.num_outs, self.batch_size, net.num_ins])
        for i in range(net.num_outs):
            for j in range(net.num_ins):
                if self.pdes.time_dependent:
                    self.d_records[first_order_derivatives[i][j]] = d_values[
                        i, :, j]
                else:
                    self.d_records[first_order_derivatives[i][
                        j + 1]] = d_values[i, :, j]

    def batch_cal_second_order_derivatives(self, net, ins):
        for i in range(net.num_outs):

            def func(ins):
                return net.nn_func(ins)[:, i:i + 1]

            d_values = batch_hessian(func, ins, create_graph=True)
            d_values = paddle.reshape(
                d_values, shape=[net.num_ins, self.batch_size, net.num_ins])
            for j in range(net.num_ins):
                for k in range(net.num_ins):
                    if self.pdes.time_dependent:
                        self.d_records[second_order_derivatives[i][j][
                            k]] = d_values[j, :, k]
                    else:
                        self.d_records[second_order_derivatives[i][j + 1][
                            k + 1]] = d_values[j, :, k]

    def batch_eq_loss(self, net, ins):
        self.batch_cal_first_order_rslts(net, ins)
        self.batch_cal_first_order_derivatives(net, ins)
        if self.pdes.need_2nd_derivatives:
            self.batch_cal_second_order_derivatives(net, ins)
        # TODO(lml): add support for aux_func
        eq_loss_l = [0.0 for _ in range(self.pdes.num_pdes)]
        if self.aux_func is not None:
            eq_loss_l = self.aux_func(ins)
        for idx in range(self.pdes.num_pdes):
            for item in self.pdes.get_pde(idx):
                tmp = item.coefficient
                for de in item.derivative:
                    tmp = tmp * self.d_records[de]
                eq_loss_l[idx] += tmp
        self.d_records.clear()
        eq_loss = paddle.reshape(paddle.stack(eq_loss_l, axis=0), shape=[-1])
        return paddle.norm(eq_loss, p=2)

    def eq_loss(self, net, ins):
        self.cal_first_order_rslts(net, ins)
        self.cal_first_order_derivatives(net, ins)
        if self.pdes.need_2nd_derivatives:
            self.cal_second_order_derivatives(net, ins)
        eq_loss_l = [0.0 for _ in range(self.pdes.num_pdes)]
        if self.aux_func is not None:
            eq_loss_l = self.aux_func(ins)
        for idx in range(self.pdes.num_pdes):
            for item in self.pdes.get_pde(idx):
                tmp = item.coefficient
                for de in item.derivative:
                    tmp = tmp * self.d_records[de]
                eq_loss_l[idx] += tmp
        self.d_records.clear()
        return eq_loss_l

    def bc_loss(self, u, batch_id):
        # print("u.shape: ", u.shape)
        bc_u = paddle.index_select(u, self.geo.bc_index[batch_id])
        # TODO(lml): support index select with batch id
        bc_value = self.pdes.bc_value
        if self.pdes.bc_check_dim is not None:
            bc_u = paddle.index_select(bc_u, self.pdes.bc_check_dim, axis=1)
        # print("bc_u.shape: ", bc_u.shape)
        # print("bc_value.shape: ", bc_value.shape)
        bc_diff = bc_u - bc_value
        if self.bc_weight is None:
            return paddle.norm(bc_diff, p=2)
        else:
            bc_weight = paddle.to_tensor(self.bc_weight, dtype="float32")
            return paddle.sum(bc_diff * bc_diff * bc_weight)

    def ic_loss(self, u, batch_id):
        if self.pdes.time_dependent:
            ic_u = paddle.index_select(u, self.geo.ic_index[batch_id])
            # TODO(lml): support index select with batch id
            ic_value = self.pdes.ic_value
            ic_diff = ic_u - ic_value
            return paddle.norm(ic_diff, p=2)
        else:
            return 0

    def batch_run(self, net, batch_id):
        b_datas = self.geo.get_step()
        u = net.nn_func(b_datas)
        eq_loss = 0
        if self.run_in_batch:
            eq_loss = self.batch_eq_loss(net, b_datas)
        else:
            eq_loss_l = []
            for data in b_datas:
                eq_loss_l += self.eq_loss(net, data)
            eq_loss = paddle.stack(eq_loss_l, axis=0)
            eq_loss = paddle.norm(eq_loss, p=2)
        bc_loss = self.bc_loss(u, batch_id)
        ic_loss = self.ic_loss(u, batch_id)
        return eq_loss, bc_loss, ic_loss
