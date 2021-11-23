import paddle
from paddle import grad, reshape
from paddle.autograd.utils import (
    _tensors, 
    _stack_tensor_or_return_none, 
    _replace_none_with_zero_tensor
)

def batch_jacobian(func, inputs, create_graph=False, allow_unused=False):
    inputs = _tensors(inputs, "inputs")
    outputs = _tensors(func(*inputs), "outputs")
    batch_size = inputs[0].shape[0]
    for input in inputs:
        assert input.shape[0] == batch_size
    for output in outputs:
        assert output.shape[0] == batch_size
    fin_size = len(inputs)
    fout_size = len(outputs)
    flat_outputs = tuple(
        reshape(
            output, shape=[batch_size, -1]) for output in outputs)
    jacobian = tuple()
    for i, flat_output in enumerate(flat_outputs):
        jac_i = list([] for _ in range(fin_size))
        for k in range(flat_output.shape[1]):
            row_k = grad(
                flat_output[:, k],
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)
            for j in range(fin_size):
                jac_i[j].append(
                    reshape(
                        row_k[j], shape=[-1])
                    if isinstance(row_k[j], paddle.Tensor) else None)
        jacobian += (tuple(
            _stack_tensor_or_return_none(jac_i_j) for jac_i_j in jac_i), )
    if fin_size == 1 and fout_size == 1:
        return jacobian[0][0]
    elif fin_size == 1 and fout_size != 1:
        return tuple(jacobian[i][0] for i in range(fout_size))
    elif fin_size != 1 and fout_size == 1:
        return jacobian[0]
    else:
        return jacobian


def batch_hessian(func, inputs, create_graph=False, allow_unused=False):
    inputs = _tensors(inputs, "inputs")
    outputs = func(*inputs)
    batch_size = inputs[0].shape[0]
    for input in inputs:
        assert input.shape[0] == batch_size
    assert isinstance(outputs, paddle.Tensor) and outputs.shape == [
        batch_size, 1
    ], "The function to compute batched Hessian matrix should return a Tensor of shape [batch_size, 1]"

    def jac_func(*ins):
        grad_inputs = grad(
            outputs,
            ins,
            create_graph=True,
            retain_graph=True,
            allow_unused=allow_unused)
        return tuple(
            _replace_none_with_zero_tensor(grad_inputs[i], inputs[i])
            for i in range(len(inputs)))

    return batch_jacobian(
        jac_func, inputs, create_graph=create_graph, allow_unused=allow_unused)

