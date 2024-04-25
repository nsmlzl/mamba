import torch
from einops import rearrange
import pytest

from causal_conv1d import causal_conv1d_fn #, causal_conv1d_update


@pytest.mark.parametrize("batch", [64])
@pytest.mark.parametrize("dim", [768])
@pytest.mark.parametrize("seqlen", [1024])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("channel_last", [True])
def test_channel_last_layout(batch, dim, seqlen, width, itype, channel_last):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    rtolw, atolw = (1e-3, 1e-3)
    torch.random.manual_seed(0)

    # reproduce following tensor:
    # x.shape torch.Size([64, 768, 1024]), x.stride() (1024, 65536, 1)
    x = rearrange(torch.randn(dim, batch, seqlen, device=device, dtype=itype), "d b s -> b d s").requires_grad_()
    x_ref = x.detach().clone().requires_grad_()

    # convert to channel-last memory layout
    x_cl = rearrange(rearrange(x, "b d s -> b s d").contiguous(), "b s d -> b d s")

    weight = torch.randn(dim, width, device=device, dtype=torch.float32).requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias = torch.randn(dim, device=device, dtype=torch.float32).requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()

    initial_states = None
    return_final_states = False
    activation = "silu"

    # generate ouput for normal and channel-last memory layout
    out = causal_conv1d_fn(x_cl, weight, bias, initial_states=initial_states, return_final_states=return_final_states, activation=activation)
    out_ref = causal_conv1d_fn(x_ref, weight_ref, bias_ref, initial_states=initial_states, return_final_states=return_final_states, activation=activation)

    # convert back
    out = rearrange(rearrange(out, "b d s -> d b s").contiguous(), "d b s -> b d s")
    print(out[-1,-1,:])
    print(out_ref[-1,-1,:])

    assert out.shape == out_ref.shape
    assert out.stride() == out_ref.stride()
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    # check backward path
    g = torch.randn_like(out)
    g_ref = g.detach().clone()

    out.backward(g)
    out_ref.backward(g_ref)

    print(x_ref.grad)
    print(x.grad)
    assert torch.allclose(x.grad, x_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch", [64])
@pytest.mark.parametrize("dim", [768])
@pytest.mark.parametrize("seqlen", [1024])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("channel_last", [True])
def test_state_transfer(batch, dim, seqlen, width, itype, channel_last):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    rtolw, atolw = (1e-3, 1e-3)
    torch.random.manual_seed(0)

    # reproduce following tensor:
    # x.shape torch.Size([64, 768, 1024]), x.stride() (1024, 65536, 1)
    x_ref = rearrange(torch.randn(dim, batch, seqlen, device=device, dtype=itype), "d b s -> b d s").requires_grad_()
    x0 = x_ref[:,:,:seqlen//2].detach().clone().requires_grad_()
    x1 = x_ref[:,:,seqlen//2:].detach().clone().requires_grad_()

    # convert to channel-last memory layout
    x0_cl = rearrange(rearrange(x0, "b d s -> b s d").contiguous(), "b s d -> b d s")
    x1_cl = rearrange(rearrange(x1, "b d s -> b s d").contiguous(), "b s d -> b d s")

    weight_ref = torch.randn(dim, width, device=device, dtype=torch.float32).requires_grad_()
    weight0 = weight_ref.detach().clone().requires_grad_()
    weight1 = weight_ref.detach().clone().requires_grad_()
    bias_ref = torch.randn(dim, device=device, dtype=torch.float32).requires_grad_()
    bias0 = bias_ref.detach().clone().requires_grad_()
    bias1 = bias_ref.detach().clone().requires_grad_()

    activation = "silu"

    # generate reference output and output with init transfer
    out_ref = causal_conv1d_fn(x_ref, weight_ref, bias_ref, initial_states=None, return_final_states=False, activation=activation)
    (out0, final_states0) = causal_conv1d_fn(x0_cl, weight0, bias0, initial_states=None, return_final_states=True, activation=activation)
    out1 = causal_conv1d_fn(x1_cl, weight1, bias1, initial_states=final_states0, return_final_states=False, activation=activation)
    out = torch.cat((out0, out1), dim=2)

    # convert back
    out = rearrange(rearrange(out, "b d s -> d b s").contiguous(), "d b s -> b d s")
    print(out[-1,-1,:])
    print(out_ref[-1,-1,:])

    assert out.shape == out_ref.shape
    assert out.stride() == out_ref.stride()
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    # check backward path
    g_ref= torch.randn_like(out)
    g = g_ref.detach().clone()

    out_ref.backward(g_ref)
    out.backward(g)

    assert torch.allclose(torch.cat((x0.grad, x1.grad), dim=2), x_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(weight0.grad + weight1.grad, weight_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(bias0.grad + bias1.grad, bias_ref.grad, rtol=rtol, atol=atol)
