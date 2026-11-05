"""Data-free forward/backward example; not an accuracy experiment."""
import argparse

import torch

from busemann_transformer import BusemannMHA, Hyperboloid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    spatial = (0.2 * torch.randn(1, 16, 16, device=args.device)).requires_grad_()
    point = Hyperboloid.proj(torch.cat((spatial[..., :1] * 0, spatial), dim=-1), 1.0)
    for name in ("busemann_log_depth", "busemann_log_depth_linear_dictionary"):
        attention = BusemannMHA(
            d_model=16, num_heads=4, c=1.0, dropout=0.0,
            attn_proj="hyp_linear", score_type=name,
            linear_boundary_directions=8,
        ).to(args.device)
        output = attention(point)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            Hyperboloid.minkowski_dot(output, output),
            output.new_full((1, 16, 1), -1.0), rtol=1e-4, atol=1e-4,
        )
        output[..., 1:].square().mean().backward(retain_graph=True)
        assert torch.isfinite(spatial.grad).all()
        print(f"{name}: shape={tuple(output.shape)}, finite forward/backward")


if __name__ == "__main__":
    main()
