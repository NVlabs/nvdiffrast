import nvdiffrast.torch as dr
import torch

def main():
    device = torch.device("cuda:0")

    for minibatch_size in [1, 4]:
        for channel_count in range(1, 18): # Channel count > 16 is expected to fail
                for filter_mode in ['nearest', 'linear']:
                    for per_channel_border_value in [True, False]:
                        if per_channel_border_value:
                            border_values = torch.rand(channel_count, dtype=torch.float32).tolist()
                        else:
                            border_values = torch.rand(1).item() # Channels share a border value

                        tex = torch.rand(minibatch_size, 64, 64, channel_count, dtype=torch.float32, device=device)

                        uvs = torch.rand(minibatch_size, 64, 64, 2, dtype=torch.float32, device=device)
                        uvs[..., 0] = 1.1 * (2 * (torch.rand_like(uvs[..., 0]) < 0.5) - 1) # Invalidate uvs to trigger border value sampling

                        # Failure is expected for more than 16 channels
                        try:
                            tex_samples = dr.texture(tex, uvs, filter_mode=filter_mode, boundary_mode='values', border_values=border_values)
                        except Exception as e:
                            assert channel_count > 16 and ("up to 16 channels" in str(e)), f"Unexpected error: {e}"
                            continue

                        assert channel_count <= 16, "Expected failure for more than 16 channels"

                        for c in range(channel_count):
                            assert torch.allclose(tex_samples[..., c], torch.tensor(border_values[c] if per_channel_border_value else border_values, device=device))

    print("All tests passed.")

if __name__ == '__main__':
    main()
