import time
from pathlib import Path

from diffusers import FlaxAutoencoderKL
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


image_path = "input2.png"
model_name = "stabilityai/sdxl-vae"

dtype = jnp.float32


def load_image(path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(size, Image.BILINEAR)
    image = np.asarray(image).astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return image


def save_image(chw_image: np.ndarray, path: str | Path) -> None:
    image = np.transpose(chw_image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


print(f"jax backend              : {jax.default_backend()}")

# 1) load vae
vae, params = FlaxAutoencoderKL.from_pretrained(model_name, from_pt=True, dtype=dtype)


@jax.jit
def encode_fn(params, x: jnp.ndarray):
    posterior = vae.apply({"params": params}, x, deterministic=True, method=vae.encode).latent_dist
    latents_mode = posterior.mode()
    latents = latents_mode * vae.config.scaling_factor
    return posterior.mean, posterior.logvar, latents_mode, latents


@jax.jit
def decode_fn(params, latents: jnp.ndarray):
    decoded = vae.apply(
        {"params": params},
        latents / vae.config.scaling_factor,
        deterministic=True,
        method=vae.decode,
    ).sample
    return decoded

# 2) count params
total_params = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
print(f"Total params: {total_params:,} ({total_params / 1e6:.2f} M)")

# 3) preprocess image
x = load_image(image_path)
x = jnp.asarray(x, dtype=dtype)

# warmup
_ = encode_fn(params, x)
posterior_mean, posterior_logvar, latents_mode, latents = encode_fn(params, x)
posterior_mean, posterior_logvar, latents_mode, latents = jax.block_until_ready(
    (posterior_mean, posterior_logvar, latents_mode, latents)
)
_ = decode_fn(params, latents)

# 4) encode timing
start = time.perf_counter()
posterior_mean, posterior_logvar, latents_mode, latents = encode_fn(params, x)
posterior_mean, posterior_logvar, latents_mode, latents = jax.block_until_ready(
    (posterior_mean, posterior_logvar, latents_mode, latents)
)
encode_time = time.perf_counter() - start

# 5) print encoder output shapes
print("input shape              :", x.shape)
print("posterior.mean shape     :", posterior_mean.shape)
print("posterior.logvar shape   :", posterior_logvar.shape)
print("latent(mode) shape       :", latents_mode.shape)
print("scaled latent shape      :", latents.shape)

print("latent min/max           :", float(jnp.min(latents)), float(jnp.max(latents)))
print("latent has nan           :", bool(jnp.isnan(latents).any()))

# 6) decode timing
start = time.perf_counter()
decoded = decode_fn(params, latents)
decoded = jax.block_until_ready(decoded)
decode_time = time.perf_counter() - start

print("decoded shape            :", decoded.shape)
print("decoded min/max raw      :", float(jnp.min(decoded)), float(jnp.max(decoded)))
print("decoded has nan          :", bool(jnp.isnan(decoded).any()))

# 7) clamp to [0, 1]
decoded_vis = jnp.clip(decoded / 2.0 + 0.5, 0.0, 1.0)
orig_vis = jnp.clip(x / 2.0 + 0.5, 0.0, 1.0)

# 8) reconstruction error
mse = float(jnp.mean((decoded_vis - orig_vis) ** 2))
mae = float(jnp.mean(jnp.abs(decoded_vis - orig_vis)))

if mse > 0:
    psnr = 10 * np.log10(1.0 / mse)
else:
    psnr = float("inf")

print(f"MSE                      : {mse:.8f}")
print(f"MAE                      : {mae:.8f}")
print(f"PSNR                     : {psnr:.4f} dB")

# 9) timing
total_time = encode_time + decode_time
print(f"Encode time              : {encode_time * 1000:.2f} ms")
print(f"Decode time              : {decode_time * 1000:.2f} ms")
print(f"Total time               : {total_time * 1000:.2f} ms")

# 10) convert to numpy for save / vis
orig = np.asarray(orig_vis[0])
recon = np.asarray(decoded_vis[0])

print("orig min/max             :", float(orig.min()), float(orig.max()))
print("recon min/max            :", float(recon.min()), float(recon.max()))

# 11) direct save images
save_image(orig, "orig.png")
save_image(recon, "reconstructed.png")

print("saved orig.png")
print("saved reconstructed.png")

# 12) compare figure
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.transpose(orig, (1, 2, 0)))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(recon, (1, 2, 0)))
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.savefig("comparison.png", dpi=200, bbox_inches="tight")
print("saved comparison.png")
plt.show()
