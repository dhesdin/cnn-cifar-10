# Report — ResNet18 on CIFAR-10

## Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 3070 Ti |
| Batch size | 128 |
| Max epochs | 15 |


---

## Results Summary

| Run | Input | Freeze | LR | Best Acc | Best Epoch | Epochs Run | Time | VRAM | GPU Util |
|-----|-------|--------|----|----------|------------|------------|------|------|----------|
| 1 | 32×32 | Yes | 0.001 | 41.3% | 5 | 10/15 | 24s | 737 MB | 74% |
| 2 | 224×224 | Yes | 0.001 | 80.5% | 14 | 15/15 | 389s | 1956 MB | 93% |
| 3 | 224×224 | No | 0.0001 | 95.5% | 7 | 12/15 | 779s | 5743 MB | 94% |
| 3b (repro) | 224×224 | No | 0.0001 | 95.5% | 10 | 12/15 | 772s | 5947 MB* | 95% |

---

## Run Notes

### Run 1 — Wrong input size
Input resized to 32×32 before feeding a ResNet18 pretrained on ImageNet (expected 224×224). The backbone cannot extract meaningful features at that resolution, leading to unusable representations and low accuracy (41.3%). No overfitting observed.

### Run 2 — Frozen backbone, correct input size
Input correctly resized to 224×224. Backbone frozen, only the classifier head is trained. Good convergence: both train and test loss decrease steadily, reaching ~0.6. Accuracy reaches **80.5%**. No overfitting. ~26s per epoch. VRAM usage increases due to larger input size, but remains manageable.

### Run 3 — Full fine-tuning (backbone unfrozen)
Backbone unfrozen with a lower LR (0.0001). Strong convergence early on, but **overfitting starts around epoch 7–8** — train loss keeps dropping to ~0.03 while test loss stops improving and oscillates between ~0.15 and ~0.18. Accuracy reaches **~95.5%** at epoch 7, then plateaus (oscillating between ~94.9% and 95.5%) rather than clearly degrading. ~64s per epoch due to full backprop through the backbone. VRAM usage is significantly higher due to full fine-tuning.

### Run 3b — Reproduction of run 3
Same config as run 3 (same `config.yaml`, seed 42), re-run to check reproducibility.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------|---|---|---|---|---|---|---|---|---|----|----|----|
| Run 3 acc (%) | 93.4 | 94.0 | 94.5 | 94.2 | 95.0 | 95.4 | **95.5** | 95.2 | 95.1 | 94.9 | 95.4 | 95.3 |
| Run 3b acc (%) | 93.3 | 93.9 | 94.8 | 94.4 | 94.7 | 94.9 | 95.4 | 95.1 | 94.9 | **95.5** | 95.0 | 95.3 |

- Same best accuracy (95.5%), same number of epochs (12), same time (~772s vs 779s), average losses within 0.0005.
- Per-epoch accuracy differs by at most ~0.6 pt: the run is **not bit-exact reproducible** because the seed does not cover cuDNN non-deterministic kernels nor DataLoader workers.
- The best epoch moved from 7 to 10: after epoch ~7 the accuracy only oscillates by ±0.3 pt around 95.2%, so *which* epoch is the best is mostly noise. Model selection on a single best epoch should be read with that in mind.
- Early stopping still triggers at epoch 12: the epoch-10 gain (+0.08 pt over epoch 7) is below `delta` (0.1 pt), so it is not counted as an improvement.

# Limitations & Improvements
- Progressively fine tuning : start with frozen backbone and adjust gradually some parameters like the lr, the number of epochs, the batch size (to reduce VRAM usage but it will increase training time), and the number of layers to unfreeze.
- More data augmentation to reduce overfitting.
- Full determinism: set `torch.backends.cudnn.deterministic = True` / `benchmark = False` and seed the DataLoader workers (`worker_init_fn` + `generator`) to get bit-exact reruns.
- Save the best-epoch checkpoint instead of the last one: the saved model is the epoch-12 one, not the best-accuracy one.
