```markdown
# Week 08 Progress Update(July 10 2026)

## Done
1. Continued the SSL side-channel experiments by increasing pre-training epochs to 100 for each method.
2. CPC, MAE, and BYOL were successfully improved after hyperparameter and representation tuning.
3. TS2Vec was tested with both 10 epochs and 100 epochs. The 10-epoch setting performed better, while the 100-epoch setting degraded.
4. SimCLR was also tested with 100 epochs and still reached final rank 0.

### The following chart shows the graphs for TS2Vec, SimCLR, CPC, MAE, and BYOL:

1. TS2Vec epoch 10  
![tsvec epoch 10](weekly_updates/week08/figures/ts2vec_ep10/ts2vec_ep10_linear_probe_rank.png)

2. TS2Vec epoch 100
![tsvec epoch 100](weekly_updates/week08/figures/ts2vec_ep100/ts2vec_ep100_linear_probe_rank.png)

3. SimCLR epoch 100  
![simclr epoch 100](weekly_updates/week08/figures/simclr_ep100/simclr_ep100_linear_probe_rank.png)

4. CPC epoch 100  
![cpc epoch 100](weekly_updates/week08/figures/cpc_ref_pred6_neg10_ep100/cpc_ref_pred6_neg10_ep100_linear_probe_rank.png)

5. MAE best run  
![mae epoch 100](weekly_updates/week08/figures/mae_patch5_mask30_dim320_ep100/mae_patch5_mask30_dim320_ep100_linear_probe_rank.png)

6. BYOL best run  
![byol best run](weekly_updates/week08/figures/byol_weakaug_shift3_noise0p01_scale0p0_mask0_ema0p996_proj128_meanmax_ep100/byol_weakaug_shift3_noise0p01_scale0p0_mask0_ema0p996_proj128_meanmax_ep100_linear_probe_rank.png)

### Result Summary
| Method | Variant | Epochs | Batch Size | LR | Representation Dim | Final Rank | Min Rank | Rank-0 Trace | Train Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TS2Vec | TS2Vec ep10 | 10 | 64 | `1e-3` | 320 | **0** | **0** | **167** | 327.40s |
| TS2Vec | TS2Vec ep100 | 100 | 64 | `1e-3` | 320 | 8 | 2 | -1 | 3219.10s |
| SimCLR | 1D SimCLR ep100 | 100 | 64 | `1e-3` | 320 | **0** | **0** | 6087 | 450.83s |
| CPC | CPC-ref pred6 neg10 ep100 | 100 | 64 | `2e-4` | 320 | **0** | **0** | 5723 | 647.46s |
| MAE-1D | patch5 mask30 mean+max ep100 | 100 | 128 | `1e-4` | 640 | **0** | **0** | 1783 | 1623.76s |
| BYOL-1D | weak augmentation mean+max ep100 | 100 | 128 | `3e-4` | 640 | **0** | **0** | **3** | 491.40s |

### Hyperparameter Details

| Method | Key Hyperparameters |
|---|---|
| TS2Vec ep10 | `output_dims=320`, `hidden_dims=64`, `depth=6`, `encoding_window=full_series`, `target_byte=2`, `classifier=LogisticRegression` |
| TS2Vec ep100 | `output_dims=320`, `hidden_dims=64`, `depth=6`, `encoding_window=full_series`, `target_byte=2`, `classifier=LogisticRegression` |
| SimCLR | `repr_dim=320`, `proj_dim=128`, `augmentation=random_shift(10)+gaussian_noise(0.05)`, `target_byte=2`, `classifier=LogisticRegression` |
| CPC | `repr_dim=320`, `context_dim=320`, `prediction_steps=6`, `negative_samples=10`, `encoder_strides=(2,2,2,2,1)`, `representation=context_mean`, `target_byte=2`, `classifier=LogisticRegression` |
| MAE-1D | `trace_len=700`, `patch_size=5`, `mask_ratio=0.30`, `embed_dim=320`, `depth=4`, `num_heads=8`, `norm_pix_loss=False`, `representation=mean+max`, `pooled_repr_dim=640`, `target_byte=2`, `classifier=LogisticRegression` |
| BYOL-1D | `repr_dim=320`, `pooled_repr_dim=640`, `proj_dim=128`, `hidden_dim=512`, `ema_decay=0.996`, `augmentation=random_shift(3)+gaussian_noise(0.01)`, `scale_jitter=0`, `time_mask=0`, `representation=mean+max`, `target_byte=2`, `classifier=LogisticRegression` |

### Observations

1. TS2Vec reached final rank 0 with 10 epochs, but degraded when trained for 100 epochs. This suggests that longer SSL pre-training does not always improve leakage-discriminative representations.
2. SimCLR still reached final rank 0 after 100 epochs, and its rank-0 trace improved from the previous 10-epoch run.
3. CPC reached final rank 0 after increasing training to 100 epochs with `prediction_steps=6` and `negative_samples=10`.
4. MAE became successful after changing the downstream representation from mean pooling to mean+max pooling. This suggests that local leakage information was being diluted by mean pooling alone.
5. BYOL improved significantly after replacing strong augmentation with weak augmentation and changing representation pooling to mean+max. The best BYOL run reached rank 0 after only 3 attack traces.

## Plan for Next Week
* Verify the best-performing configurations with repeated runs to check stability.
* Compare rank curves across all best methods in one combined plot.
* Clean up experiment output paths and ensure each run has a unique `run_name`.
* Prepare a concise summary table for the advisor meeting.

## Blockers
* Some earlier MAE runs reused the same output directory name, so several rank plots may have been overwritten. Future runs should use unique names such as `mae_patch5_mask30_meanmax_dim320_ep100`.
```
