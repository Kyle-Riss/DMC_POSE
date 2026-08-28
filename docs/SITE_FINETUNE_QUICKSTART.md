# Fixed-camera site fine-tuning

The general base checkpoint is immutable. Each installation produces a new
site checkpoint and report under its own run directory.

1. Copy `external_datasets/manifests/site_finetune_template.json` and list
   reviewed fall and hard-negative clips. `group_id` must identify a person or
   recording session; one group cannot cross train/validation/test splits.
2. Validate paths, labels, and leakage without loading the model:

```bash
python scripts/finetune_swin3d_site.py \
  --manifest external_datasets/manifests/site_hospital_a_v1.json \
  --out-dir runs/video_verifier/sites/hospital_a_v1 \
  --validate-only
```

3. Run conservative adaptation. Only the last Swin3D block, normalization and
   binary head are updated; the rest of the large pretrained model stays
   frozen:

```bash
python scripts/finetune_swin3d_site.py \
  --manifest external_datasets/manifests/site_hospital_a_v1.json \
  --out-dir runs/video_verifier/sites/hospital_a_v1 \
  --epochs 3 --device cuda
```

The selected epoch and threshold come only from validation. Test is evaluated
after selection when supplied. A missing test set blocks promotion and the
result remains an offline site candidate. Deployment must still go through
central shadow validation; this command never changes the live service.
