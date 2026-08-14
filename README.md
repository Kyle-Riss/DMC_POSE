# Encrypted Private Snapshot

This repository intentionally contains no plaintext source tree or Git history.

- `DMC_POSE_secure_snapshot_20260814.tar.gz.gpg` is the encrypted current-worktree snapshot.
- The passphrase file is stored separately and must never be committed or uploaded.
- Verify the ciphertext with `sha256sum -c SHA256SUMS` before decrypting.

Restore on an authorized machine:

```bash
gpg --batch --pinentry-mode loopback \
  --passphrase-file DMC_POSE_secure_snapshot_20260814.key \
  --output DMC_POSE_secure_snapshot_20260814.tar.gz \
  --decrypt DMC_POSE_secure_snapshot_20260814.tar.gz.gpg

mkdir DMC_POSE
tar -xzf DMC_POSE_secure_snapshot_20260814.tar.gz -C DMC_POSE
```

The decryption key is not recoverable from this repository.
