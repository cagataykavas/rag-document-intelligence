# Lossless public evidence packaging

The oversized public evidence file is stored as ordered UTF-8 chunks in
`evidence/packed/`. No records, whitespace, metrics, timings, source hashes or
attribution have been changed. This is transport packaging, not a new experiment.
The manifest records the original path, exact byte count, SHA-256, Git blob SHA-1,
and each chunk's byte count and SHA-256. Chunks are sequential, at most 48,000
bytes, split only at UTF-8 boundaries, without inserted delimiters or newlines.
Concatenation is byte-for-byte lossless; chunks need not be independently valid JSON.

## Materialize before tests, audits or reading the original file

From a clean checkout on Linux/macOS (Python 3.10+; no dependencies or network):

```sh
python tools/restore_evidence.py
python -m pytest -q
ruff check src app tests tools
```

The command verifies every chunk and the reconstructed file before writing the
original path. The materialized original is ignored by Git. Repeating the command
accepts only an identical existing regular file. Missing/corrupt/reordered chunks,
unsafe paths, symlink inputs/outputs/parents, and unrelated existing output files
are rejected. Parent directories must already exist. POSIX directory descriptors,
no-follow opens and exclusive file creation prevent following symlinks or replacing
an existing file. Interrupted writes may leave a partial output, which the next
run refuses; inspect and remove that partial file explicitly before retrying.

The manifest is an integrity record, not a signature: establish provenance from
the reviewed Git commit. Do not use an untrusted modified manifest. To reproduce
a new measurement, use a separate output directory as the experiment guides say.
Original measured application source and all other evidence files are unchanged.
CI materializes before testing; no model inference or downloads are added by this
packaging command. Fifteen added tests cover exact/idempotent reconstruction,
missing/corrupt/reordered chunks, incorrect lengths/digests, unsafe destinations,
existing files, and symlink outputs, parent directories and chunks.

Local measurement commit IDs and connector-created public commit IDs may differ.
Use the frozen file digests and publication tree mapping to establish equivalence;
packaging does not invalidate the original source archives or measurements.
