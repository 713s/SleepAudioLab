# Local data directory

Keep personal sleep recordings local. The repository `.gitignore` excludes common audio files and the working subdirectories below.

Suggested layout:

```text
data/
├── raw/          # original phone recordings
├── processed/    # normalized/chunked intermediates
└── labels/       # local annotations for future classifiers
```

Do not commit private overnight recordings to the public repository.
