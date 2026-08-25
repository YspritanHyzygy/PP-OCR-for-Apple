# Final v2 release evidence

`benchmarks/v2-final.json` is created only after all public corpora, the one-shot private holdout, iPhone 16 Pro measurements, and iOS 17 compatibility checks finish. The release workflow rejects a missing or incomplete file.

The report records:

- `releaseTag: "v2"` and `decision: "pass"`;
- anonymous private-holdout totals for exactly 36 images, confirmation that it was used for finalists only and not for tuning, and zero newly introduced crashes, empty results, or missing lines;
- iPhone 16 Pro OS, thermal state, `ALL` compute units, three warm-up runs, stage timing, memory, and energy evidence;
- iOS 17 compile, Core ML load, and inference results;
- a quality-gate result for tiny, small, and medium;
- a substantive-improvement result for the default small tier.

The file contains aggregate measurements only. Private photos and transcriptions remain outside git.
