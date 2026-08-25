# OCR evaluation contract v1

This document defines the reproducible scoring contract for candidate model reports. Raw predictions never decide whether they passed; the evaluator applies these rules after inference.

## Input records

Each image record has a stable ID, dataset, language, scenario, pixel width and height, ground-truth lines, and predicted lines. Each line contains a clockwise polygon in source-image pixel coordinates and a transcription. Ground-truth lines may set `ignore: true` for the dataset's do-not-care regions.

Korean and automatic-language routing are not model-quality samples. They remain separate Verto fallback tests. Empty-image samples are allowed and pass detection only when no non-ignored prediction remains.

## Polygon matching

Invalid or self-intersecting polygons are repaired with the geometry library's zero-width buffer operation; polygons still empty after repair are rejected as malformed input.

Predictions and non-ignored ground truth are assigned one-to-one with the Hungarian algorithm, maximizing polygon intersection-over-union. A pair is a detection true positive only when IoU is at least `0.5`. Unmatched ground truth is a false negative; unmatched prediction is a false positive. Predictions whose intersection-over-prediction area with an ignored polygon is at least `0.5` are removed before matching.

Detection precision, recall, and Hmean are micro-aggregated from counts. When both precision and recall are zero, Hmean is zero.

## Text normalization and distance

Transcriptions are normalized to Unicode NFC, CRLF and CR are converted to LF, and leading or trailing ASCII space, tab, and newline characters are removed. Case, punctuation, diacritics, full-width characters, and internal whitespace are preserved.

Characters are Unicode extended grapheme clusters (`\X`). CER is Levenshtein substitutions plus deletions plus insertions divided by the number of ground-truth grapheme clusters. A matched pair contributes its edit operations. An unmatched ground-truth line contributes deletion of all its characters. An unmatched prediction contributes insertion of all its characters.

The Paddle-hosted CTW1500 archive stores the transcription field as the placeholder `0` for every polygon. CTW1500 therefore contributes to detection metrics only; it is excluded from CER, exact-line accuracy, and end-to-end character accuracy.

The displayed end-to-end character accuracy is clamped to zero when insertions push CER above 1. Paired bootstrap deltas use the unclamped `1 - CER` value so two poor candidates do not become indistinguishable merely because both displayed accuracies reached zero.

Raw Verto reports keep cold Core ML compile/load time in the run metadata. Per-image timings separate canvas/crop/tensor preprocessing, detector inference plus DB post-processing, accumulated recognizer inference, CTC decode, recognizer time per attempted line, and end-to-end time. These are measurements only; Verto does not decide whether a candidate passes.

Exact-line accuracy uses only matched lines and requires identical normalized strings. End-to-end character accuracy is `max(0, 1 - total_edits / max(1, total_ground_truth_characters))`. Line recall is the matched non-ignored ground-truth line count divided by all non-ignored ground-truth lines.

## Aggregation and confidence intervals

Reports include overall micro metrics and slices by dataset, supported language, and scenario. Candidate-versus-baseline deltas use paired bootstrap resampling at the image level with NumPy random seed `20260824` and `10,000` replicates. The reported interval is the percentile 95% interval from the 2.5th and 97.5th percentiles.

Private final-holdout images are evaluated only after public-data candidate selection. A failure on that holdout rejects the candidate; it does not become a tuning example.
