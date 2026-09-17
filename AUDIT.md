# Independent Review

This review checked existing media, manifests, feature tensors, histories and
checkpoints without retraining, modifying splits, selecting on test, or pushing.
Machine-readable evidence: [verification.json](results/audit/verification.json).

## Findings

1. **Test-based selection risk:** mean wins by validation macro F1 (about 0.313
   versus 0.204). Attention's higher test accuracy cannot reverse that selection.
   Both previously inspected test results are exploratory comparisons. They do
   not establish an SE improvement or an independent generalization estimate.
2. **Source independence remains unverified:** both named DVD source groups
   cross splits. No detected audio overlap is not a guarantee against repeated
   babies, source cues, transformations, or short shared fragments.
3. **Publication inconsistencies corrected:** README/status previously said
   untrained, while current checkpoints existed. Default notebook review still
   showed the old 153-clip results. Current artifacts are now separately packaged
   under results/current_175; historical artifacts remain untouched.
4. **Legacy checkpoint regression fixed:** adding SE unconditionally prevented
   loading pre-SE attention weights. Restore now infers legacy architecture from
   state keys when explicit metadata is absent. New checkpoints record SE usage.
   SE bottlenecks also have a minimum width of one for small feature dimensions.
5. **Historical causal claim unsupported:** contamination undermines the old
   81.25% estimate, but the fraction attributable to leakage is unknown. Dataset,
   split and architecture changes prevent a controlled attribution.

## Verified

- Root and publication media match all 175 hashes and manifest assignments.
- Each class physically contains 28 train-folder / 7 test-folder videos, with
  22/6/7 internal assignments. Only these folders enter feature preparation.
- Explicit train-only restrictions and declared groups stay together.
- Feature rows, labels, masks, class ordering and hashes match current media.
- Existing best checkpoints are the earliest validation-F1 maxima in histories.
- Recomputed validation/test confusion matrices and scalar metrics match saved
  results, including the attention validation/test reversal.
- Ten raw-video spot checks (one validation and test clip per class) reproduce
  cached mel and visual tensors. This is not an exhaustive raw-feature rebuild.
- The attention implementation correctly masks logits to negative infinity
  before softmax; multiplying logits by a binary mask would be incorrect.
- Six previously reported duplicate/subclip pairs were separately screened;
  all are now confined to training. Their scores are recorded in the report.
- Available normalized eairh source intervals have no cross-split overlap.
- All 18 unit tests passed, including padding invariance, SE gradients, and legacy
  checkpoint loading. All four packaged current/historical checkpoints load and
  produce matching outputs through root code and notebook definitions.
- Notebook executes all 22 cells in review-only mode, not training mode.

## Audio Screen Scope

FFmpeg decoded mono audio at 8 kHz. FFT cross-correlation was normalized with
local means and energies (Pearson correlation at each alignment). All 8,200
cross-split pairs were screened, including cross-class pairs. An absolute score
above 0.85 would be flagged; none was found. Eligible alignments require at least
250 ms AND 75% of the shorter clip. Shorter shared fragments, time stretching,
noise and other transformations may evade this screen. It is not identity matching.

The repartition script hard-codes assignments; it does not reconstruct or preserve
the claimed connected-component graph. Non-eairh manifests largely lack original
source timestamps, so their temporal isolation cannot be independently established
from metadata alone. Assigned validation groups encode split policy, not evidence
that clips depict different babies. Exhaustive visual comparison was not performed.

## Interpretation

The attention reversal is real in the saved artifacts, not a discovered class-order
or checkpoint mismatch. Source/scene composition, duration differences and small
sample variance are possible explanations, not established causes. Duration summaries
are included in the report. Do not adjust the split to improve these known test scores.

For a stronger experiment, predefine source/infant groups and compare architectures
using grouped validation within development data. Reserve genuinely new, independently
labeled source footage for final evaluation. Current eairh boundaries and cry labels
still need human verification. Keep this as a research prototype, not medical guidance.
