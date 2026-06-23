# LAB NOTEBOOK — compression-ecg

Living document of the optimization loop. Three tiers: INSIGHTS (durable
lessons), GRAVEYARD (one line per dead idea), ACTIVE LOG (last ~15 verbose
iteration blocks; aged blocks roll into the graveyard via
`python3 loop.py compact`).

**Objective**: maximize lossless compression ratio (raw_bytes / compressed_bytes)
of MIT-BIH ECG, byte-exact, on the subject-disjoint HOLDOUT records.
**Scoring**: primary = `ratio_holdout`; promotion is holdout-only. Deterministic
evaluator (same codec + frozen data → identical ratio). Hard gate = byte-exact
`decode(encode(x)) == x` on every record + a static no-peeking import check.

## INSIGHTS
- **Data**: MIT-BIH Arrhythmia v1.0.0. 12 records — train {101,103,105,108,209,
  215,230}, holdout {100,119,207,214,232}. Each record = 2 leads (MLII + a
  precordial lead), 360 Hz, 650,000 samples/lead (~30 min). Split is
  subject-disjoint (no record in both).
- **Representation we compress** (canonical `.ecg`): WFDB format-212 unpacked to
  channel-major int16 LE, with a 10-byte header. Each record = 2,600,010 bytes.
  All codecs (baselines + candidates) compete on THIS — so a win is real
  modeling, not undoing the 12-bit packing.
- **Sample statistics**: 12-bit signed ADC codes; real values cluster near the
  ~1024 baseline and almost never exceed ±2048, so the high ~5 bits of each
  int16 are near-constant (sign/zero). General compressors already exploit this.
- **Verified baselines (holdout ratio)**: gzip-9 **2.26×**, xz-9e **3.23×**,
  bzip2-9 **3.85× ← the bar**. (zstd-19 ≈ 2.7–3.3×, not shipped as a baseline.)
- **Seed champion (3.90×)**: per-channel first-difference (int16 wraparound) +
  LZMA. It only *ties* bzip2 (+1.2%). Plain delta is not enough — bzip2's BWT is
  a strong opponent on ECG.
- **The path to a convincing win**: the lossless-ECG literature wins with a
  PREDICTOR + a RESIDUAL ENTROPY CODER. Better prediction (fixed AR/polynomial
  like FLAC orders 1–3, or adaptive LMS) shrinks residual magnitude; coding the
  residuals with Golomb–Rice or a range/arithmetic coder beats handing them to a
  generic backend. Cross-lead correlation (predict lead 2 from lead 1) is extra
  headroom. Realistic ceiling ≈ 4.5–5× on this representation.
- **Performance constraint**: every eval compresses 5 holdout + 7 train records
  (~13 MB + ~18 MB). Keep candidates fast — vectorize predictors in numpy. A
  pure-Python per-sample arithmetic coder over 1.3M samples/record will be slow;
  prefer numpy transforms feeding LZMA/bz2, or a vectorized Rice coder, before
  attempting a scalar entropy coder.
- **LZMA exploits ECG periodicity via dictionary matching, not entropy alone**: on train/215, order-2 predictor reduces residual std by 49% but LZMA output is 5.8 KB LARGER than order-1. LZMA finds repeating heartbeat signatures (~385-sample period) in first-difference space; higher-order diffs fragment this structure. Conclusion: only use a higher-order predictor if paired with a coder that benefits from lower per-sample entropy (Rice/arithmetic), not with LZMA. Alternatively, a predictor that PRESERVES the ECG's periodic structure while reducing residuals (e.g., AR model fitted to the beat period) could pair well with LZMA.
- **LZMA beats bz2 when paired with delta, despite bz2 beating LZMA on raw ECG**: delta+LZMA=3.8959× vs delta+bz2=3.8771× (holdout). Train/215 probe (bz2 locally better) didn't generalize. Delta preprocessing changes the statistical structure in a way that favors LZMA's sliding-window dictionary matching over bz2's BWT. Exhausted: both delta backends now tried. Next: need real signal modeling — LMS adaptive predictor or rice-coder lineage.
- **MIT-BIH is an arrhythmia database — beat-to-beat prediction fails**: period-lag predictor (predict s[i] from s[i-T]) dropped from 3.90× to 2.89×. Irregular beat timing (PVCs, AF, compensatory pauses) means s[i-T] is often the wrong beat, producing large aperiodic residuals. LZMA's ~385-sample periodicity exploit is about the repeating QRS SHAPE within one channel's byte stream, NOT inter-beat temporal similarity. Any approach that relies on beat-to-beat alignment will fail on this dataset.
- **Codec contract**: one file `candidates/<id>.py` with `encode(bytes)->bytes`
  and `decode(bytes)->bytes`, lossless, allowed imports only (numpy/struct/zlib/
  bz2/lzma/math/array/collections/itertools/functools/typing/heapq). Parse the
  canonical header; model per channel.
- **NLMS hyperparameter choice is per-record — best-of-N across (order, mu) pairs is essential**: order=4 mu=0.10 only helped record 207 (+0.015× total). Adding order=2 mu=0.20 and order=8 mu=0.05 gave +0.11× — 7× larger gain. Each holdout record's optimal NLMS config differs by morphology complexity; best-of-N per channel captures this without regression risk.
- **The NLMS grid is now near-exhausted — diminishing returns**: extending best-of-8 → best-of-12 with long-memory taps (order=16 mu=0.05, order=32 mu=0.03, spanning a full QRS at 360 Hz) gained only +0.02× (iter 12), vs +0.11× from the prior 4→8 widening. Long taps win only on the wide-QRS minority (record 207); most records still pick raw or a short order. NLMS cost is dominated by the 650k-iter Python loop, NOT the order, so high-order configs are cheap to add — but the ratio headroom from more (order,mu) points is largely spent. Future gains need a structurally different lever (cross-lead context, or a residual entropy coder), not more NLMS taps.
- **Cross-lead delta correlation is essentially zero in MIT-BIH**: adding ch0_delta as an extra NLMS tap for ch1 (joint order+1 filter, adaptive, regression-proof via best-of-16) gave 0.000× gain on holdout (iter 13). The best-of-16 never selected any cross-lead option — the NLMS cross-lead weight converges to ~0 for all 5 holdout records. Co-temporal QRS events in MLII and precordial leads do NOT produce exploitable linear delta correlation, presumably because their morphologies differ enough to cancel any predictive signal. Cross-lead in delta space is fully exhausted. Remaining unexplored lever: rice-coder (Golomb–Rice on NLMS residuals as bz2 alternative).
- **LZMA ceiling is hit at preset=6 for seed's interleaved format**: all presets 6–9 give IDENTICAL compressed output (604,660B on train/101). Any further gain requires CHANGING THE DATA FORMAT, not tweaking LZMA settings.
- **Byte-splitting fails because hi bytes are free in LZMA matches**: LZMA's 770-byte heartbeat match token covers BOTH lo and hi bytes for zero extra cost. Separating them forces explicit coding of the hi plane (~152KB for its ~1 bit/sample binary entropy), swamping any lo-plane benefit.
- **Variable-length zigzag encoding + bz2 confirmed at 4.09×**: zigzag-encode delta (signed→unsigned), pack as 1 byte if zzag<128 (99.57% of ECG deltas) → stream ≈1.3MB. bz2 BWT exploits ≈385-byte heartbeat period on this half-size stream. Harness confirmed: ratio_holdout=4.0907, ratio_train=4.0098. No overfitting. Probe prediction was spot-on. The mechanism: halving the pre-compression stream doubles the frequency of periodic heartbeat patterns relative to bz2's BWT block size.
- **Second-order predictor + varzz + bz2 fails even without LZMA (3.92×)**: fixpred2_varzz_bz2_v1 confirmed -0.17× vs champion. Root cause: T/P-wave reversals in ECG create mid-range residuals (20-50 range in zigzag) that vary beat-to-beat more than QRS shape does. This spreads bz2's byte distribution and makes beat patterns LESS stereotyped, reducing BWT match frequency. The first-order delta is the sweet spot: QRS is stereotyped and T/P residuals are small → maximally periodic byte stream for bz2.
- **Per-channel bz2 adds +0.013× (4.1043×)**: in the combined stream (1.3MB), bz2's 900KB block puts ch0 fully in block 1 but splits ch1 across blocks 1 and 2. The split means ch1 gets 649 periods in block 1 and 1039 in block 2 — separate BWT transforms, no cross-block matching. Per-channel: each 650KB stream gets one 900KB block → 1688 periods fully visible to BWT → marginally better compression of ch1.
- **Nibble-pair encoding adds +0.054× (4.1586×) via best-of-two per channel**: pack consecutive delta pairs where both zzag<8 into 1 byte (3+3 bits), shrinking stream by 35-44% for high-small% records. bz2 sees ~2220 heartbeat periods per block vs 1688. BUT for records with small%<70% (e.g. arrhythmia record 215, 60% small), the greedy pair boundaries shift inconsistently at each heartbeat position across beats — BWT alignment is destroyed and bz2 LOSES bytes. Fix: per-channel best-of-two (try both nibble and varzz, pick smaller bz2, store a 1-byte flag). Guarantees no regression; 6/7 train records benefit from nibble, 1/7 (215) falls back to varzz with <2 byte overhead.
- **Rice-k beats bz2 on near-iid NLMS residuals (+0.006×, 4.3113×)**: after NLMS removes periodic structure from residuals, bz2's BWT finds less to exploit. For channels where NLMS has truly flattened the distribution (irregular-rhythm records), Rice-k (k=1/2/3, vectorized numpy) achieves near-entropy coding with 4-byte overhead vs bz2's block overhead. Best-of-30 (12 bz2 + 18 Rice options) guarantees no regression; gain is modest because most channels still have residual periodicity that bz2 exploits.
- **Rice-NLMS encoding grid is fully exhausted (iter 15)**: extending to best-of-48 by adding LZMA (preset=4) as a third block backend and Rice k=0 and k=4 gave 0× gain on holdout. bz2 BWT beats LZMA on ALL holdout channels even after NLMS — confirming that some periodic structure survives NLMS, making BWT superior to LZ77 sliding window even on these near-iid residuals. Rice k=0 never wins (residual magnitudes too high even for best NLMS configs). Remaining gain must come from a BETTER PREDICTOR (LPC per segment or cascade NLMS) to further eliminate the residual structure that keeps bz2 winning.
- **LPC (MMSE-optimal per stationary segment) gave 0× vs NLMS on diff domain (iter 16)**: per-512-sample Levinson-Durbin LPC doesn't beat NLMS on first differences. ECG is non-stationary (QRS vs baseline within each 1.4-beat window), so LPC's "optimal" block coefficients are computed on mixed-morphology segments and add overhead (coefficient table per segment). NLMS's online adaptation already achieves near-LPC quality without the overhead. Diff-domain linear predictors are exhausted.
- **The 0× wall (iters 15–19)**: five consecutive failed attempts — LZMA+Rice extension, per-segment LPC, period predictor on residuals, cascade NLMS (order-1 targeting lag-1=0.807), raw-signal NLMS — all stall at exactly 4.3113×. bz2's BWT captures the remaining residual structure in the packed byte stream implicitly; neither diff-domain nor raw-domain predictors can remove what bz2 already exploits. Raw-domain NLMS (AR on int16 samples) did not produce better residuals than diff-domain NLMS for any holdout channel. The architecture bottleneck is the bz2+NLMS combination itself: next attempt must change the entropy coding layer (context-adaptive Rice or per-segment parameter selection), not just the predictor.
- **Stream compaction beats byte-phase stability for bz2's BWT (iter 21, 0×)**: phase-stable packing (exactly 1 byte/sample, rare zzag≥255 escaped to a side stream) was added as a best-of-36 option to give every QRS shape a single consistent byte phase — the hypothesis being that varzz/nibble's 2-byte escapes AT the QRS peaks cause cross-beat phase drift that splits each QRS into multiple BWT patterns. Result: psb was NEVER selected for any holdout channel. The phase-stable main stream is exactly n bytes (650K), but nibble compacts the same channel to ~400-500K; bz2's gain from the shorter/denser input dominates the phase-alignment benefit. So the nibble-alignment lesson does NOT generalize into "fixed phase wins" — what bz2 actually rewards is a SHORTER stream, and the per-channel best-of already buys phase consistency implicitly by only choosing nibble when small% is high enough that pairs form stably. Entropy-layer reshaping that does not also shrink the stream cannot beat the champion.
- **First wall-breach candidate must do BOTH (post-iter-21 synthesis)**: six straight 0× iters (15–21) span every predictor family AND a packing-layer reshape. The only levers that ever moved holdout were ones that made the bz2 input strictly SHORTER (varzz halving +0.19, nibble compaction +0.05) or strictly SMALLER-magnitude residuals that then compacted shorter (NLMS grid +0.11/+0.02). bz2 is a near-optimal exploiter of the QRS-shape repetition already; to beat it you must either (a) feed it a stream that is both shorter AND retains the periodicity (a denser entropy-optimal compaction that is still byte-granular and phase-consistent — e.g. a 4-or-6-bit fixed-width small-value plane), or (b) replace bz2 with a coder that models cross-beat repetition AND sub-byte entropy jointly (an adaptive arithmetic/range coder with a context conditioned on recent residual magnitude). Pure-marginal-entropy coders (Rice) and pure-reshape (psb) both already lost.
- **Split-tap NLMS breaks the 0× wall (iter 26, +0.002×)**: after 11 consecutive failures (iters 15–25), split-NLMS (short taps at lags 1-16 + long taps at FFT-estimated beat period T to T+15) achieved +0.002× by reducing QRS residuals at the integer level — something all encoder-layer and same-lag predictors couldn't do. The cross-beat taps give NLMS access to the previous beat's QRS morphology without explicit onset detection. Gain is modest; next step is to widen the split config space (more period estimates, tighter T centering, more (short, long, mu) combos in the best-of-N).


## Live lineages
| lineage | status | best | notes |
|---|---|---|---|
| delta-backend | STALLED | 3.90× | both backends tried (LZMA=3.90×, bz2=3.88×); varzz-bz2 supersedes this lineage |
| fixed-predictor | GRAVEYARD | 3.78× | higher-order diffs break LZMA/bz2 heartbeat-period matching; only viable with Rice/arithmetic coder |
| nlms-bz2 | STALLED | 4.305× | best-of-12: raw + NLMS order=2/4/8/16/32, each with varzz/nibble; grid exhausted. superseded by rice-nlms |
| psb-nlms | GRAVEYARD | 4.3113× | phase-stable 1-byte/sample packing (zzag≥255 escaped to side stream) as best-of-36 option; 0× (iter 21). Never selected — nibble compacts the channel shorter (~400-500K vs psb's fixed 650K) and bz2 rewards the shorter stream over phase consistency |
| cross-lead | GRAVEYARD | 4.305× | joint NLMS (within-ch1 + ch0 cross-tap) gave 0× gain; best-of-16 never selects xl options; delta-domain cross-lead correlation ≈ 0 in MIT-BIH |
| rice-nlms | STALLED | 4.3113× | Rice-k (k=1/2/3) as additional backend to bz2; best-of-30 (iter 14 +0.006×). Iter 15 extended to best-of-48 (+LZMA, +k=0/4): 0× gain. Grid fully exhausted; bz2 wins all channels |
| nibble4-nlms | GRAVEYARD | 4.3113× | 4-bit fixed-width nibble + bz2 phase-consistent packing: 0× (iter 22). Main stream 650K→325K bytes but longer than nibble's ~400K; bz2 rewards shorter stream over phase consistency |
| beat-template | GRAVEYARD | 4.3113× | Onset-aligned template subtraction: 0× on raw diff (iter 23) and 0× on NLMS residuals (iter 24). bz2 already captures cross-beat QRS byte patterns; template adds onset list overhead with no net gain |
| badapt-rice | GRAVEYARD | 4.3113× | Block-adaptive Rice (k=0-7 per 256-sample block): 0× (iter 25). Per-block k can only improve on fixed-k Rice; Rice still loses to bz2's cross-beat periodicity on most channels |
| split-nlms | ACTIVE | 4.3133× | Split-tap NLMS (short lags 1-16 + long lags at beat period T): +0.002× (iter 26). First wall breach after 11 consecutive 0×. Cross-beat taps reduce QRS residuals at source. Config needs tuning |
| lpc-bz2 | GRAVEYARD | 4.3113× | per-512-sample Levinson-Durbin LPC on diff stream, best-of-45; 0× vs champion (iter 16). MMSE-optimal per stationary block but ECG is non-stationary within each 1.4-beat segment; coefficient overhead negates any residual savings |
| stage2-period | GRAVEYARD | 4.3113× | NLMS then OLS period predictor r2=r1-w*r1[n-T] on residuals; best-of-60; 0× (iter 17). Beat-period lag-T correlation in NLMS residuals is small vs lag-1 autocorrelation (0.807); period stage never selected for any holdout channel |
| cascade-nlms | GRAVEYARD | 4.3113× | stage-2 NLMS (orders 1/2/4, multiple mu) on stage-1 residuals; best-of-65 including order-1/mu=0.50 targeting lag-1=0.807; 0× (iter 18). bz2's BWT already captures the lag-1 autocorrelation in the packed byte stream; removing it at integer level doesn't reduce bz2's output |
| raw-nlms | GRAVEYARD | 4.3113× | NLMS AR predictor on raw int16 signal (not first-differences); best-of-50 = champion 30 + 4 raw configs × 5 packs; 0× (iter 19). Raw-domain AR(2/4/8/16) residuals not better than diff-domain NLMS for any holdout channel; best-of-50 always selects champion diff-domain options |
| period-predictor | GRAVEYARD | 2.89× | predict s[i] from s[i-T]; MIT-BIH is an arrhythmia DB — beat timing/morphology are irregular, so lag-T residuals are LARGE and aperiodic |
| byte-split | GRAVEYARD | 3.28× | hi bytes cost ~152KB extra when separated from LZMA's 770-byte match tokens; lo-plane match-distance improvement is swamped |
| varzz-bz2 | STALLED | 4.09× | combined stream hits 4.09×; per-channel supersedes this variant |
| fixed2-varzz-bz2 | GRAVEYARD | 3.92× | order-2 predictor + varzz + bz2: T/P-wave variability spreads byte distribution and breaks BWT beat matching |
| perchan-varzz-bz2 | STALLED | 4.10× | per-channel bz2 gives each 650KB varzz stream its own 900KB block → 1688 heartbeat periods fully in BWT; confirmed 4.1043× holdout (iter 8); nibble-adapt supersedes |
| nibble-adapt-bz2 | STALLED | 4.16× | per-channel best-of-two (nibble-pair vs varzz), pick smaller bz2; confirmed 4.1586× holdout (iter 9); nlms-bz2 supersedes |

## Leaderboard snapshot
(maintained by hand from `leaderboard.json` after each iteration)
- Champion: **split_nlms_v1** — ratio_holdout 4.3133× (iter 26).
- Baselines: gzip 2.26× · xz 3.23× · bzip2 3.85×.
- iter 1 seed: 3.90× (former champion).
- iter 2 fixed_pred_v1: 3.78× (not promoted).
- iter 3 delta_bz2_v1: 3.88× (not promoted).
- iter 4 period_pred_v1: 2.89× (not promoted).
- iter 5 byte_split_v1: 3.28× (not promoted).
- iter 6 varzz_bz2_v1: 4.09× (PROMOTED).
- iter 7 fixpred2_varzz_bz2_v1: 3.92× (not promoted).
- iter 8 perchan_varzz_bz2_v1: 4.10× (PROMOTED).
- iter 9 nibble_adapt_bz2_v1: 4.16× (PROMOTED).
- iter 10 nlms_bz2_v1: 4.17× (PROMOTED).
- iter 11 nlms_bz2_v2: 4.28× (PROMOTED).
- iter 12 nlms_bz2_v3: 4.31× (PROMOTED).
- iter 13 xlms_bz2_v1: 4.305× (not promoted — cross-lead 0× gain, graveyard).
- iter 14 rice_nlms_v1: 4.3113× (PROMOTED).
- iter 15 rice_nlms_v2: 4.3113× (not promoted — best-of-48 = best-of-30; LZMA and k=0/4 never selected).
- iter 16 lpc_bz2_v1: 4.3113× (not promoted — LPC on diff domain, 0× gain).
- iter 17 stage2_period_v1: 4.3113× (not promoted — period predictor on NLMS residuals, 0× gain).
- iter 18 cascade_nlms_v2: 4.3113× (not promoted — cascade NLMS incl. order-1/mu=0.50, 0× gain).
- iter 19 raw_nlms_v1: 4.3113× (not promoted — raw-domain NLMS orders 2/4/8/16, best-of-50, 0× gain).
- iter 21 psb_nlms_v1: 4.3113× (not promoted — phase-stable byte packing, best-of-36, 0× gain; never selected, nibble compacts shorter).
- iter 22 nibble4_nlms_v1: 4.3113× (not promoted — 4-bit fixed-width nibble packing, 0× gain; longer stream than nibble).
- iter 23 beat_tmpl_v1: 4.3113× (not promoted — beat template on raw diff, 0× gain; baseline dominates without NLMS).
- iter 24 beat_tmpl_v2: 4.3113× (not promoted — NLMS+beat template on residuals, 0× gain; onset overhead and bz2 already captures byte-level patterns).
- iter 25 badapt_rice_v1: 4.3113× (not promoted — block-adaptive Rice, 0× gain; bz2 still wins via cross-beat periodicity).
- iter 26 split_nlms_v1: 4.3133× (PROMOTED — split-tap NLMS at beat period breaks 11-consecutive-0× wall, +0.002×).

## GRAVEYARD
| id | lineage | primary | verdict | why it failed |
|---|---|---|---|---|
| fixed_pred_v1 | fixed-predictor | 3.7801 | gate pass, NOT promoted | Order-2 reduces sample-to-sample std but LZMA compresses order-2 residuals WORSE (388 KB vs 382 KB/channel): higher-order differences destroy the repeating heartbeat pattern that LZMA's dictionary matching exploits. |
| delta_bz2_v1 | delta-backend | 3.8771 | gate pass, NOT promoted | bz2 backend worse than LZMA when paired with delta (3.88 vs 3.90): delta preprocessing changes structure to favor LZMA's sliding-window over bz2's BWT; train/215 probe suggesting bz2 was better didn't generalize. |
| period_pred_v1 | period-predictor | 2.8911 | gate pass, NOT promoted | MIT-BIH is the arrhythmia database — irregular beat timing and morphology (PVCs, AF) mean s[i-T] is usually the WRONG beat to predict from; beat-to-beat residuals are large and aperiodic; LZMA loses the pattern matching it needs. |
| byte_split_v1 | byte-split | 3.2783 | gate pass, NOT promoted | hi bytes forced to explicit coding (~152KB binary entropy per record) vs FREE in LZMA's 770-byte match; total overhead swamps lo-plane benefit. |
| fixpred2_varzz_bz2_v1 | fixed2-varzz-bz2 | 3.9248 | gate pass, NOT promoted | Second-order predictor + varzz + bz2: T/P-wave reversals create mid-range zigzag values that vary beat-to-beat, spreading bz2's byte distribution and reducing BWT period matching below order-1 delta. |
| xlms_bz2_v1 | cross-lead | 4.305 | gate pass, NOT promoted | Joint NLMS with ch0_delta as extra tap for ch1: best-of-16 never selected any cross-lead option; NLMS cross-lead weight converges to ~0 for all holdout records. Delta-domain cross-lead correlation is essentially zero in MIT-BIH; co-temporal QRS events in MLII vs precordial leads do not produce exploitable linear delta correlation. |
| lpc_bz2_v1 | lpc-bz2 | 4.3113 | gate pass, NOT promoted | Per-512-sample Levinson-Durbin LPC on diff stream, best-of-45: 0× vs champion. MMSE-optimal per stationary block, but ECG is non-stationary within each ~1.4-beat window; coefficient table overhead (n_segs × lpc_order int16s) negates residual savings; bz2 still wins all channels. |
| stage2_period_v1 | stage2-period | 4.3113 | gate pass, NOT promoted | NLMS then OLS period predictor r2=r1-w*r1[n-T] on NLMS residuals, best-of-60: 0× vs champion. After NLMS the heartbeat-lag correlation in residuals is small; period stage never selected for any holdout channel. The high lag-1 autocorrelation (0.807) in NLMS residuals is NOT a lag-T phenomenon. |
| cascade_nlms_v2 | cascade-nlms | 4.3113 | gate pass, NOT promoted | Stage-2 NLMS (order=1/2/4 + multiple mu) on stage-1 residuals, best-of-65 including order-1/mu=0.50 targeting lag-1=0.807: 0× vs champion. bz2 BWT already exploits the lag-1 autocorrelation implicitly in the packed byte stream; removing it at integer level (cascade NLMS) doesn't reduce bz2's compressed output. |
| raw_nlms_v1 | raw-nlms | 4.3113 | gate pass, NOT promoted | NLMS AR predictor on raw int16 signal (not first-differences), orders 2/4/8/16, best-of-50: 0× vs champion. Best-of-50 always selected a diff-domain option; raw-domain residuals not better than diff-domain NLMS on any holdout channel. |
| psb_nlms_v1 | psb-nlms | 4.3113 | gate pass, NOT promoted | Phase-stable 1-byte/sample packing (rare zzag≥255 escaped to uint16 side stream), best-of-36: 0× vs champion. Never selected for any holdout channel — psb's fixed 650K main stream is longer than nibble's ~400-500K compacted stream, and bz2 rewards the shorter input more than the consistent byte phase. Entropy-layer reshape that doesn't shrink the stream can't beat bz2. |
| nibble4_nlms_v1 | nibble4-nlms | 4.3113 | gate pass, NOT promoted | 4-bit fixed-width nibble (zzag<15 direct, zzag≥15 escape + overflow stream), best-of-36: 0× (iter 22). Main stream is phase-consistent but at 325K bytes is longer than nibble's ~400K; overflow adds ~39KB bz2 overhead; net result longer than nibble+bz2. Phase consistency without stream length reduction cannot beat bz2. |
| beat_tmpl_v1 | beat-template | 4.3113 | gate pass, NOT promoted | Onset-aligned template subtraction on raw first-differences, best-of-55: 0× (iter 23). Without NLMS, baseline residuals dominate; template only helps QRS but baseline cost overwhelms the QRS savings. |
| beat_tmpl_v2 | beat-template | 4.3113 | gate pass, NOT promoted | NLMS+onset-aligned template subtraction of QRS patterns from NLMS residuals, best-of-55: 0× (iter 24). Template options never selected for any holdout channel. Onset list overhead (~3.4KB/channel) eats the small template gain. bz2 already captures the cross-beat QRS byte patterns implicitly; integer-level template removal adds no further compression. |
| badapt_rice_v1 | badapt-rice | 4.3113 | gate pass, NOT promoted | Block-adaptive Golomb-Rice (optimal k per 256-sample block), best-of-36: 0× (iter 25). BAR never selected — it can only improve on fixed-k Rice, which already loses to bz2 on most channels; per-block k cannot overcome bz2's cross-beat periodicity advantage. |
| seed | delta-backend | 3.8959 | promoted | delta + generic backend only ties bzip2 (3.85×) — a generic backend wastes the s |
| fixed_pred_v1 | fixed-predictor | 3.7801 | not promoted | LZMA gains from ECG's PERIODIC structure (dictionary matches repeat heartbeats), |
| delta_bz2_v1 | delta-backend | 3.8771 | not promoted | delta preprocessing restructures the byte stream to favor LZMA's LZ77 dictionary |
| period_pred_v1 | period-predictor | 2.8911 | not promoted | MIT-BIH is the ARRHYTHMIA database — irregular beat timing (PVCs, AF, compensato |
| byte_split_v1 | byte-split | 3.2783 | not promoted | LZMA's 770-byte heartbeat match token covers BOTH lo and hi bytes for free. Sepa |

## ACTIVE LOG
(append iteration blocks below — newest last; format in PROMPT.md §5)

### [varzz_bz2_v1] zigzag-delta variable-length pack + bz2 — 2026-06-19
lineage: varzz-bz2   parent: byte_split_v1 (probe)
hypothesis: zigzag-encode per-channel int16 wraparound deltas (0→0, -1→1, 1→2, ...) then variable-length pack: 1 byte if zzag<128 (99.57% of ECG deltas), 2 bytes up to 32639, 3-byte sentinel for full uint16 range. Concatenate all channels, compress with bz2-9. Stream shrinks from 2.6 MB (int16) to ~1.3 MB; bz2 BWT then exploits the ~385-byte heartbeat period at half the original match distance.
verdict: gate_passed=True  primary ratio_holdout=4.0907  ratio_train=4.0098  delta_vs_champion=+0.1948  PROMOTED TO CHAMPION
WHY: Halving the pre-compression stream size doubles the density of ECG period patterns relative to bz2's BWT block, letting bz2 find and exploit more heartbeat-period matches. Probe prediction of ~4.09× was exact. No overfitting (holdout 4.09× ≈ train 4.01×).
next: try adaptive LMS predictor (varzz-bz2 backend) — smaller residuals should push past 4.3× toward the literature ceiling. Alternatively, try rice-coder lineage on varzz residuals.
ITERATION DONE: varzz_bz2_v1 gate pass primary=4.0907 (champion: varzz_bz2_v1 4.0907)

### [fixpred2_varzz_bz2_v1] second-order fixed predictor + varzz + bz2 — 2026-06-19
lineage: fixed2-varzz-bz2   parent: varzz_bz2_v1
hypothesis: order-2 residuals have std≈6 vs delta std≈12; more 1-byte varzz tokens → shorter stream → denser bz2 BWT period matching. Revival of fixed-predictor idea with varzz+bz2 backend (fixed_pred_v1 failed with LZMA).
verdict: gate_passed=True  primary ratio_holdout=3.9248  compressed_kb_holdout=3234.7  ratio_train=3.8436  delta_vs_champion=-0.1659  NOT promoted
WHY: The second-order predictor amplifies T/P-wave reversals. Unlike QRS (stereotyped, same residual pattern per beat), T/P-wave shapes vary beat-to-beat. Order-2 residuals during T/P reversals fall in the 20-50 zigzag range (still 1-byte, but mid-range), spreading the byte distribution and making heartbeat patterns LESS repetitive for bz2's BWT. The gain on smooth segments is outweighed by this distribution spreading effect. First-order delta is the sweet spot: QRS is stereotyped and T/P deltas are tiny → maximally periodic, concentrated byte stream for bz2.
next: try per-channel separate bz2 — in combined stream (1.3MB), ch1 is split across bz2 blocks 1 and 2 (900KB block size), giving BWT only 649 periods in block 1 vs full 1688 if ch1 had its own block.
ITERATION DONE: fixpred2_varzz_bz2_v1 gate pass primary=3.9248 (champion: varzz_bz2_v1 4.0907)

### [perchan_varzz_bz2_v1] per-channel bz2 on varzz streams — 2026-06-19
lineage: perchan-varzz-bz2   parent: varzz_bz2_v1
hypothesis: combined 1.3MB stream spans ~1.45 bz2 blocks (900KB); ch0 fits in block 1 but ch1 is split (649 periods in block 1, 1039 in block 2 — separate BWT transforms). Per-channel: each 650KB stream in own 900KB block → 1688 heartbeat periods fully visible to BWT for both channels.
verdict: gate_passed=True  primary ratio_holdout=4.1043  compressed_kb_holdout=3093.2  ratio_train=4.023  delta_vs_champion=+0.0136  PROMOTED TO CHAMPION
WHY: Per-channel bz2 eliminates the BWT block split that was costing ch1 ~0.013× compression. Small but reproducible gain (+0.33% holdout). The per-channel approach gives each lead's periodic heartbeat stream a clean, uninterrupted BWT, confirming the 900KB block boundary was a real bottleneck.
next: try interleaving channel deltas before varzz (sample-interleaved: d_ch0[0], d_ch1[0], d_ch0[1], ...) and compressing as single bz2 stream — QRS events in both channels co-occur, potentially creating stronger joint period patterns for bz2. Also: investigate whether 2-byte token positions carry structure exploitable by a secondary coder.
ITERATION DONE: perchan_varzz_bz2_v1 gate pass primary=4.1043 (champion: perchan_varzz_bz2_v1 4.1043)

### [nibble_adapt_bz2_v1] per-channel nibble-pair vs varzz best-of-two + bz2 — 2026-06-19
lineage: nibble-adapt-bz2   parent: perchan_varzz_bz2_v1
hypothesis: 81.97% of ECG delta zigzag values have zzag<8 (|delta|<4). Pack CONSECUTIVE PAIRS where both zzag<8 into ONE byte (3 bits + 3 bits, range 0x00-0x3F), shrinking the stream by 35-44% for high-small% records. Single zzag<128 → 0x40+z (range 0x40-0xBF). Large values use 2- or 3-byte escapes. Shorter stream → more heartbeat periods per bz2 BWT block (~2220 vs 1688) → better BWT match exploitation. BUT: for records with low small% (e.g. arrhythmia record 215, 59.6% small), the greedy pair boundary shifts with each adjacent value, creating inconsistent byte offsets at heartbeat positions across beats → BWT alignment disrupted → bz2 gets WORSE. Solution: per-channel best-of-two — encode with both nibble and varzz, bz2 both, store the smaller with a 1-byte flag. This guarantees no regression vs champion.
probe: train/101 (82% small): nibble saves 7,173B (ch0) + 7,305B (ch1) = 14,478B vs champ. Train/215 (60% small): nibble is 8,962B WORSE — varzz wins both channels. Best-of-two: train total saves 49,631B vs champion (vs 40,669B with pure nibble). Projected holdout improvement: +0.05 to +0.10×.
verdict: gate_passed=True  primary ratio_holdout=4.1586  compressed_kb_holdout=3052.8  ratio_train=4.0676  delta_vs_champion=+0.0543  PROMOTED TO CHAMPION
WHY: Nibble-pair halves the encoding of the most common delta values, shrinking the bz2 input by ~35% for records with 75%+ small deltas. bz2 then sees ~2220 heartbeat periods per block vs 1688 with varzz → more BWT matches → better compression. The best-of-two fallback correctly identified 215 (and any similar holdout record) as needing varzz, preventing regression while capturing gains on cleaner records.
KEY LESSON: nibble-pair alignment shifts hurt BWT when small% < ~70%. The greedy pair boundary makes byte offsets at any given heartbeat sample position depend on the preceding sample's value — if that preceding value is sometimes large (>8), the pair doesn't form and everything after shifts by 1 byte. Cross-beat inconsistency kills BWT matching.
next: explore other stream-compaction approaches: (a) pair-boundary-stable encoding (e.g., even-indexed positions always paired vs always single, regardless of value — stable alignment at cost of non-optimal token packing), (b) nibble-pair with a stronger grouping heuristic (e.g., encode in 4-sample windows, select pack scheme per window globally), (c) adaptive-predictor lineage (LMS on delta stream, smaller residuals, then nibble+bz2). Remaining gap to literature ceiling (~4.5-5×): ~0.34-0.84×.
ITERATION DONE: nibble_adapt_bz2_v1 gate pass primary=4.1586 (champion: nibble_adapt_bz2_v1 4.1586)

### [nlms_bz2_v1] NLMS-4 adaptive prediction + best-of-four + bz2 — 2026-06-19
lineage: nlms-bz2   parent: nibble_adapt_bz2_v1
hypothesis: NLMS order=4 mu=0.1 prediction on per-channel first-differences; quantised residuals fed to nibble/varzz+bz2. Best-of-four (raw_vz, raw_nb, nlms_vz, nlms_nb) guarantees no regression. NLMS converges to local ECG statistics and reduces residuals for irregular-rhythm records (MIT-BIH arrhythmia records 108, 230, 207) where global AR coefficients fail. Island rule satisfied: nlms-bz2 is a new lineage distinct from the last three promotions (varzz-bz2, perchan-varzz-bz2, nibble-adapt-bz2).
probe (training, bz2 output): 108.ecg -0.86% (-5.7KB), 230.ecg -0.29% (-1.7KB), 105.ecg -0.02%, all others 0.00%. Total training: -0.17% (-7.5KB). nlms_nb wins for 108/230; raw_nb wins for regular records (101/103/209/215). Local holdout estimate: 4.1736× (+0.015).
losslessness mechanism: weight updates use the QUANTISED residual (not float error), so encoder and decoder weight states stay byte-identical. Prediction is rounded to int, residual stored as int16 modulo-65536 → exact recovery guaranteed.
verdict: gate_passed=True  primary ratio_holdout=4.1736  compressed_kb_holdout=3041.8  ratio_train=4.0739  delta_vs_champion=+0.015  PROMOTED TO CHAMPION
per-record holdout: 100.ecg 0.00%, 119.ecg 0.00%, 207.ecg -1.58% (-9.8KB), 214.ecg -0.04%, 232.ecg -0.14%
WHY: Record 207 (left bundle branch block) has very wide QRS complexes and irregular morphology. NLMS-4 adapts to local delta statistics within each morphology cluster, shrinking residual magnitude where raw delta is less stereotyped. bz2 then exploits the residual's tighter distribution for better BWT match density. For regular-rhythm records (100, 119), raw delta is already maximally periodic → NLMS adds noise → best-of-four correctly selects raw_nb.
next: (a) tune mu and order for the irregular-record subpopulation (current mu=0.1 order=4 may not be optimal); (b) try cross-lead NLMS (predict ch1 from ch1 history + ch0 context) for correlated-lead records; (c) explore per-segment mu adaptation (higher mu during QRS, lower during baseline).
ITERATION DONE: nlms_bz2_v1 gate pass primary=4.1736 (champion: nlms_bz2_v1 4.1736)

### [nlms_bz2_v2] best-of-8 NLMS hyperparameter sweep — 2026-06-19
lineage: nlms-bz2   parent: nlms_bz2_v1
hypothesis: expand champion's best-of-4 to best-of-8 by adding NLMS order=2 mu=0.20 (fast-adapting, short memory) and order=8 mu=0.05 (slow-adapting, long memory) alongside existing order=4 mu=0.10 and raw options; each NLMS config paired with both varzz and nibble packing; wider hyperparameter space should find better predictors for holdout records with varied morphology.
verdict: gate_passed=True  primary ratio_holdout=4.2845  compressed_kb_holdout=2963.1  ratio_train=4.1546  delta_vs_champion=+0.1109  PROMOTED TO CHAMPION
WHY: The single NLMS config in the champion (order=4, mu=0.10) was suboptimal for multiple holdout records. Adding order=2 (fast convergence due to small tap count) and order=8 (longer memory, sees ~22ms of QRS context at 360 Hz) let each channel independently select the best predictor. The +0.11× gain is 7× larger than the previous NLMS iteration's +0.015×, confirming that hyperparameter choice matters greatly per record — different ECG morphologies benefit from different adaptation speeds.
next: (a) expand further: try order=16, order=32 (covers full QRS at 360Hz), or half-integer mu values; (b) per-segment mu adaptation — higher mu during detected QRS onset (large |delta| bursts), lower during baseline; (c) explore NLMS on raw signal instead of first-difference (could capture longer-range dependencies like baseline drift); (d) try dormant rice-coder lineage now that residuals are smaller.
ITERATION DONE: nlms_bz2_v2 gate pass primary=4.2845 (champion: nlms_bz2_v2 4.2845)

### [nlms_bz2_v3] best-of-12: long-memory NLMS taps (order 16/32) — 2026-06-19
lineage: nlms-bz2   parent: nlms_bz2_v2
hypothesis: a QRS spans ~30-36 samples at 360 Hz, but the champion's NLMS orders (2/4/8) only see ~6-22 ms of context. Add order=16 mu=0.05 and order=32 mu=0.03 (best-of-12) so a longer filter on the first-difference can model broad wide-QRS deflections (record 207 LBBB, the record NLMS already helped most). Best-of-N is monotonic, so the new configs can only help.
verdict: gate_passed=True  primary ratio_holdout=4.305  compressed_kb_holdout=2949.0  ratio_train=4.174  delta_vs_champion=+0.0205  PROMOTED TO CHAMPION
WHY: Long-memory taps capture a fraction of the broad-deflection structure the short filters miss, but the gain (+0.02×) is ~5× smaller than v2's grid-widening (+0.11×) — diminishing returns. Most holdout records still select raw or a short NLMS order; the long taps win only on the wide-QRS minority. The NLMS-grid lever is now largely exhausted.
next: ISLAND RULE NOW BINDING — nlms-bz2 produced the last 3 promotions (iter 10/11/12). Next iteration MUST advance a different/dormant lineage: (a) cross-lead — predict ch1's first-difference from ch0's co-temporal diff (QRS co-occurs across leads) as an added best-of-N option, decode ch0 first so its diff is available (regression-proof); (b) rice-coder — vectorized Golomb–Rice on the NLMS residual as an alternative backend to bz2 per channel.
ITERATION DONE: nlms_bz2_v3 gate pass primary=4.305 (champion: nlms_bz2_v3 4.305)

### [xlms_bz2_v1] cross-lead joint NLMS for ch1 (island rule) — 2026-06-19
lineage: cross-lead   parent: nlms_bz2_v3
hypothesis: add ch0_delta[i] as one extra tap in a joint (order+1)-tap NLMS filter for ch1, alongside k within-channel taps. Unlike the prior failed pure-cross-lead predictor (std=72 vs within-channel std=4), the NLMS weight on the cross-lead tap converges to ~0 if ch0 is uninformative, and to the MMSE coefficient otherwise. Key mechanism: at QRS onset in ch1, within-channel NLMS context (recent small baseline deltas) is uninformative; ch0_delta[i] is large at the same sample (co-temporal QRS), providing a leading indicator. Regression-proof via best-of-16 vs champion's best-of-12. Satisfies island rule (cross-lead lineage, different from last 3 nlms-bz2 promotions).
verdict: gate_passed=True  primary ratio_holdout=4.305  compressed_kb_holdout=2949.0  ratio_train=4.1747  delta_vs_champion=+0.000  NOT promoted
WHY: The best-of-16 never selected any cross-lead option for any holdout or train channel. The NLMS cross-lead weight converges to ~0 for all records. MIT-BIH ECG lead deltas (MLII vs precordial) have essentially zero exploitable linear correlation in the delta domain — co-temporal QRS events produce large deltas in both channels, but their morphological difference (magnitude, polarity variation) prevents any stable MMSE linear cross-lead coefficient from forming. Cross-lead lineage is fully exhausted.
next: rice-coder — the only remaining dormant lineage. Vectorized Golomb–Rice on NLMS residuals as an alternative entropy backend to bz2; after best-of-12 NLMS the residuals are small (~std 4) and nearly independent; Rice may approach or surpass bz2 if NLMS has removed most of the heartbeat periodicity that bz2's BWT exploits.
ITERATION DONE: xlms_bz2_v1 gate pass primary=4.305 (champion: nlms_bz2_v3 4.305)

### [rice_nlms_v1] Golomb-Rice backend alongside bz2 for all NLMS configs — 2026-06-19
lineage: rice-nlms   parent: nlms_bz2_v3
hypothesis: after NLMS removes periodic structure from residuals, bz2's BWT finds less to exploit; Rice-k (k=1/2/3) achieves near-entropy coding with negligible overhead, potentially beating bz2 on irregular-rhythm channels. Best-of-30 = 12 bz2 options + 18 Rice options per channel.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta_vs_champion=+0.0063  PROMOTED TO CHAMPION
WHY: For channels where NLMS has reduced residuals to near-iid, bz2's BWT overhead is net-negative (no exploitable structure). Rice-k's pure entropy coding wins by ~4-byte overhead vs bz2's block headers. Gain is modest (+0.006×) because most channels still have residual periodicity that bz2 exploits well; only a subset of channels on irregular-rhythm records switch to Rice.
next: try Rice k=0 (pure unary, optimal for distributions heavily concentrated near 0) and k=4 (longer remainder, better for slightly heavier tails) as additional options. Alternatively explore whether LZMA — which was better than bz2 for raw delta — is also better for NLMS residuals (different residual structure might favor LZMA's sliding window).
ITERATION DONE: rice_nlms_v1 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [rice_nlms_v2] LZMA backend + Rice k=0,4 extension — 2026-06-19
lineage: rice-nlms   parent: rice_nlms_v1
hypothesis: best-of-48 = 6 predictor configs x 8 options: add varzz+LZMA (preset=4) as third block backend alongside bz2, extend Rice k to {0,1,2,3,4}. After NLMS removes heartbeat periodicity, LZMA's LZ77 may beat bz2 BWT on quasi-iid residuals; k=0 covers near-zero mean residuals, k=4 covers heavier tails.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.202  delta_vs_champion=+0.000  not promoted
WHY: bz2 BWT wins over LZMA on NLMS residuals for ALL holdout channels — confirms that even after NLMS, the residuals retain enough block-level periodic structure for BWT to exploit. Rice k=0 never wins (mean abs residual too high even after NLMS; k=1 already covers the optimum). Rice k=4 also never wins (k=3 is sufficient for heavier tails). The rice-nlms lineage's encoding grid is now fully exhausted: best-of-48 = best-of-30 on holdout.
next: residuals still have exploitable periodic structure (bz2 wins). Two remaining unexplored levers: (a) per-segment LPC (autocorrelation + Levinson-Durbin per 512-sample block) as an alternative to NLMS — LPC is the MMSE-optimal predictor per block, while NLMS approaches it gradually; (b) two-stage cascade predictor (first NLMS pass, second short-order NLMS on the residuals). The fact that bz2 still outperforms Rice means there is still remaining periodic structure — a better predictor should reduce it.
ITERATION DONE: rice_nlms_v2 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [lpc_bz2_v1] per-segment Levinson-Durbin LPC on diff stream — 2026-06-19
lineage: lpc-bz2   parent: rice_nlms_v1
hypothesis: block-wise LPC (Levinson-Durbin, MMSE-optimal per 512-sample segment) applied to diff stream; vectorized forward pass; best-of-45 = champion 30 + LPC orders 4/8/16 × 5 pack types.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta_vs_champion=+0.000  not promoted
WHY: LPC is MMSE-optimal for STATIONARY AR processes but ECG is non-stationary: each 512-sample block (~1.4 beats) contains baseline, P-wave, QRS, T-wave — mixed morphologies. Coefficients fitted to a mixed block cannot predict QRS onset from baseline context. Coefficient overhead (n_segs × lpc_order int16s per channel) further eats into any savings. bz2 continues to win on all channels. Diff-domain linear prediction is exhausted.
next: cascade NLMS (stage-2 NLMS on stage-1 NLMS residuals) to remove the lag-1 autocorrelation=0.807 identified in NLMS-8 residuals.
ITERATION DONE: lpc_bz2_v1 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [stage2_period_v1] two-stage NLMS then period predictor on residuals — 2026-06-19
lineage: stage2-period   parent: rice_nlms_v1
hypothesis: after champion NLMS, NLMS residuals have lag-1 autocorrelation=0.807 (high short-range) but the stage2_period_v1 targets the heartbeat-period lag; OLS period predictor r2[n]=r1[n]-w*r1[n-T] on residuals; best-of-60 = champion 30 × 2 (with/without period stage) × 5 packs.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta_vs_champion=+0.000  not promoted
WHY: After NLMS, heartbeat-period (lag-T) correlation in residuals is small — the irregular rhythms (arrhythmia DB) mean r1[n-T] predicts r1[n] poorly. The dominant residual autocorrelation is lag-1 (0.807), not lag-T. Period stage never selected by best-of-60 for any holdout channel. Confirms period lag is not the bottleneck.
next: cascade NLMS with order-1 stage-2 to target the lag-1 autocorrelation directly.
ITERATION DONE: stage2_period_v1 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [cascade_nlms_v2] cascade NLMS incl. order-1/mu=0.50 targeting lag-1=0.807 — 2026-06-19
lineage: cascade-nlms   parent: rice_nlms_v1
hypothesis: NLMS-8 residuals have lag-1 autocorrelation=0.807 (probed on train/101); stage-2 NLMS order=1 mu=0.50 models this AR(1) structure, converging to a[0]~0.807 and reducing residual variance by ~65%. Best-of-65 = champion 30 + 7 cascade pairs (v1's 4 pairs + 3 new with order-1 stage-2) × 5 packs. Regression-proof by construction.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta_vs_champion=+0.000  not promoted
WHY: Despite lag-1 autocorrelation=0.807 in NLMS residuals, cascade NLMS gave 0× gain. bz2's BWT already captures the lag-1 autocorrelation in the packed byte stream by grouping similar n-gram contexts; removing it at the integer level (stage-2 NLMS) does not further reduce bz2's compressed output. All diff-domain prediction improvements are now exhausted (LPC, cascade NLMS, stage-2 period). The bottleneck is the repeating QRS-burst residual SHAPE across beats, which bz2 exploits and which no causal diff-domain predictor can preemptively remove.
next: try NLMS on the RAW SIGNAL (not first differences). AR(2) on raw ECG gives near-zero residuals for smooth sine-like segments (s[i] ≈ 2cos(2π/T)s[i-1] - s[i-2] for a sine); residuals concentrate at QRS events only; MIT-BIH's irregular QRS timing → less cross-beat periodicity for bz2 to exploit → Rice may win. Requires format change (raw-domain channels, no cumsum).
ITERATION DONE: cascade_nlms_v2 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [raw_nlms_v1] NLMS on raw signal (not first-differences) — 2026-06-19
lineage: raw-nlms   parent: rice_nlms_v1
hypothesis: diff-domain NLMS residuals still carry lag-1 autocorrelation=0.807. Raw-domain AR(N) NLMS predicts smooth P/T-wave segments near-exactly (s[i] ≈ 2cos(w)s[i-1]-s[i-2] for a sine), concentrating residuals only at QRS spikes. Best-of-50 = champion 30 + 4 raw configs (orders 2/4/8/16) × 5 packs. Regression-proof.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  delta_vs_champion=+0.000  not promoted
WHY: Raw-domain AR predictor residuals are not better than diff-domain NLMS residuals for any holdout channel. The best-of-50 always selected a diff-domain option. The diff-domain is already a near-optimal predictor for smooth ECG segments (it reduces lag-1 correlation); raw-domain orders 2/4/8/16 add nothing new. The 4.3113× wall persists across all predictor architectures tried — diff-domain NLMS, LPC, cascade NLMS, period predictors, and now raw-domain NLMS.
next: evaluate existing vss_nlms_v1.py (variable step-size NLMS with activity-gated mu: mu_high during QRS, mu_low during baseline) — if it also stalls at 4.3113×, the path forward requires a different entropy coding layer (context-adaptive Rice conditioned on prev residual magnitude, or within-record bz2 block-size tuning).
ITERATION DONE: raw_nlms_v1 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [psb_nlms_v1] phase-stable byte packing (entropy-layer lever) — 2026-06-19
lineage: psb-nlms   parent: rice_nlms_v1
hypothesis: switch the lever from predictor to ENCODING. varzz/nibble use 2-byte escapes for the large values that occur AT QRS peaks, so the byte count before each QRS varies beat-to-beat and the same QRS morphology lands at different byte phases — bz2's BWT sees it as several distinct patterns (the nibble-alignment lesson). Phase-stable packing emits EXACTLY 1 byte/sample (zzag clipped 0..254) and routes rare zzag≥255 to a uint16 side stream via marker 255, so every QRS shape sits at a consistent byte phase. Added as best-of-36 (6 predictor configs × 6 packs), regression-proof.
verdict: id=psb_nlms_v1  iter=21  gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta vs champion rice_nlms_v1: +0.0  not promoted
WHY: psb was never selected for any holdout channel. Its main stream is exactly n bytes (650K), but nibble compacts the same channel to ~400-500K — bz2's gain from a shorter/denser input dominates the byte-phase benefit. The nibble-alignment lesson does NOT generalize to "fixed phase wins"; bz2 actually rewards a SHORTER stream, and per-channel best-of already buys phase consistency implicitly by choosing nibble only when small% is high enough for stable pairs. Sixth straight 0× — predictor families AND a packing reshape are now both exhausted at the wall.
next: a coder that wins must do BOTH — shrink the stream AND keep periodicity (e.g. a 4/6-bit fixed-width small-value plane that is shorter than nibble yet phase-consistent), OR replace bz2 with an adaptive range coder whose context conditions on recent residual magnitude (joint cross-beat-repetition + sub-byte entropy). Pure-reshape (psb) and pure-marginal-entropy (Rice) have both lost; the next attempt should target a denser-AND-periodic compaction, not another predictor.
ITERATION DONE: psb_nlms_v1 gate pass primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [nibble4_nlms_v1] 4-bit fixed-width nibble packing — 2026-06-19
lineage: nibble4   parent: rice_nlms_v1
hypothesis: every sample = exactly 4 bits. zzag 0-14 encoded directly (1 nibble); zzag≥15 uses marker 0xF in main stream + overflow to a separate varzz stream. Main stream = ceil(N/2) bytes — perfectly phase-consistent (heartbeat period 192.5 bytes). Overflow stream contains only large values at QRS peaks. Both streams bz2-compressed independently. Best-of-36 = champion 30 + nibble4 as best-of-6 new options (one per predictor config), regression-proof.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta vs champion rice_nlms_v1: +0.0  not promoted
WHY: nibble4 never selected for any holdout channel. The main stream is phase-consistent (no 2-byte escapes) but the overflow stream (~39KB of QRS residuals) adds bz2 overhead that erases the phase-alignment benefit. The main stream (650K samples → 325K bytes) is shorter than a raw varzz stream but LONGER than nibble (~400K bytes) because nibble efficiently codes the majority 90%+ small values. bz2 rewards the shorter/denser nibble stream over the fixed-width nibble4 main stream. Phase consistency alone is insufficient; the stream must also be shorter.
next: try beat-template subtraction to directly remove the cross-beat QRS structure that bz2 exploits.
ITERATION DONE: nibble4_nlms_v1 gate fail primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [beat_tmpl_v1] onset-aligned template subtraction on raw diff — 2026-06-20
lineage: beat-template   parent: rice_nlms_v1
hypothesis: explicit beat template subtraction on raw first-differences. Detect QRS onsets via |diff|>threshold with refractory period. For each beat, subtract the previous beat's template (90-sample window aligned to onset). Residuals are the beat-to-beat shape variation only. Onset positions stored as delta-encoded uint16 list. Best-of-55 = champion 30 + 25 template options (5 predictor configs × 5 pack types).
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta vs champion rice_nlms_v1: +0.0  not promoted
WHY: Template on raw diff gave no improvement. Without NLMS, the baseline residuals (P/T waves, baseline wander) are too large — same performance as the raw seed at ~3.90×. The template helps only QRS bursts but the baseline dominates the total compressed size. NLMS is still required for baseline compression.
next: apply beat template to NLMS residuals (not raw diff) so NLMS handles baseline and template handles cross-beat QRS shape.
ITERATION DONE: beat_tmpl_v1 gate fail primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [beat_tmpl_v2] onset-aligned template subtraction on NLMS residuals — 2026-06-20
lineage: beat-template   parent: rice_nlms_v1
hypothesis: NLMS+template: run NLMS (5 configs) then subtract the previous beat's template from NLMS residuals (90-sample window aligned to QRS onset). NLMS handles baseline; template removes the cross-beat QRS shape that bz2 currently exploits. Onset detection on NLMS residuals (|res|>20, refractory 100). Best-of-55 = champion 30 + 25 NLMS+template options.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta vs champion rice_nlms_v1: +0.0  not promoted
WHY: Even with NLMS+template, no improvement. The template options (flags 30-54) are NEVER selected for any holdout channel. Root cause: (1) onset list overhead (~3.4KB per channel for ~1688 onsets × 2 bytes) eats all potential gains from template residuals; (2) after NLMS-32, the QRS residuals may already be near-zero (NLMS-32 spans the full QRS width), so template subtraction gives diminishing returns; (3) most crucially: bz2's BWT already exploits the cross-beat QRS pattern at the byte level without the template explicitly removing it at the integer level. The template reduces integer values but bz2 ALREADY handles the resulting byte patterns efficiently. Nine consecutive 0× iterations spanning every predictor family and encoding reshape.
next: try block-adaptive Rice (k chosen per 256-sample block rather than globally) — baseline blocks get k=0 (1.5 bits/sample), QRS blocks get k=3.
ITERATION DONE: beat_tmpl_v2 gate fail primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [badapt_rice_v1] block-adaptive Golomb-Rice (B=256) — 2026-06-20
lineage: badapt-rice   parent: rice_nlms_v1
hypothesis: fixed-k Rice wastes bits on baseline blocks (k=2 → 3.1 bits/sample) vs optimal k=0 (1.5 bits/sample). Per-256-sample block: pick k = argmin total bits. Saves ~1.5 bits/sample on 55% baseline = 71KB/channel. Best-of-36 = champion 30 + 6 BAR options (one per predictor config). Block overhead: ~6.4KB/channel.
verdict: gate_passed=True  primary ratio_holdout=4.3113  compressed_kb_holdout=2944.7  ratio_train=4.198  delta vs champion rice_nlms_v1: +0.0  not promoted
WHY: BAR never selected for any holdout channel. Block-adaptive Rice can only improve on fixed-k Rice, which ALREADY loses to bz2 on most channels. bz2 exploits cross-beat QRS periodicity (same QRS byte pattern at ~192-byte intervals) — Rice codes each sample independently regardless of k. Even with optimal per-block k, Rice cannot close the gap where bz2 has structural advantage from the periodic QRS byte patterns. The fixed Rice options in the champion only win on ~1-2 channels where NLMS has nearly eliminated periodicity. For all other channels, bz2's cross-beat exploitation dominates and per-block k adjustment cannot overcome it. Ten consecutive 0× iterations.
next: split-tap NLMS — short taps (lags 1-16) + long taps at estimated beat period T (lags T to T+15). Cross-beat taps allow the filter to predict current QRS from previous beat's QRS morphology, reducing QRS residuals from zzag~8 to zzag~2 on regular rhythms adaptively. No onset detection needed.
ITERATION DONE: badapt_rice_v1 gate fail primary=4.3113 (champion: rice_nlms_v1 4.3113)

### [split_nlms_v1] split-tap NLMS at estimated beat period — 2026-06-20
lineage: split-nlms   parent: rice_nlms_v1
hypothesis: standard NLMS-32 uses consecutive taps at lags 1-32 and cannot see the previous beat's QRS morphology (at lag ~T ≈ 200-500 samples). A split filter with short taps at lags 1-16 (local trend prediction) AND long taps at lags T to T+15 (cross-beat QRS prediction) allows NLMS to predict the current QRS from the previous beat's shape. T is estimated per channel via FFT autocorrelation of diff^2, stored as uint16 in channel data. 3 split configs (16+16 taps, mu 0.03/0.01; 8+16 taps, mu 0.03). Best-of-45 = champion 30 + 15 split options (3 configs × 5 pack types). Regression-proof.
verdict: gate_passed=True  primary ratio_holdout=4.3133  compressed_kb_holdout=2943.3  ratio_train=?  delta vs champion rice_nlms_v1: +0.002  PROMOTED TO CHAMPION
WHY: Cross-beat taps at the estimated heartbeat period reduce QRS residuals at the INTEGER level — something all encoder-layer and same-lag predictors (iters 15–25) could not do. The split filter allows NLMS to predict "the current QRS will look like the previous beat's QRS" and converge to the beat-to-beat residual. This reduces the remaining periodicity that bz2 was exploiting. The gain is modest (+0.002×) because: (1) the 3 split configs in best-of-45 may not cover the optimal (short_order, long_order, mu) for every holdout record's heart rate; (2) irregular-rhythm records (207, LBBB) have variable T so the long taps learn weights ≈ 0 and add no benefit. Gain comes from regular-rhythm records where T is stable.
next: widen the split config space — try more period estimates (search both primary and secondary autocorrelation peaks), tighter long-tap centering (lags T-8 to T+7 vs T to T+15), more (short_order, long_order, mu) combinations in a larger best-of-N.
ITERATION DONE: split_nlms_v1 gate pass primary=4.3133 (champion: split_nlms_v1 4.3133)
