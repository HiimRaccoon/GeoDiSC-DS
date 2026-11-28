# GeoLocation Geo-DiSC-DS Implementation Checklist

**Status:** Code-first implementation is underway. T01–T11 and local testing are separate from UIT deployment verification (V01–V13) and scientific execution (R/F items). Deployment verification is authorized after implementation; full scientific execution requires separate user approval.

**Sources:** [Master Markdown](GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), [Common v1.3](specs/00_common.md), [E0 v1.1](specs/01_E0_original.md), [E1E2 v1.2](specs/02_E1E2_hybrid_baseline.md), [E3 v1.1](specs/03_E3_geodisc_ds.md), [Evaluation v1.1](specs/04_evaluation.md), [Implementation Plan](plan.md). Follow their precedence and confirmed decisions; this checklist adds no research policy.

Task IDs T01–T11 map directly to plan Tasks 1–11. V IDs track outstanding deployment/runtime evidence; R IDs track scientific execution. Implementation dependencies may use verified fixtures or mocks; scientific execution dependencies require actual verified outputs, not merely successful scheduler exits. Gate IDs group evidence and do not impose a global barrier on independent stages. All paths below are planned targets, not existing implementation claims.

## P0 — Contracts, data and metrics

### T01 — Bootstrap, configuration and artifact contracts

**Depends on:** Source documents above. **Targets:** `pyproject.toml`, `configs/{benchmark,methods,uit}.json`, `src/geobench/{__init__,__main__,cli,config,artifacts}.py`, `tests/test_config_artifacts.py`.

- [x] T01.1 — Add failing fixtures for strict confirmed-config validation, canonical identities, dependency hashes, root containment, partial publication and competing writer claims; run `pytest tests/test_config_artifacts.py -q`.
- [x] T01.2 — Implement the plan's minimal package/CLI and config/artifact interfaces; separate research configuration from unresolved deployment inputs. Reject silent defaults and research overrides.
- [x] T01.3 — Implement staging → atomic publication → verified lock, persistent ownership and mismatch rejection. Never repair a mismatch by refreshing hashes or reclaim a remote writer merely because its PID is absent locally.
- [x] T01.4 — Re-run the targeted tests. **Acceptance:** no incomplete artifact can appear frozen; changed dependencies invalidate consumers; single-writer claims are enforced. Evidence: 23 passed, including review-driven publication and containment regressions.

### T02 — Cleaned inventory, image geometry and fixed split

**Depends on:** T01; actual publication additionally V01/V02/V04. **Targets:** `src/geobench/{data,images}.py`, `tests/test_data_images.py`; output `splits/` and E0 real-pool artifacts under the new root.

- [x] T02.1 — Test split boundary counts, including singleton, 2, 3, 9, 10 and 15 originals; test feasible duplicate components, infeasible quotas and cross-label duplicates using both file SHA-256 and decoded RGB hashes. Evidence: `tests/test_data_images.py`, 24 passed.
- [x] T02.2 — Implement inventory/class IDs, exact-duplicate audit, deterministic component allocation, support/group manifests and E0 real membership per Common §§3–4 and E0 §§2–3. Do not clean the dataset again or import old splits.
- [x] T02.3 — Implement shared decoding/hash semantics and image interfaces. Test center letterbox output `[3,384,896]`, normalization/padding, native reflection-pad/crop round trips and metadata-insensitive RGB identity; do not change DINO preprocessing. Evidence: `tests/test_data_images.py`, 24 passed.
- [x] T02.4 — Run `pytest tests/test_data_images.py -q`. **Acceptance:** quotas and train/validation/test isolation hold; conflicts block freeze; all later methods reference the same E0 real pool. Evidence: 26 passed using fixture evidence; actual dataset remains V04.

### T03 — Support-aware metrics, ranking and formatting

**Depends on:** T01. **Targets:** `src/geobench/metrics.py`, `tests/test_metrics.py`.

- [x] T03.1 — Add fixtures for supported/unsupported classes, empty groups, full-split confusion counts before group averaging, all-equal logits yielding Top-5 IDs `[0,1,2,3,4]`, and missing-seed rejection.
- [x] T03.2 — Implement only Evaluation §§3–5 metrics: exact descending-logit/ascending-ID ranking, support-aware N/A handling, three-seed mean/sample std and percentage display with two decimals while retaining full-precision raw values.
- [x] T03.3 — Run `pytest tests/test_metrics.py -q`. **Acceptance:** hand-computed fixtures match; seeds `[0.1,0.2,0.3]` render `20.00 ± 10.00%`; no pooling predictions across seeds or masking unsupported output classes. Evidence: 8 passed using Python 3.13, PyTorch 2.8.0 CPU and pytest 9.2.0.

- [x] G-P0 — Close P0 after T01–T03 fixture tests pass and contract review confirms the Common/E0/Evaluation acceptance requirements are represented. Evidence: 72 tests passed in the full suite; independent review findings were addressed. Actual deployment and dataset verification remain V01/V02/V04, not fixture claims.

## P0 review follow-up — 2026-11-27

- [x] Enforce OS-backed active writer leases across claim/resume, staging lifetime and publication; same-token subprocess/thread contention, required recoverable credentials, and abrupt-process-exit recovery have regression coverage.
- [x] Validate explicit/default CLI configurations before data access and bind canonical verified configuration hashes to `splits/configuration.json` under the data lock.
- [x] Validate official condition/seed descriptors, exact dependency roles, lock identities, hashes and upstream lineage; A4 binds the canonical A3 pool.
- [x] Provide `publish_readiness` to import explicit V01/V02 observations and V04 inventory/audit evidence through the same fixture-testable producer contract.
- [x] Reject internal staging locks in authoritative verification; enforce canonical artifact paths and E0 `pool.json`/`pool_lock.json` schemas.
- [x] Preserve current master, specifications, plan and checklist in Git with private deployment identifiers redacted from public copies. Full local originals remain available.

Evidence: full regression suite **72 passed** on 2026-11-27. Metrics/research configuration values are unchanged. Actual UIT V01/V02/V04 evidence remains outstanding; no P1 work or Slurm submission was performed.

## Final targeted P0 fixes — F06–F09

- [x] F06 — Authenticated recovery completes matching prefixes of interrupted staged files, publication copies and temporary locks. Unexpected temporary content and mismatched published artifacts remain unchanged and block publication.
- [x] F07 — File-based CLI inputs and programmatic snapshots share the same strict role/schema/type/research-value validation and canonical configuration-hash checks.
- [x] F08 — Public source and fixtures contain no fixed private UIT host, login or workspace identifiers. Readiness binds a separately verified private-site profile and its supporting evidence by lock dependency.
- [x] F09 — The shared artifact contract publishes and verifies all authoritative lock names required by the confirmed specifications; each is covered by a dependency-consumption regression.

Evidence: full suite **89 passed**, including 17 focused F06–F09 cases and interrupted data/E0 staging recovery. Research configuration files, metrics and working research behavior are unchanged. Actual UIT verification remains outstanding. No P1 implementation, Slurm jobs or history rewrite.

## P1 progress and recovery issue

- P1 remains incomplete: initial profile validation, deterministic job seed/source ordering, and common learning-rate schedule primitives have fixture tests; T04–T06 and G-P1 remain unchecked.
- Read-only SSH reached the configured login alias on 2026-11-27; environment and home filesystem were inspected. This does not close V01–V06 or verify project paths, model access, quotas, compute-node access or GPU workloads.
- **T06/T10 recovery hardening:** new writer initialization persists an identity/token intent before creating the claim directory. Matching owner credentials can finish interrupted claim bytes and staging initialization under the OS lease; a different token is rejected. Legacy incomplete claims without ownership evidence remain blocked. Regression: `test_claim_initialization.py`; no PID-based reclamation is used.

## P1 — Runtime, generation and shared training

### T04 progress — fixture interfaces only

- Added the `preflight` CLI and immutable pending inspection report, local interpreter/package and storage observation helpers, pinned snapshot hash/inventory checks, role-scoped evidence integrity checks, prompt-overflow audit, and Slurm-only shared-provider dispatch. Generic locked evidence is recorded as present, not automatically verified; roles remain blocked pending role-specific conformance.
- Fixture tests: `tests/test_preflight.py` **17 passed**; full regression suite **110 passed**. Real login/Conda inspection on 2026-11-27 observed Python 3.11.18 and installed package metadata. No compute-node, model-snapshot, tokenizer, quota or GPU evidence has been verified.
- **Still required:** actual site/storage/environment orchestration and evidence creation, full snapshot component/processor verification and immutable staging, role-specific conformance validators and readiness transitions, complete production GPU probe contracts, and end-to-end P0 readiness integration. T04 and all V-gates remain unchecked. Snapshot fixture checks now accept file symlinks only within an explicitly trusted cache root; actual cached model contents still require verification.

### T04 — UIT inspection and preflight orchestration

**Implementation dependencies:** T01; use local fixture providers while T02/T05–T08/T10 are in progress. **Actual probe dependencies:** shared production providers and verified site/resources. **Targets:** `src/geobench/preflight.py`, `configs/uit.json`, `tests/{test_config_artifacts,test_slurm_contract}.py`; persistent `deployment/` evidence.

- [x] T04.1 — Implement inspection/probe descriptors with explicit unknown inputs; test invalid QoS/GRES pairs and missing CPU/RAM/root/walltime rejection. Test that profile changes cannot change research settings. Evidence: configuration/preflight regressions and locked probe-input tests.
- [ ] T04.2 — Prepare site/environment/storage inspection and immutable snapshot staging using V01–V06. Record unresolved access separately by role; do not make DINO access a global readiness flag.
- [ ] T04.3 — Implement probes calling the shared production modules, not duplicate workload implementations. Defer actual GPU probe closure to V08–V10 after providers and Slurm scripts exist.
- [ ] T04.4 — Run relevant configuration/preflight tests. **Acceptance:** every pending input has evidence/status and a scoped gate; no unmeasured resource result is marked passing.

### T05 local progress — implementation incomplete

- E1 transform selection/numerics, exact-hash holdout/duplicate audit and a fixture-tested shared E1 pool path now exist. The pool verifies common/E0 dependencies and real source hashes, keeps a deterministic per-attempt ledger, resumes matching staged prefixes, and publishes `pools/E1/pool_lock.json` after its generation lock.
- A native-resolution E2 img2img loop is exercised with mock VAE/text/UNet/scheduler providers, including strength-dependent active steps, paired latent noise and untruncated token overflow rejection. A local-snapshot runtime loader and fixture-tested candidate/hybrid pool publication exist; real model conformance remains unverified.
- Shared classifier construction and per-seed initialization use verified model/runtime locks. A fixture-tested training core saves epoch-boundary state and resumes deterministically; production run/data-loader orchestration and published checkpoint contracts remain incomplete. T05/T06 and G-P1 remain unchecked.
- Review follow-up: E1 generation records policy/backend identity before candidates and preserves source/config provenance, technical rejection/shortage and locked pool counts. The img2img primitive rejects incompatible scheduler settings and PyTorch modules outside FP32/eval. Tests use mocks and do not establish actual model compatibility. Full local regression: **133 passed**.
- Completed E1/E2 ledger entries are replayed with identity, content and duplicate checks rather than regenerated; partial ledger entries still require deterministic replay. Next: complete production training orchestration and remaining E2 conformance and recovery tests. Independent T07–T11 work remains authorized. No UIT runtime checks or scientific jobs have been performed.

### T05 — Deterministic E1 and E2 generation

**Implementation dependencies:** T01/T02 interfaces or faithful fixtures; no actual dataset/model evidence is required. **Targets:** `src/geobench/augmentation/{__init__,traditional,diffusion}.py`, `tests/{test_traditional,test_diffusion_jobs}.py`.

- [ ] T05.1 — Test confirmed E1 transform selection/order/numerics and D02 shuffle-cycle selection; assert paired sources, VAE seeds and latent noise across strengths.
- [ ] T05.2 — Test D09 first-valid acceptance: attempts 0/1 fail and 2 passes → no attempt 3/4; five failures block quota completion; infrastructure retries preserve attempt identity/cap; holdout collisions block publication.
- [ ] T05.3 — Implement E1 plus shared SD1.5 native-resolution img2img loop, scheduler/runtime, lossless outputs, technical-only QC, duplicate checks and deterministic slot ledgers per E1E2 §§3–5/11. No semantic filtering.
- [ ] T05.4 — Run `pytest tests/test_traditional.py tests/test_diffusion_jobs.py -q`. **Acceptance:** confirmed budgets/source eligibility hold; partial pools cannot enter training; resume preserves accepted outputs and rejection history. Actual-model checks belong to V08.

### T06 — Shared classifier, initialization and resume

**Implementation dependencies:** T01–T03 interfaces or faithful fixtures; actual model/runtime locks gate official initialization only. **Targets:** `src/geobench/{classifier,training}.py`, `tests/test_training_resume.py`.

- [x] T06.1 — Test shared initialization per seed, anchor traversal including last partial batch, exact optimizer-step/scheduler counts, validation RNG isolation and earliest-epoch checkpoint tie-break. Evidence: classifier, training-data and training-resume fixture suites.
- [ ] T06.2 — Implement the full Common runtime without condition-specific defaults; use shared metrics, completed-epoch last/best checkpoints and complete model/optimizer/scheduler/RNG/sampler/history state.
- [x] T06.3 — Compare uninterrupted and interrupted/resumed fixture state and outputs; reject incompatible identities; replay an interrupted epoch from its last completed boundary. Incomplete resumable work must not return completed-stage success. Evidence: deterministic common/A4 replay and interrupted training/publication integration.
- [ ] T06.4 — Run `pytest tests/test_training_resume.py -q`. **Acceptance:** deterministic resume matches uninterrupted fixtures; official initialization creation waits for verified model/runtime locks and probe-state reset.

Local progress: manifest-backed datasets verify split/pool and source hashes, and use confirmed seeded DataLoader settings. Official `train` orchestration validates descriptor, readiness, research config, runtime and initialization locks; it resumes epoch-boundary state and publishes a checkpoint lock only after 20 completed epochs. Leases are scoped to individual runs; schema version 2 includes Mixup state. A4 batches carry pre-normalization pixels and anchor metadata through same-class Mixup, with decisions retained in checkpoint history. Fixture tests compare A4 uninterrupted/resumed weights and decisions, and exercise interrupted E0 training and checkpoint publication. Runs export configuration, metadata including failure/resume events, CSV history and learning-rate steps, and validation metrics. The locked E2 protocol and shared trainer now support nine strength/seed identities; fixture integration trains all nine and supplies their locks to selection. No actual CUDA execution was performed. Remaining work includes full A4 orchestration coverage, recovery hardening, tuning fairness review and integration with T09/T10; T06 remains open. Full regression: **175 passed**.

- [ ] G-P1 — Close implementation P1 after T04 inspection machinery and T05/T06 local tests pass; actual UIT findings are tracked separately. Final GPU evidence remains role-specific V08–V10; this gate does not require DINO weights.

## P2 — E3 realization

### Local numerical progress — incomplete

- E3 generation now reuses the E2 sampler/ledger/QC engine for reference and A1/A2/A3 roles, with selected-strength dependencies, separate reference RNG and cross-role duplicate rejection. Fixture integration covers a reference bank, replay, A1 final generation and its hybrid pool; A2/A3 production integration and full G10 diagnostic conformance remain open.
- DINO extraction/mining publishes locked FP32 arrays and row metadata using the pinned model/official processor loader. A 2,480-reference mock integration checks batches, final partial batch, array shapes, tie-breaking and cache replay. Actual model/processor and GPU verification remain V06/V10; full model-component validation and adversarial recovery tests remain open.
- Regression evidence for this integration: full local suite **176 passed**; after adding E3/guidance source hashes to policy, the four focused generation/extraction integration tests passed again. No additional T07/T08 gate is closed by these partial results.

- Added and fixture-tested active-time/CADS statistics, independent condition streams, CFG/CCFG distances and guidance integration into the shared mock img2img loop. Added FP32 cosine centroid/confuser math and pre-normalization same-class Mixup primitives; added nine-record strength selection with exact tie-break.
- Strength selection now verifies locked nine-run checkpoint and history evidence before publishing its selection; official binding and end-to-end training remain outstanding. These are not completed T07–T09 pipelines: reference-bank generation, locked DINO processor/extraction, RNG checkpoint integration, full guidance diagnostics and pool management remain outstanding.
- Latest integrated regression: **163 passed**. No GPU/model/UIT scientific run has been performed.

### T07 — CADS, CCFG and Geo guidance

**Implementation dependencies:** T05 interface or faithful mocked sampler; no actual diffusion model is required. **Targets:** `src/geobench/augmentation/guidance.py`, extend `augmentation/diffusion.py`, `tests/test_guidance.py`.

- [ ] T07.1 — Test E3 Eq. (1)–(11), active-timestep endpoints/N=1, FP32 population statistics across all tokens/features, invalid/zero std rejection and fresh independent condition noise streams.
- [x] T07.2 — Test CFG-derived positive distance with CHW sum: baseline 0, raw positive `[1,1]`, CFG 2 gives `d_pos=8`; test adaptive scales, interpolation and exactly one scheduler step per active timestep. Evidence: explicit scale/interpolation assertions and shared-sampler step-count tests.
- [ ] T07.3 — Implement confirmed branch roles, paired E2/A1/A2/A3 source/VAE/latent identities, independent reference/CADS namespaces, unchanged Geo prompts and per-timestep JSONL diagnostics. Preserve G08 Geo-specific conventions rather than upstream differences.
- [ ] T07.4 — Run `pytest tests/test_guidance.py -q`. **Acceptance:** numerical fixtures and stream replay pass; prompt overflow is reported before any modification; no extra tuning or semantic filtering. Track real tokenizer audit as V07 and numerical evidence as V11.

### T08 — Reference bank, DINO mining and Mixup

**Implementation dependencies:** T02/T05–T07 interfaces or faithful fixtures; actual DINO access is a deployment gate. **Targets:** `src/geobench/augmentation/{mining,mixup}.py`, extend `diffusion.py` and trainer Mixup hook, `tests/test_mining_mixup.py`.

- [ ] T08.1 — Implement/test CADS reference quotas/source cycles for every class and reference-only singleton handling. G09 reference collisions with frozen E2 final images retry the reference slot without mutating E2; final A1/A2/A3 collisions retry under D09.
- [ ] T08.2 — Implement/test frozen official DINO processor/CLS extraction, FP32 normalization/centroids/cosines, zero-norm failure, exact-tie smallest other class ID and immutable cached confusers. Use mocks for implementation tests when real access is unavailable.
- [ ] T08.3 — Implement/test A4 batch selection, same-class replacement partners, confirmed lambda support, pre-normalization mixing and independent checkpointed RNG. Assert unchanged labels, A3 pool, anchors and optimizer steps.
- [ ] T08.4 — Run `pytest tests/test_mining_mixup.py -q`; verify NumPy array shapes/row mappings and strict JSONL diagnostics schemas. **Acceptance:** A1 final acceptance requires only reference-bank hashes, not DINO features/access; actual extraction requires V06/V10. Track real output evidence as V12.

- [ ] G-P2 — Close E3 implementation after T07/T08 fixture tests and G08 convention review pass; actual access and runtime checks are deployment gates. Record V06/V07/V10–V12 statuses explicitly; unavailable DINO access blocks dependent scientific stages, not this fixture gate or independent work.

## P3 — Binding, Slurm and reporting integration

### T09 — Strength selection, official bindings and pool identities

**Implementation dependencies:** T01/T03/T05/T06 contracts and T08 for A4, using fixtures where necessary. **Targets:** `src/geobench/{cli,config,artifacts}.py`, `tests/{test_config_artifacts,test_evaluation_reporting}.py`.

- [x] T09.1 — Test nine-run completeness, mean Validation Tail Macro-F1 at each run's overall-validation-selected checkpoint and smaller-strength exact tie-break; missing evidence blocks selection. Evidence: `tests/test_selection.py` exercises nine-run selection, missing/duplicate/invalid records, checkpoint/history consistency and immutable lock dependencies; included in the 163-test regression suite.
- [ ] T09.2 — Implement selected-strength lock and all-three-seed verified official E1E2 bindings. Keep source candidates immutable; reject identity/history/manifest mismatch and support the fresh-three-seed alternative without selective seed reruns.
- [x] T09.3 — Implement frozen descriptors for all 18 official identities and A4→A3 canonical pool binding; test no cycles, extra generation or duplicate Base identities. Evidence: 18-descriptor fixture, A4 pool/manifest/anchor-step binding, existing cycle/identity rejection regressions.
- [ ] T09.4 — Run relevant configuration/binding tests. **Acceptance:** execution accounting is 24 unique classifier executions with verified reuse or 27 with fresh E1E2; nine tuning runs remain identifiable and unselected runs are never tested.

### T10 — Slurm templates, submission and recovery

**Implementation dependencies:** T01/T04/T09 descriptor contracts or faithful fixtures. **Actual submissions:** site smoke/probe submissions require V01–V03 and their role inputs; long scientific jobs additionally require V13 and role readiness. **Targets:** `scripts/slurm/{cpu.sbatch,gpu.sbatch,submit.sh}`, `tests/test_slurm_contract.py`, CLI descriptor handling.

- [x] T10.1 — Implement thin module/env/srun wrappers, frozen array-index manifests, resolved resource arguments, persistent logs and JobID/dependency ledger. Use one task/assigned MIG GPU and one writer per pool; no distributed execution framework. Evidence: mocked submission, verified resource/array limits, interpreter-seed dispatch and replacement ledger tests.
- [ ] T10.2 — Test array identity stability, unknown resource rejection, single-writer ownership, lock verification despite `afterok`, incomplete-stage non-success and recovery with replacement JobID relationships. Use `afterany` only for incident collection.
- [x] T10.3 — Individually run `bash -n scripts/slurm/cpu.sbatch`; record its exit status and require zero. Evidence: local Git Bash exit 0.
- [x] T10.4 — Individually run `bash -n scripts/slurm/gpu.sbatch`; record its exit status and require zero. Evidence: local Git Bash exit 0.
- [x] T10.5 — Individually run `bash -n scripts/slurm/submit.sh`; record its exit status and require zero. Recheck every added/modified Slurm shell script individually as well. Evidence: local Git Bash exit 0.
- [ ] T10.6 — Run `pytest tests/test_slurm_contract.py -q`. **Acceptance:** all three independent syntax checks and contract tests pass; actual compute-node smoke/signal/requeue tests remain V13, not implied by shell parsing.

### T11 — Evaluation, exports and final reporting

**Implementation dependencies:** T02/T03/T06/T09 interfaces or faithful fixtures; actual trained models are not required for local tests. **Targets:** `src/geobench/{evaluation,reporting}.py`, `tests/test_evaluation_reporting.py`.

- [x] T11.1 — Test missing/duplicate IDs, wrong labels, nonfinite logits, row-map mismatch, raw `[N,124]` logits and full `124×124` confusion consistency; recompute CSV rankings/metrics from raw outputs. Evidence: prediction export/recomputation tests, including deliberately re-signed fixture files with incorrect labels/row maps.
- [ ] T11.2 — Implement Evaluation §§2–8 inference, per-run artifacts, coverage verification, atomic evaluation locks, main/ablation aliases, group/per-class tables and reporting without checkpoint reselection.
- [x] T11.3 — Test full-precision raw values versus two-decimal percentages, sample std, N/A versus incomplete results, exactly three seeds, no unselected-candidate test and per-run test eligibility without an all-18 barrier. Evidence: metrics/reporting fixtures, missing-seed rejection, aliases and individual E0/reused-E1E2 evaluation integration.
- [ ] T11.4 — Run `pytest tests/test_evaluation_reporting.py -q`, then integrated `pytest tests -q`. **Acceptance:** confirmed acceptance checks across all five specs have test/evidence coverage; templates contain no invented results.

- [ ] G-P3 — Close local implementation integration after T09–T11 tests and shell checks pass. Deployment separately freezes reviewed source/config/environment identities, closes applicable V gates with real shared-code probes, and creates three shared initializations after discarding probe state before corresponding scientific runs.

## Code-first transition evidence — 2026-11-28

- Consolidated local integration now includes official descriptors and whole-triplet E1E2 reuse/fresh planning, A4 pool binding, frozen Slurm arrays/submission intents and replacement ledgers, fixed-checkpoint evaluation/raw exports, and recomputed eighteen-run reports. CLI interfaces are in `docs/PIPELINES.md`.
- Training checkpoint schema 3 retains the complete best-epoch optimizer/RNG/history snapshot on CPU for stable publication recovery. Training's frozen validation snapshot is separate from evaluation's finalized `metrics.json`; no locked training artifact is rewritten for test results.
- Shared-provider classifier/diffusion/DINO probe orchestration, model snapshot materialization, environment/runtime freezing, read-only private-site inspection and scoped readiness receipts now exist. Tests use mock site/model observations. They do not verify UIT, model compatibility or actual Slurm behavior.
- Final-source local regression: **190 passed**. Each of the three Slurm shell files passed its own `bash -n` invocation with exit 0. No jobs were submitted, no UIT deployment was performed, and no original dataset files were modified.
- Open implementation acceptance still requires the unchecked T04–T11 conformance items, especially comprehensive recovery/adversarial coverage, full A4 run orchestration coverage, strict complete G10 diagnostics/failure schemas and final report/provenance completeness. Phase gates remain open; passing these fixtures is not a claim that T01–T11 or the scientific benchmark is complete.

- Documentation aligns implementation, deployment verification and separately approved scientific execution milestones. Existing T01–T11 and V/R/F identifiers and prior test evidence are retained.
- T04 snapshot verification now supports file symlinks within an explicitly trusted cache root and rejects escaping links. T04 remains incomplete; actual site, storage, snapshot component and GPU evidence remain pending.
- Automatic approval review initially rejected a T05 command by citing the earlier T04-only instruction. The user subsequently reaffirmed broader authorization; T05 work is now underway.

## Detailed implementation order — local only

1. Complete T04–T06 shared contracts and functional pipelines using synthetic inputs and mocked or locally available models; missing providers may be implemented in parallel through stable interfaces.
2. Complete T07–T08 against the shared sampler and trainer, testing equations, RNG, references, mining and Mixup with deterministic fixtures.
3. Complete T09–T11 descriptor binding, recovery/Slurm scripts, evaluation and reporting against fixture artifacts. Run every required script syntax check and the integrated regression suite.
4. Review and push the complete implementation. Only then start the separately tracked deployment verification stage below; failures trigger fixes, local regressions, a new commit and server retest. Never infer actual verification from fixtures.

## Deployment and server integration — after implementation

After T01–T11 and local integration tests pass, commit/push the code, deploy a separate checkout to UIT, inspect actual environment/data/storage/model/Slurm inputs and perform CPU/GPU smoke and small shared-code end-to-end tests through Slurm. Diagnose observed failures, add local regression tests, commit/push and retest on server. Actual V-gates stay unchecked until evidence is saved. Full scientific runs require separate user approval.

## Outstanding verification register — actual evidence required

Each item remains unchecked until evidence is saved under the new artifact root. These gates do not block implementation or local mock/fixture tests. These are verification requirements, not permission to revise D01–D10, G01–G10 or EVAL-D01/D02.

- [ ] V01 — **Connection/paths; owner T04; blocks server operations.** Verify UIT VPN, SSH alias `uit` → `<PRIVATE_ACCOUNT>@<PRIVATE_UIT_HOST>:22`, key reference `<PRIVATE_SSH_KEY_PATH>`, login identity `<PRIVATE_LOGIN_HOST>` and `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/`. Keep keys/tokens out of artifacts. Confirm separate code checkout and read-only cleaned dataset at `GeoLocation_DataPreprocessing/data/cleaned_dataset/` beneath that workspace.
- [ ] V02 — **Storage; owners T01/T04; blocks actual publication.** Confirm new version root (candidate `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/benchmarks/geolocation_geodisc_ds_v1/`), quota/free bytes/inodes, model-cache paths, compute-node visibility and atomic/ownership semantics. Estimate output/checkpoint/diagnostic storage; never overwrite/reuse old experiment artifacts or delete them to make space.
- [ ] V03 — **Slurm resources; owners T04/T10; blocks submissions.** Load `slurm/slurm/25.05`; verify account `uit`, partition `normal`, H200 MIG compatibility, CPU count/host RAM, array/concurrency limits and effective account/partition/QoS walltime. Revalidate the observed three-day limit and request the maximum permitted walltime, `3-00:00:00` if confirmed; do not substitute an arbitrary shorter limit.
- [ ] V04 — **Dataset/decoder; owner T02; blocks data freeze.** Verify actual 49,997 JPG/124-class inventory, readability, decoder identity, both content hashes, duplicate feasibility/labels, split quotas and support sets. Publish audit/data/E0 locks only after all checks pass.
- [ ] V05 — **Environment/base models; owners T04/T05/T06; blocks affected model use.** Inspect existing environment via `source /datastore/<PRIVATE_ACCOUNT>/miniconda3/etc/profile.d/conda.sh` then `conda activate geoguessr-bench`. Record actual Python/package origins, Conda/pip exports, PyTorch/CUDA/cuDNN/driver compatibility and deterministic API behavior. Resolve immutable EfficientNet weights and concrete SD1.5 repository/revision/components/hashes/cache paths. Reuse only if compatible; otherwise validate a version-specific clone rather than silently changing the shared environment.
- [ ] V06 — **E3 V01 / DINO access; owners T04/T08; blocks extraction/mining only and their consumers.** Materialize `facebook/dinov3-vitb16-pretrain-lvd1689m` revision `5931719e67bbdb9737e363e781fb0c67687896bc`; verify actual weights/config/official processor contents, hashes, cache path and processor input shape. No checkpoint/processor fallback. Missing access does not block E0, E1E2, CADS reference generation, A1 or mock-based tests.
- [ ] V07 — **E3 V02 / tokenizer audit; owners T04/T07; blocks affected prompt roles.** Audit every rendered base/Geo/confuser/artifact prompt with the actual SD1.5 tokenizer, including special tokens and limit 77. Preserve master strings; report overflow before modification and obtain separate approval for any revision. Geo-only overflow blocks A3/A4, not valid base/A1 stages.
- [ ] V08 — **Diffusion GPU probe; owners T04/T05/T07; blocks scientific generation.** Under Slurm, test actual shared sampler on training sources, native dimensions, strength grid and up to three guidance branches. Record memory/throughput after warmup, actual versions/device/config identity and output/QC conformity. Use verified common generation profile across E2/E3; no DINO model is required for this sampler probe.
- [ ] V09 — **Classifier GPU probe; owners T04/T06/T08; blocks tuning/main training.** Probe full `[128,3,384,896]` forward/backward/update with optimizer allocation, FP32/AMP OFF, inference, actual DataLoader and A4 mixing path. Record memory/throughput and worst epoch/checkpoint duration; lock one shared classifier profile, discard probe model/optimizer/BatchNorm/RNG state and reload pristine initialization.
- [ ] V10 — **DINO GPU probe; owners T04/T08; blocks actual extraction.** Verify frozen FP32 batch 32 plus final partial batch and official preprocessing under Slurm; save device/runtime/performance evidence. Its failure must not invalidate V08/V09 or block reference/A1 work.
- [ ] V11 — **E3 V03 / numerical conformance; owners T03/T07/T08; blocks corresponding E3 mechanisms.** Save Eq. (1)–(11), statistics/norm failures, branch/paired RNG, retry and Mixup test evidence. Preserve G08's already-documented static upstream comparison; do not claim untested upstream runtime parity.
- [ ] V12 — **E3 V04 / materialized artifacts; owners T01/T05/T08/T11; blocks each artifact's consumers.** Verify actual bank index/hashes, NumPy arrays/row maps/shapes, strict per-timestep JSONL, candidate ledgers, pool/run/evaluation locks and resume states as each stage publishes them. Fixture outputs do not substitute for production artifacts.
- [ ] V13 — **Recovery/site smoke; owners T06/T10; blocks long workloads.** Verify harmless Slurm compute-node smoke, signal forwarding, requeue permissions, persistent checkpoints, safe writer recovery and downstream rejection of incomplete work. Set signal grace using measured epoch/checkpoint time; resume completed epochs/slots under the same identity. If a full epoch cannot fit maximum walltime, report/block rather than change research settings or add mid-epoch policy.

For V08–V10, begin with QoS `mig3-71`, GRES `gpu:3g.71gb:1`; if required, probe unchanged workloads on `mig7-141`, `gpu:7g.141gb:1`. Available `mig1-18`/`gpu:1g.18gb:1` and `mig2-35`/`gpu:2g.35gb:1` are not automatic fallbacks. Lock compatible role profiles before their experiments. GPU workloads run only through Slurm; unresolved constraints block affected work without changing confirmed training/generation settings.

## P4 — Scientific execution checklist (future execution only)

**Targets:** new-root `deployment/jobs/`, `splits/`, `pools/`, `augmentation/`, `initializations/`, `runs/`, `logs/`; use plan §4 stage graph and §5 protocol. Each stage verifies its own lock dependencies; scheduler status alone never opens the next gate.

| Official condition | Seed 42 | Seed 123 | Seed 2026 | Identities |
|---|---|---|---|---:|
| E0 | E0/42 | E0/123 | E0/2026 | 3 |
| E1E2 = Base | E1E2/42 | E1E2/123 | E1E2/2026 | 3 |
| E3-A1 | A1/42 | A1/123 | A1/2026 | 3 |
| E3-A2 | A2/42 | A2/123 | A2/2026 | 3 |
| E3-A3 | A3/42 | A3/123 | A3/2026 | 3 |
| E3-A4 | A4/42 | A4/123 | A4/2026 | 3 |
| Total | 6 | 6 | 6 | 18 |

- [ ] R01 — **Deps:** T01–T03, V01–V04. Freeze actual new inventory/split/supports and shared E0 real pool. **Gate/test:** verify data locks, hashes, quotas and no file/pixel duplicate leakage; no old split reuse.
- [ ] R02 — **Deps:** R01, T04/T06/T10, V05/V09/V13, applicable G-P3 evidence. Freeze runtime/profile/source locks and create the three pristine shared initialization artifacts. **Gate/test:** reload/hash equality per seed; reject residual probe state.
- [ ] R03 — **Deps:** R02. Train E0 for all three seeds using only E0 membership. **Gate/test:** E0 acceptance criteria, full schedules, best/last/history/checkpoint locks and resume evidence. This branch does not wait for diffusion or DINO access.
- [ ] R04 — **Deps:** R01, T05. Generate/freeze the single shared E1 pool. **Gate/test:** exact eligible-class quotas, transform/attempt ledgers, technical QC and unchanged real membership.
- [ ] R05 — **Deps:** R04, T05/T10, V05/V07 base roles/V08/V13. Generate/freeze three E2 candidate pools at strengths `{0.25,0.35,0.50}` and hybrid manifests. **Gate/test:** paired source/VAE/latent identities, same E0/E1/budgets, complete quotas and verified locks; one writer per pool.
- [ ] R06 — **Deps:** R02/R05, T09. Run all nine strength candidates (three strengths × three common seeds). **Gate/test:** each completes common schedule and records Tail Macro-F1 at its overall-validation-best checkpoint; never evaluate candidate tests.
- [ ] R07 — **Deps:** all R06 outputs. Select/freeze shared strength by three-seed mean Tail Macro-F1, smaller-strength exact tie-break. Verify all three selected runs and bind as official E1E2, or execute all three fresh official runs if reuse cannot be verified. **Gate/test:** immutable candidate evidence and three valid official identities before reused E1E2 test.
- [ ] R08 — **Deps:** R07, T07/T08, V07 base roles/V08/V11. Generate/freeze the CADS-only 124×20 reference bank using selected E2 final hashes for G09. **Gate/test:** exact quotas/namespace/isolation and reference–E2 duplicate rejection. No DINO checkpoint dependency and no dependency on A1 final pool.
- [ ] R09 — **Deps:** R08 and selected base locks. Generate/freeze A1 final/hybrid pool, then train its three seeds using R02 initialization/runtime. **Gate/test:** bank-hash duplicate checks, E3-A1 acceptance and complete classifier locks. A1 does not wait for R10 or DINO access.
- [ ] R10 — **Deps:** R08, V06/V10. Extract/cache DINO features, centroids, similarities and Top-1 confusers. **Gate/test:** actual arrays/mappings/FP32 norms/ties and mining locks per E3 §5; no reference images become classifier training inputs.
- [ ] R11 — **Deps:** R07/R10, V07 applicable roles/V08/V11. Generate/freeze A2 and A3 final/hybrid pools using the same cached bank/confusers. **Gate/test:** shared selected strength/budgets/paired attempts, technical-only QC, diagnostics and unchanged prompts. A3 additionally requires successful Geo token audit.
- [ ] R12 — **Deps:** R11 A3 pool, T08/T09. Bind A4 to the exact A3 synthetic/hybrid pool; generate no new images. **Gate/test:** same pool hashes, anchors and optimizer-step counts, with only the confirmed training-time Mixup hook enabled.
- [ ] R13 — **Deps:** R02 and each condition's ready R11/R12 pool. Train A2/A3/A4 for all three seeds (nine runs); do not wait for unrelated unfinished conditions. **Gate/test:** same seed initial states/runtime, full schedule/resume artifacts and A3/A4 step equality. Together with R09 this yields 12 E3 runs.
- [ ] R14 — **Deps:** T11 and each frozen official best checkpoint/binding from R03/R07/R09/R13. Export validation/test predictions and all required per-run metrics/artifacts. **Gate/test:** real-split coverage, recomputation, full label space, checkpoint/test isolation and evaluation lock. No all-18 wait for eligible per-run test; complete reused E1E2 triplet binding is required.
- [ ] G-P4 — Verify exactly 18 official classifier identities/evaluations plus all nine strength-search records; account for 24 unique executions with reuse or 27 without. **Gate:** shared E0/E1, E2/E3 budgets, selected strength, cached A2 bank and A3/A4 pool/steps all match; record shortages/failures instead of silently changing methodology.

## P5 — Final acceptance and handoff

**Depends on:** G-P4, T11 and all applicable verification evidence. **Targets:** new-root `reports/`, evaluation/report locks and deployment/artifact index; follow Evaluation §§6–10 and plan §§8–9.

- [ ] F01 — Recompute official metrics from preserved raw outputs; verify full three-seed aggregation, support tables, per-class/group results and confusion matrices. **Test/gate:** no missing seeds, invented values, unsupported-class zeros or precision loss in raw files.
- [ ] F02 — Publish main E0/E1E2/A4 and ablation Base/A1/A2/A3/A4 tables with confirmed percentage formatting. **Test/gate:** aliases resolve existing official evidence without new runs; all required reporting artifacts and locks verify.
- [ ] F03 — Audit acceptance criteria in all five specs and close V01–V13 with actual evidence or explicitly report blocked/incomplete stages. **Test/gate:** no unresolved applicable verification can coexist with a benchmark-complete claim.
- [ ] F04 — Assemble source/config/environment/model snapshots, verified UIT resource/max-walltime profile, JobID/dependency ledger, artifact index, checkpoints/resume states and final reports. Preserve unselected/rejected evidence; do not clean or upload artifacts as a side effect.
- [ ] G-P5 — Declare completion only after F01–F04 and actual 18-run evaluation/report locks pass. Passing code tests or writing this checklist alone is not benchmark completion.
