# GeoLocation Geo-DiSC-DS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Implementation/delegation must be requested separately; the current task only writes specifications and this plan.

**Goal:** Xây benchmark tái lập được cho E0, E1E2 và E3-A1–A4, chạy trên UIT Slurm và xuất đầy đủ evidence/results theo các specs đã chốt.

**Architecture:** Một Python package, một CLI và các modules theo trách nhiệm; không xây workflow engine hoặc dịch vụ. CLI dùng cùng manifests/configs trên local fixtures và UIT; Slurm chỉ cấp tài nguyên/thực thi, còn artifact identities/locks quyết định dữ liệu hoặc run đã hoàn tất. Một sampling loop diffusion dùng chung cho E2/CADS/CCFG/Geo, một classifier trainer dùng chung với Mixup hook riêng A4.

**Tech Stack:** Python trong môi trường UIT đã xác minh; PyTorch/torchvision, diffusers, transformers/Hugging Face Hub, NumPy, Pillow, PyYAML và pytest. Dùng Python standard library cho CLI, CSV/JSON/JSONL, hashing và filesystem publication; Slurm 25.05 qua Bash. Khóa **actual compatible versions**, không tự nâng shared environment hoặc chọn package versions theo scores.

**Spec:** [Master Markdown](GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), [Common v1.3](specs/00_common.md), [E0 v1.1](specs/01_E0_original.md), [E1E2 v1.2](specs/02_E1E2_hybrid_baseline.md), [E3 v1.1](specs/03_E3_geodisc_ds.md), [Evaluation v1.1](specs/04_evaluation.md).

**Status:** Chỉ lập kế hoạch. Chưa tạo package/scripts/environment, kết nối UIT, chạy tests/preflight, tải weights hoặc thực thi benchmark. `docs/tasks.md` chưa được điền; các checkboxes trong plan là implementation work tương lai.

## Global Constraints

- Cleaned input: 49.997 JPG / 124 classes; không cleaning lại, không sửa dataset, không import split/experiment artifacts cũ.
- Fixed split mới theo Common Spec, split seed 42; exact duplicates qua file SHA-256 và decoded RGB hashes phải cùng split, giữ quota; conflict/infeasible allocation block freeze.
- Training seeds `[42,123,2026]`, augmentation seed 314159; shared classifier initial state theo seed.
- EfficientNet-B0 input `896 × 384`, tensor `[B,3,384,896]`, center letterbox giữ tỷ lệ; batch 128, FP32, AMP OFF và toàn bộ common runtime giữ nguyên.
- Shared E0 real pool và E1 traditional pool; E1 medium target 100, E2/E3 tail target `min(100,5*n_train)`, singleton không final synthetic/Mixup.
- E2: SD1.5 img2img, DPM-Solver++ realization D04, 50 configured steps, CFG 2.0, batch 1, native reflection-pad `/8` rồi crop; không TI/LoRA.
- D01–D10, G01–G10 và EVAL-D01/D02 đã confirmed; không thay equations, prompt text, hyperparameters hoặc ablation để khớp code upstream hay tài nguyên server.
- Strength grid `{0.25,0.35,0.50}`, đủ ba seeds/strength; checkpoint theo overall Validation Macro F1, strength theo mean Validation Tail Macro F1, exact tie lấy strength nhỏ hơn.
- References: 124 × 20 CADS synthetic, training sources only; frozen DINOv3 CLS 768, checkpoint/revision/processor G04, FP32 batch 32; references không vào classifier training.
- A2 cache được dùng nguyên ở A3/A4. A4 dùng nguyên pool A3 và cùng optimizer steps; only same-class Mixup, trước normalization, đúng ratio/lambda/RNG đã chốt.
- QC chỉ technical/hash duplicate, first-valid tối đa 5 attempts/output slot; shortage/identity/leakage block publication. Không semantic filtering/quality thresholds.
- Primary metric Macro F1 trên supported GT classes; full 124 logits và full-split confusion counts. Logits descending/class ID ascending cho exact ties. Reports percentage hai decimals, mean ± sample std `ddof=1`; raw values full precision.
- GPU workloads chỉ chạy trong Slurm allocation; dùng persistent storage và maximum effective permitted walltime. Research config không có fallback; nếu cần thay phải có explicit approval riêng.

## Review Focus

1. **Duplicate components không thể vừa giữ quota vừa tách splits:** fail preparation, không âm thầm deduplicate/nới quota; Task 2 có fixture feasible và infeasible.
2. **Retry làm lệch paired source/noise giữa strengths/variants:** seeds/index theo slot/attempt, không theo completion order; Tasks 5/7 kiểm tra first-valid và stable pairing.
3. **CCFG positive distance vô tình dùng raw prediction hoặc mean thay sum:** Task 7 kiểm tra CFG-derived distance và CHW sum với numeric fixture.
4. **Slurm timeout/requeue bị xem như stage success hoặc tạo hai writers:** Tasks 4/6/10 kiểm tra interrupted publication, epoch rollback và lock/job ownership.
5. **Reference leakage và double-count official reuse:** Tasks 8/11 kiểm tra cross-role duplicate rejection, official binding, Base/A4 aliases và 18-run accounting.

---

## 1. Kiến trúc và module boundaries

### 1.1. Source layout dự kiến

```text
pyproject.toml
configs/
  benchmark.json                 # Confirmed research settings từ specs
  methods.json                   # E1/E2/A1/A2/A3/A4 flags và parameters
  uit.json                       # Deployment inputs; không override research
src/geobench/
  __init__.py
  __main__.py                    # Entry point gọi CLI
  cli.py                         # Argparse, dispatch, exit status; không chứa algorithms
  config.py                      # Load/validate/resolve configs và run/job descriptors
  artifacts.py                   # Hash, canonical records, locks, atomic publication
  images.py                      # RGB decode/hash, classifier geometry/normalize, native pad/crop
  data.py                        # Inventory, duplicate components, quota split, manifests/groups
  classifier.py                  # Model/loss/optimizer/scheduler, shared initial states
  training.py                    # Một common trainer, validation, history, checkpoint/resume
  metrics.py                     # Pure ranking/counts/metrics/aggregation/formatting
  evaluation.py                  # Inference, coverage, prediction/logit/metric artifacts
  reporting.py                   # Main/ablation tables, alias mapping, final report
  preflight.py                   # Deployment inspection/readiness và isolated GPU probes
  augmentation/
    __init__.py
    traditional.py               # E1 operators, quota execution
    diffusion.py                 # SD/VAE/scheduler, source/seed jobs và shared sampling loop
    guidance.py                  # Pure CADS, CFG, CCFG, active schedule và diagnostics
    mining.py                    # Reference bank orchestration, DINOv3, centroids/confusers
    mixup.py                     # A4 batch selection, real partners, lambda và RNG state
scripts/slurm/
  cpu.sbatch                     # Generic CPU stage allocation/CLI launch
  gpu.sbatch                     # Generic one-GPU stage/array worker
  submit.sh                      # Thin staged submission, manifest/index mapping, job-ID ledger
tests/
  test_config_artifacts.py
  test_data_images.py
  test_metrics.py
  test_training_resume.py
  test_traditional.py
  test_diffusion_jobs.py
  test_guidance.py
  test_mining_mixup.py
  test_slurm_contract.py
  test_evaluation_reporting.py
```

Đây là proposed implementation structure, chưa tạo files. Không nhân bản trainer cho từng condition hoặc sampler cho từng ablation. Dùng dict/Path/tensors và CSV/JSON contracts hiện có; không thêm registry/plugin framework, database, web UI, experiment-tracking service hoặc custom distributed scheduler.

### 1.2. Interfaces chốt cho implementation

`Path` là `pathlib.Path`; JSON-compatible metadata dùng dict/list; `Tensor` là PyTorch tensor. Các hàm trả `Path` trỏ tới artifact/lock đã publish, không trả “success” khi còn output dang dở.

| Module | Interface chính | Trách nhiệm / dependency |
|---|---|---|
| config | `load_config(path: Path) -> dict`; `resolve_run(root: Path, run_id: str) -> dict` | Validate fixed values và resolve descriptors, không apply deployment overrides vào research |
| artifacts | `sha256_file(path: Path) -> str`; `verify_lock(path: Path) -> dict`; `publish_lock(root: Path, descriptor: dict) -> Path` | Canonical hashes, dependency/file verification, một writer và atomic commit |
| images | `classifier_geometry(image) -> Tensor`; `normalize_classifier(x: Tensor) -> Tensor`; `pad_native(image) -> tuple[Tensor,dict]`; `crop_native(x: Tensor, padding: dict) -> Tensor` | Geometry tách normalization để A4 mix trước normalize; file/pixel hashes đúng Common |
| data | `prepare_data(dataset_root: Path, root: Path) -> Path` | Inventory → deterministic duplicate-aware split → common data lock |
| classifier | `create_initializations(root: Path) -> Path`; `build_classifier(config: dict)` | Common weights/heads/states và optimizer/scheduler; không variant-specific config |
| metrics | `rank_logits(logits: Tensor) -> Tensor`; `compute_metrics(y_true, logits, groups: dict) -> dict`; `aggregate_seeds(records: list[dict]) -> dict` | Pure full-label/support-aware calculations; không model/file inference |
| traditional | `generate_e1(root: Path) -> Path` | Cùng E0/group/config dependencies → một frozen E1 pool |
| diffusion | `generate_pool(root: Path, descriptor: dict) -> Path` | Pool role/variant/strength từ descriptor; shared jobs/pad/VAE/denoise/QC/retry loop |
| guidance | `predict_step(context: dict) -> tuple[Tensor,dict]` | Eq. (1)–(11) theo branch flags; trả noise prediction và JSON-compatible diagnostics |
| mining | `build_references(root: Path) -> Path`; `mine_confusers(root: Path) -> Path` | Reference lock → model/processor extraction → feature/mining locks |
| mixup | `mix_batch(batch: dict, context: dict) -> tuple[dict,dict]` | Thay eligible anchors trước normalization, giữ labels/counts; independent RNG |
| training | `train_run(root: Path, run_id: str) -> Path` | Verify descriptors/locks, shared init hoặc compatible resume, full schedule và checkpoint lock |
| evaluation | `evaluate_run(root: Path, run_id: str, split: str) -> Path` | Resolve official binding/checkpoint, common inference, frozen outputs |
| reporting | `write_reports(root: Path) -> Path` | Chỉ đọc verified evaluation locks để aggregate/render đúng aliases |
| preflight | `inspect_deployment(root: Path, profile: dict) -> Path`; `probe_gpu(root: Path, role: str) -> Path` | Deployment evidence trước freeze; không selection theo validation/test |

CLI là các thin commands `prepare-data`, `preflight`, `initialize`, `generate-e1`, `generate-e2`, `tune`, `select-strength`, `bind-e1e2`, `references`, `mine`, `generate-e3`, `train`, `evaluate`, `report`, `verify`. Mỗi command nhận explicit artifact root và descriptor/run ID; Slurm array index chỉ chọn một row từ frozen job manifest, không tự tạo hyperparameters.

## 2. UIT integration và deployment inputs

### 2.1. Thông tin người dùng đã cung cấp

| Mục | Giá trị |
|---|---|
| Connection | UIT VPN bắt buộc; SSH alias `uit`, host `<PRIVATE_UIT_HOST>`, port `22`, user `<PRIVATE_ACCOUNT>` |
| Login node | `<PRIVATE_LOGIN_HOST>` |
| SSH key | `<PRIVATE_SSH_KEY_PATH>`; chỉ tham chiếu path, không đưa key/token vào repo hoặc artifacts |
| Slurm module/account/partition | `slurm/slurm/25.05` / `uit` / `normal` |
| Hardware | NVIDIA H200, MIG profiles dưới đây |
| Workspace | `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/` |
| Repository | `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/GeoLocation_CADS_CCFG` |
| Read-only cleaned dataset | `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/GeoLocation_CADS_CCFG/data/cleaned_dataset/` |
| Existing Conda env | `geoguessr-bench`, từ `/datastore/<PRIVATE_ACCOUNT>/miniconda3/` |

| QoS | GRES | Deployment use |
|---|---|---|
| `mig1-18` | `gpu:1g.18gb:1` | Available profile; không tự giảm xuống cho common classifier |
| `mig2-35` | `gpu:2g.35gb:1` | Available profile, chưa được chọn |
| `mig3-71` | `gpu:3g.71gb:1` | Preferred initial profile để preflight |
| `mig7-141` | `gpu:7g.141gb:1` | Larger profile được cho phép thử nếu preflight cần |

Không chạy CUDA/model probes, generation, feature extraction, training hoặc inference trên login node. Login node chỉ dùng kết nối, inspect metadata nhẹ, stage code/configs và submit/monitor Slurm.

### 2.2. Tách code, input và new benchmark artifacts

- Proposed new code checkout: `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/Geo_CADS_CCFG_MixUp/`; xác minh path chưa xung đột trước deploy, không ghi vào old repository.
- Proposed **new** version root: `/datastore/<PRIVATE_ACCOUNT>/<PRIVATE_WORKSPACE_OWNER>/benchmarks/geolocation_geodisc_ds_v1/`; đây là candidate deployment input, chưa được tạo/chấp thuận quota.
- New root chứa `deployment/`, `configs/`, `splits/`, `augmentation/`, `pools/`, `initializations/`, `runs/`, `reports/`, `logs/` và `model_cache/` theo artifact schemas trong specs.
- Dataset giữ ở path đã cung cấp, không copy/clean lại; inventory IDs/fingerprint dùng dataset-relative paths, absolute dataset root chỉ là deployment binding.
- Old splits/images/checkpoints/reports/experiment locks không được overwrite hoặc reuse. Old source chỉ là reference cho ba mechanisms đã cho phép; không import old defaults.
- HF/model caches có thể đặt dưới new version root để quota/provenance rõ; SD1.5/DINOv3 actual snapshot locations vẫn là deployment inputs cần verify. Pretrained snapshot phải đúng immutable hashes, không lấy old experiment checkpoints làm model weights.

### 2.3. Environment inspection trước reuse

Trong deployment sau này, activate đúng existing env:

```bash
source /datastore/<PRIVATE_ACCOUNT>/miniconda3/etc/profile.d/conda.sh
conda activate geoguessr-bench
```

Inspect Python executable/module origins, package versions, Conda explicit spec/pip freeze, PyTorch build và driver/CUDA compatibility. GPU checks phải ở Slurm compute allocation; không giả định environment cũ hỗ trợ DINOv3 processor/API hoặc confirmed scheduler flags.

Nếu compatible, reuse environment và khóa actual resolved package/code/model identities. Nếu không compatible, chuẩn bị version-specific cloned environment thay vì sửa shared env cũ trong im lặng; validate API/numerical contracts rồi mới freeze. Không thay FP32, batch, worker count hoặc model để làm vừa environment. Không cài dependencies giữa các official runs.

## 3. Preflight và outstanding verification

Preflight ở đây là deployment work được người dùng yêu cầu, không thêm classifier metric hoặc sửa confirmed research specs. Tất cả trạng thái dưới đây hiện **chưa chạy**.

### 3.1. Site/resource readiness

1. Verify VPN/SSH alias/key routing và actual login identity; xác minh workspace/source/data paths, read access và storage filesystem.
2. Load `slurm/slurm/25.05`, xem version và read-only scheduler state: `sinfo`, `scontrol show partition normal`, `sacctmgr show qos`, `sacctmgr show assoc user=<PRIVATE_ACCOUNT> account=uit`. Nếu association limits bị ẩn thì cần site/admin confirmation, không đoán.
3. Validate `account=uit`, partition/QoS/GRES compatibility, available H200 MIG profiles, CPU/host RAM requests, job/array/concurrency limits và effective maximum walltime.
4. Revalidate **previously observed 3-day QoS limit**. Nếu effective maximum đúng 3 ngày, long-running jobs request `--time=3-00:00:00`; nếu khác, request đúng maximum được phép. Không hard-code 24h/48h ngắn hơn và không dùng unlimited giả khi chưa được phép. Effective limit phải xét cả account/partition/QoS rules, không chỉ một field. [Slurm QOS reference](https://slurm.schedmd.com/qos.html)
5. Verify quota/free capacity/inodes và new artifact/cache roots; estimate storage từ measured output sizes, generated/reference counts và checkpoints/traces, không xóa old artifacts để lấy chỗ.
6. Validate shared filesystem atomic publication/ownership behavior và paths nhìn thấy từ compute node. Persistent `/datastore` là authority; node-local scratch chỉ là cache tùy chọn, không được giữ bản checkpoint/log duy nhất.

### 3.2. Isolated GPU readiness probes

- Start `mig3-71` với `--gres=gpu:3g.71gb:1`, account/partition đã verify. Ghi actual device/MIG identity, driver, PyTorch/CUDA/cuDNN, environment và code/config hashes.
- Classifier probe dùng **full batch `[128,3,384,896]`**, common model/loss/optimizer FP32, AMP OFF, forward/backward/update và optimizer-state allocation thật; kiểm tra cả inference và A4 mixing path bằng fixture hợp lệ, không đổi anchors/steps.
- Diffusion probe dùng native padded source sizes, generation batch 1, SD1.5 locked runtime; bao phủ configured strength grid và tối đa ba guidance branches/Geo conditioning. Chỉ training-source images được dùng; không chọn settings theo chất lượng hoặc validation/test scores.
- DINOv3 probe dùng pinned model/official processor, FP32 batch 32 và partial last batch. Không fallback checkpoint/processor nếu access chưa sẵn sàng.
- Sau warmup ổn định, ghi full-step CUDA peak allocated/reserved memory, observed device memory, step/epoch hoặc image latency và throughput với số iterations/duration đã ghi rõ. Tách measurements theo workload và end-to-end DataLoader path; không so throughput thành benchmark performance metric.
- Nếu `mig3-71` không đủ, thử **cùng research config** trên `mig7-141`/`gpu:7g.141gb:1`. Nếu cả hai không đáp ứng, block và báo constraint; không tự giảm batch/resolution, bật AMP, đổi attention/backend hoặc thêm accumulation.
- Chọn và khóa deployment profile trước tuning/main runs. Mọi classifier runs dùng một common profile; mọi E2/E3 generation dùng cùng generation profile. Role DINOv3 có thể có profile riêng đã verify, nhưng không đổi processor/runtime. Không chuyển profile giữa strengths/ablations để hoàn tất riêng run.
- Hủy probe model/optimizer/BatchNorm/scaler state, reset RNG và reload đúng shared initialization trước training. Probe updates không được tính vào official steps hoặc checkpoints.

Lưu `deployment/preflight.json`/logs cùng frozen environment/profile/model identity. Không ghi bằng chứng memory/throughput chưa đo thành pass. Đây không phải batch-size search hoặc training hyperparameter tuning.

### 3.3. Verification register phải được đóng trước bước liên quan

| Requirement | Chưa được xác minh | Gate / owner task |
|---|---|---|
| UIT profile/site | Partition/QoS/GRES compatibility, CPU/RAM, concurrency, effective max walltime | Tasks 4/10, trước submit thực |
| Persistent storage | Final artifact root/quota/inodes/cache paths và atomic write semantics | Tasks 1/4, trước publication |
| Environment | Actual Python/PyTorch/CUDA/library versions, API compatibility và deterministic operations | Tasks 4/6/7, trước shared runtime freeze |
| SD1.5 | Concrete base repository/revision/weights/tokenizer/VAE/scheduler config hashes, access/cache | Tasks 4/5, trước generation |
| E3 V01 | DINOv3 revision `5931719e67bbdb9737e363e781fb0c67687896bc` đã resolve, weights/config/official processor contents và input shape chưa verified | Tasks 4/8, trước extraction |
| E3 V02 | Token audit của mọi base/Geo/confuser/artifact prompt, special tokens <=77 | Tasks 4/7; overflow report trước khi sửa, cần explicit prompt revision riêng nếu overflow |
| E3 V03 | Numerical conformance Eq. (1)–(11), std/norm failures, paired RNG, Mixup stage và retry cap | Tasks 3/7/8; không claim upstream runtime parity |
| E3 V04 | Actual `.npy` mappings/shapes, JSONL diagnostics, frozen bank index, pool/run locks và resume states | Tasks 1/5/8/11 trước downstream publication |
| Dataset/split | Actual counts/readability, duplicate labels/feasibility, support sets và content hashes | Task 2, trước split lock |
| Recovery | Signal/requeue/timeout permission, one-writer behavior, epoch/checkpoint time so với walltime | Tasks 6/10, trước long jobs |
| Evaluation | Exact-tie ranking, support-aware recomputation, percentage formatting và reuse identity | Tasks 3/11, trước validation/official publication |

G08 source inspection **đã hoàn tất** ở E3 Spec, pin commit `5d0585a9a007c07f1742d5fd187528872d6c8e64`; giữ Geo differences, không sửa master để khớp upstream. Tests/runtime realization vẫn chưa có. Không còn pending policy trong D01–D10/G01–G10/EVAL-D01/D02; resource/access/overflow evidence không được ngầm chuyển thành quyền thay policy.

## 4. Slurm execution và job dependencies

### 4.1. Thin execution contract

`cpu.sbatch` và `gpu.sbatch` load module/activate verified env, set explicit working directory, resolve root/job manifest, rồi launch CLI qua `srun`. GPU jobs dùng một task/một assigned MIG GPU, không DataParallel/DDP; CPU/RAM allocations được materialize từ preflight, không tự đổi DataLoader `num_workers=4`.

Các submit options gồm account/partition/QoS/GRES, resolved maximum walltime, CPU/RAM, persistent output/error paths và job manifest. Không đặt shell variables trong `#SBATCH` rồi kỳ vọng được expand; wrapper truyền resolved arguments cho `sbatch`. `afterok` chỉ là scheduler barrier, CLI vẫn verify upstream locks. [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html)

Job arrays dùng `SLURM_ARRAY_TASK_ID` để lookup frozen rows; task index không suy từ thứ tự JobID. Throttle concurrency bằng `%C`, C lấy từ verified account/site capacity, không làm tăng batch size. Re-submit chỉ missing/failed rows có cùng logical identity, không thêm random seeds. [Slurm job-array reference](https://slurm.schedmd.com/job_array.html)

### 4.2. Stage graph

| Stage / job family | Resource | Depends on | Completion authority |
|---|---|---|---|
| Inventory/split | CPU | New root, dataset read-only, decoder identity | Common data lock |
| Environment/model/token/GPU probes, initialization | CPU + allocated GPU probes | Data lock và model access | Deployment report, shared runtime/initial-state locks |
| E0 training, 3 seeds | GPU array | Shared locks + E0 pool | 3 complete E0 checkpoint locks |
| E1 generation | CPU | Data/E0/config locks | One E1 pool lock |
| E2 generation, 3 strengths | GPU array, one pool/writer per strength | Data/E0/E1 và generation config | 3 candidate pool + hybrid manifest locks |
| E2 search, 9 rows | GPU array | Candidate pools, E1, shared runtime/init | Full candidate run evidence |
| Strength select + official E1E2 bind | CPU | All 9 candidate runs | Selection lock + 3 verified official bindings, hoặc fresh-three-seed path |
| CADS reference bank | GPU, one bank writer | Selected strength/base config, real train/E2 final hashes | 124×20 bank lock |
| A1 final generation | GPU, one pool writer | Selected base/strength, bank hash index | A1 final/hybrid pool locks; không cần DINO mining |
| DINOv3 mining | GPU extraction + CPU cosine | Bank và verified model/processor | Feature/mining/confuser locks |
| A2/A3 final generation | GPU, distinct pool writers | Same bank/mining cache, selected config; A3 token audit | A2/A3 pool locks |
| A3→A4 binding | CPU | A3 canonical pool, confirmed Mixup config | A4 pool reference/identity |
| E3 classifier runs, 4×3 | GPU arrays by ready pool | Own pool + shared init/runtime | 12 checkpoint locks |
| Official evaluation | GPU inference + CPU metrics | Per-run frozen best/official binding, fixed real splits | Evaluation locks |
| Main/ablation reporting | CPU | All required official evaluation locks | Report lock |

Source/generation/mining jobs không cần E0 trained weights. A1 sampler không dùng DINO/CCFG nhưng final acceptance phải có bank hashes cho G09; bank không phụ thuộc A1 final pool nên không có vòng. A2/A3 chỉ dùng cùng cached mapping. A4 không có generation job.

Initial implementation chạy một sequential writer cho mỗi pool/class-order domain, không chia một pool thành nhiều concurrent class writers vì exact-duplicate acceptance có deterministic order. Parallelize **independent pools/strengths/runs**, không thêm distributed generation coordinator.

`afterok` descendants của failed jobs có thể cần submit lại dependencies sau recovery; không giả định requeue parent tự sửa mọi dependency. Chỉ submit downstream khi upstream đã có lock hợp lệ và lưu mới JobID relationships. `afterany` chỉ dùng cho status/incident collection, không mở training/report gate. [Dependency semantics](https://slurm.schedmd.com/sbatch.html)

### 4.3. Walltime và resumability

Request maximum permitted walltime đã verify cho long workloads; nếu tổng job dài hơn limit thì tiếp tục bằng cùng logical run trên allocations kế tiếp. Không rút schedule hoặc bỏ epochs cho vừa một job.

Plan signal trước timeout, chọn grace từ measured worst epoch/checkpoint duration; wrapper/worker phải xử lý signal để dừng nhận unit mới và commit safe boundary. Không tạo mid-epoch resume. Nếu epoch không kịp commit, allocation sau restore last completed epoch rồi replay. Generation commit ở slot/job boundaries. Cần verify signal forwarding và requeue permissions ở UIT; `--requeue` không tự tạo application checkpoint. [Slurm signal/requeue options](https://slurm.schedmd.com/sbatch.html), [scontrol recovery commands](https://slurm.schedmd.com/scontrol.html).

Checkpointed-but-incomplete stage không exit thành công như stage completed để release `afterok`; ghi resumable status và tiếp tục/re-submit cùng descriptor. Sau một timeout, không retry numerical/data errors vô hạn như lỗi hạ tầng. Nếu một full epoch không thể hoàn tất trong maximum effective allocation, block và báo resource constraint, không tự thêm mid-epoch policy.

## 5. Complete benchmark protocol và accounting

### 5.1. Official matrix — đúng 18 identities

| Condition | Seed 42 | Seed 123 | Seed 2026 | Count |
|---|---|---|---|---:|
| E0 | E0/42 | E0/123 | E0/2026 | 3 |
| E1E2 = Base | E1E2/42 | E1E2/123 | E1E2/2026 | 3 |
| E3-A1 | A1/42 | A1/123 | A1/2026 | 3 |
| E3-A2 | A2/42 | A2/123 | A2/2026 | 3 |
| E3-A3 | A3/42 | A3/123 | A3/2026 | 3 |
| E3-A4 | A4/42 | A4/123 | A4/2026 | 3 |
| **Tổng** | **6** | **6** | **6** | **18** |

### 5.2. Execution order và selection

1. Tạo new inventory/split/supports, shared E0/E1 pools, resolved runtime/model locks và three initial states. E0 có thể chạy khi dependencies riêng sẵn sàng.
2. Sinh E2 tail pools cho đúng ba strengths, same paired sources/VAE/latent seeds, same E1/real pools và budgets. Freeze hybrid manifest từng candidate trước search.
3. Train đủ **9 candidate executions**, ba strengths × `[42,123,2026]`, full common schedule. Mỗi run chọn best theo overall Validation Macro F1; lấy Tail Macro F1 tại checkpoint đó.
4. Select theo mean tail score ba seeds; exact tie chọn smaller strength. Khóa s*, không test unselected candidates hoặc retune ở E3.
5. Verify/reuse cả ba selected executions bằng official binding downstream, không sửa candidate checkpoints. Nếu không đủ verification, fresh cả ba official E1E2 runs, không rerun riêng seed thấp.
6. Tạo references/bank/mining/cache rồi final A1/A2/A3 pools theo graph; same selected E2 config/strength. Reuse nguyên A3 pool cho A4, không generation mới.
7. Chạy đủ 12 E3 official runs từ shared initial state tương ứng seed, không warm-start từ trained Base/variant khác. A4 chỉ khác Mixup, cùng anchors/steps A3.
8. Final test chỉ frozen official best checkpoint; reused E1E2 phải bind đủ bộ ba trước test. Không thêm global policy buộc đợi mọi 18 runs trước một per-run evaluation hợp lệ; report hoàn chỉnh cần đủ evidence của bảng.
9. Aggregate đúng metrics/support policies qua ba seeds; main E0/E1E2/A4 và ablation Base/A1/A2/A3/A4 resolve cùng official artifacts. Render percentages `.2f`, std cũng nhân 100; không rounding raw values.

Với verified reuse: **24 unique full classifier executions = 9 search + 3 E0 + 12 E3**, trong đó selected three search runs là ba official E1E2 identities; chỉ **6 unselected search executions** là overhead ngoài 18 official identities. Nếu fresh path: **27 executions = 9 search + 18 fresh official**, chưa tính failed attempts/retries. Không đếm Base/A4 report aliases, Slurm retries, inference hoặc generation jobs thành classifier seeds.

## 6. Implementation phases và tasks

Mỗi task thực hiện vòng nhỏ: ghi fixture/assertions → thấy failure đúng nguyên nhân → implement trong files đã nêu → chạy relevant checks → review diff và commit nếu có Git. Không tạo empty test chỉ mirror implementation. Các commands ở đây là **planned checks**, chưa chạy trong task lập plan.

### Task 1 — Config, artifact identity và bootstrap

**Files:** `pyproject.toml`, `configs/{benchmark,methods,uit}.json`, `src/geobench/{__init__,__main__,cli,config,artifacts}.py`, `tests/test_config_artifacts.py`.

**Interfaces:** Produces `load_config`, `resolve_run`, `sha256_file`, `verify_lock`, `publish_lock` theo mục 1.2; consumers đều dùng cùng canonical/path/version rules.

- [ ] Thêm tests `test_research_override_rejected`, `test_partial_publish_not_frozen`, `test_checksum_mismatch_blocks_read`, `test_single_writer_claim`; assertions: changed batch/seed/shape không được accept, partial stage không có published lock.
- [ ] Chạy `pytest tests/test_config_artifacts.py -q`, xác nhận FAIL do contracts chưa có.
- [ ] Implement strict config validation, path containment theo explicit root, canonical hashing, atomic same-filesystem publication và one-writer claim gắn job/run identity. Không xóa output root hoặc auto-refresh mismatched locks.
- [ ] Chạy lại test file, expected PASS; kiểm tra các stage metadata không hash timestamps vào semantic identity.
- [ ] Review/commit task; source commit hoặc archive hash là input cho deployment freeze.

### Task 2 — Cleaned inventory, images và new fixed split

**Files:** `src/geobench/{data,images}.py`, `tests/test_data_images.py`.

**Interfaces:** Consumes artifact/config helpers; produces common data lock, `classifier_geometry`/`normalize_classifier` và native pad/crop functions.

- [ ] Thêm fixtures cho `n=1,2,3,9,10,15`, exact-file/pixel duplicate components, cross-label conflict và infeasible quota; assert `15 -> 11/2/2`, no cross-split hash intersection, conflict/infeasible phải raise.
- [ ] Thêm image assertions: classifier tensor `[3,384,896]`, letterbox rounding/center offsets đúng Common; native `1536×662 ->1536×664 ->1536×662`, `1030×703 ->1032×704 ->1030×703`, pixel hash gồm dimensions/RGB8 header.
- [ ] Chạy `pytest tests/test_data_images.py -q` để có failure trước implementation; implement inventory/decoded hash/split/group/support serialization đúng specs, không old cleaning/split reuse.
- [ ] Chạy lại expected PASS; trên UIT chỉ publish real data lock sau actual 49.997/124 counts, readability và exact-duplicate feasibility audit.
- [ ] Review/commit, ghi new split schema/version và common data dependencies.

### Task 3 — Pure metrics, ranking và presentation

**Files:** `src/geobench/metrics.py`, `tests/test_metrics.py`.

**Interfaces:** Produces `rank_logits`, `compute_metrics`, `aggregate_seeds`; training/evaluation/reporting cùng dùng một implementation.

- [ ] Thêm assertions trên full 124-label fixture: all logits equal → Top-5 `[0,1,2,3,4]`; exact Top-5 boundary tie lấy smaller class ID. GT `[0,0,1]`, predictions `[0,2,1]` → Macro F1 `5/6`, Balanced Accuracy `3/4`, Weighted F1 `7/9`, Accuracy `2/3`; class 2 support=0 vẫn có FP và F1/Recall null.
- [ ] Thêm group-outside-FP case GT `[0,1]`, predictions `[0,0]`, group `{0}` → group F1 `2/3`, không 1; seeds `[0.1,0.2,0.3]` → raw mean `0.2`, sample std `0.1`, render `20.00 ± 10.00` trong cột `%`.
- [ ] Chạy `pytest tests/test_metrics.py -q` để thấy failures; implement đúng metric set, no mask, N/A vs incomplete và full-precision ranking/aggregation.
- [ ] Chạy lại expected PASS; missing seed phải block aggregate, không average subset hoặc pooled confusion matrix.
- [ ] Review/commit; freeze EVAL-D01/D02 vào shared evaluator config trước first validation.

### Task 4 — UIT environment/model/deployment preflight

**Files:** `src/geobench/preflight.py`, `configs/uit.json`, relevant assertions trong `tests/test_config_artifacts.py` và `tests/test_slurm_contract.py`.

**Interfaces:** Consumes data/config locks và role descriptors; produces deployment reports và verified runtime/model snapshots, không official checkpoints. Inspection/snapshot staging có thể thực hiện sớm; final GPU probes gọi **chính modules Tasks 5–8**, không copy implementation. Probe execution/lock closure được hoàn tất sau khi các providers có sẵn, trước P4 tuning/main runs.

- [ ] Thêm assertions invalid profile/QoS/GRES mapping bị reject, unknown CPU/RAM/root/walltime không trở thành silent defaults; classifier research settings không đổi khi profile chuyển mig3→mig7.
- [ ] Run relevant tests để có failure rồi implement read-only inspection/probe orchestration; model downloads/cache access theo immutable revisions, không secrets trong logs.
- [ ] Trên UIT, inspect existing env và site/storage inputs sớm; sau khi Tasks 5–8 cung cấp real workload paths, thực hiện GPU checks mục 3 bằng Slurm allocations. Actual memory/throughput, paths/quota và max walltime phải được ghi thành evidence, không giả định.
- [ ] Materialize SD1.5/DINOv3 snapshots, processor identity và all-prompt token audit; overflow/access failure block đúng stage, không tự sửa prompts hoặc dùng fallback model.
- [ ] Freeze deployment/shared runtime chỉ sau checks pass; reset/hủy probe state. Review/commit source, giữ machine-specific reports dưới artifact root.

### Task 5 — E1 và E2 deterministic pool generation

**Files:** `src/geobench/augmentation/{__init__,traditional,diffusion}.py`, `tests/test_traditional.py`, `tests/test_diffusion_jobs.py`.

**Interfaces:** Consumes data/E0/config/model locks, image/artifact functions; produces E1 pool và three E2 candidate tail/hybrid locks, `generate_e1`/`generate_pool`.

- [ ] Thêm deterministic assertions confirmed E1 operators/ranges/order, no repeated transform; shuffle-cycle uses every source before next cycle; same `(class,slot,attempt)` across strengths giữ source/VAE/latent seeds.
- [ ] Thêm first-valid tests: rejected attempts 0/1, valid attempt 2 → accepted index 2, không execute 3/4; all five fail → shortage/no lock; holdout collision block, infrastructure retry không reset cap.
- [ ] Chạy `pytest tests/test_traditional.py tests/test_diffusion_jobs.py -q` để thấy failures; implement E1 operations và shared E2 native VAE/scheduler loop, PNG/read-back hash/QC/ledger, no semantic filtering.
- [ ] Chạy lại tests expected PASS, rồi tiny actual-model training-source generation trong deployment checks; numerical output/dimensions/config phải khớp, không dùng val/test score.
- [ ] Review/commit và freeze đủ quotas; incomplete pool không được cấp cho tuning.

### Task 6 — Shared classifier training, initialization và resume

**Files:** `src/geobench/{classifier,training}.py`, `tests/test_training_resume.py`.

**Interfaces:** Consumes loaders/images, metrics, run descriptors/locks; produces three shared initial states và per-run common checkpoint/history locks; `train_run` không chọn config theo condition.

- [ ] Thêm assertions cùng seed load cùng initial state cho mọi condition; last batch được giữ; committed anchor coverage và steps đúng pool size/common scheduler, validation RNG không làm lệch train sampler.
- [ ] Thêm uninterrupted-vs-resumed fixture: model/optimizer/scheduler/RNG/history sau resume khớp uninterrupted trong deterministic fixture; interrupted epoch replay, incompatible identity bị reject, exact validation-score tie chọn earliest epoch.
- [ ] Chạy `pytest tests/test_training_resume.py -q` để có failures; implement full common config, validation calls shared metrics và atomic last/best state đủ recover auxiliary outputs.
- [ ] Run tests expected PASS; smoke short fixture trên GPU chỉ trong Slurm, không thay official epochs để gọi đó là official run. Exit success chỉ sau full declared schedule/lock.
- [ ] Review/commit; runtime/initialization locks phải có trước official E0 hoặc tuning jobs.

### Task 7 — CADS/CCFG/Geo sampling realization

**Files:** `src/geobench/augmentation/guidance.py`, extend `augmentation/diffusion.py`, `tests/test_guidance.py`.

**Interfaces:** `predict_step` consumes current latent/timestep/original embeddings/variant/stream identities; produces final noise tensor và per-step diagnostics, shared sampler gọi scheduler.step một lần.

- [ ] Thêm assertions Eq. (1)–(11): N>1 normalized endpoints 1/0, N=1→0, γ endpoints; embedding `[1,3]` có population std 1; zero/invalid std fail, no epsilon; Gaussian draw identity mới mỗi timestep và independent branches.
- [ ] Thêm fixture `ε_baseline=0`, raw positive `[1,1]`, w=2 → CFG positive `[2,2]`, **d_pos=8**, không raw distance 2 hoặc mean 4. Với τ_hat=0, `w_pos=2` và `w_neg=-2`; interpolation đúng 0.8 CADS +0.2 CCFG.
- [ ] Run `pytest tests/test_guidance.py -q` để thấy failures; implement stateless formulas, paired final source/latent streams, separate CADS/reference streams, A3 baseline replacement/prompt audit và per-timestep JSONL.
- [ ] Run tests expected PASS; validate A1/A2/A3 flags/raw-branch roles/scheduler call count trong shared sampler; preserve master prompt bytes. Numerical/overflow failure phải block đúng policy.
- [ ] Review/commit, ghi upstream static comparison từ E3 G08; không sửa Geo equations vì upstream code khác.

### Task 8 — A2 reference/mining, cross-role QC và A4 Mixup

**Files:** `src/geobench/augmentation/{mining,mixup}.py`, extend `diffusion.py`, `tests/test_mining_mixup.py`.

**Interfaces:** `build_references`/`mine_confusers` produce bank/feature/confuser locks; `mix_batch` giữ anchor identity/counts/labels, trả provenance cho trainer.

- [ ] Thêm assertions reference quota 20 cho cả 124 classes, 20 distinct sources khi đủ/cover-cycle khi thiếu; real embeddings không vào centroid; singleton reference-only. FP32 exact similarity tie chọn smallest other class ID, zero norm fail.
- [ ] Thêm G09 tests: reference candidate trùng frozen E2 final bị reject/retry reference slot, E2 lock không đổi; A1/A2/A3 final trùng bank bị reject trong cùng 5-attempt cap; bank required cho final acceptance nhưng A1 không load DINO/confusers.
- [ ] Thêm Mixup assertions uniform no-replacement batch subset đúng floor(0.5L), partner cùng class/replacement, 21 lambda endpoints, mix trước normalization, chỉ eligible synthetic bị thay; labels/anchors/optimizer steps và A3 pool hash không đổi.
- [ ] Run `pytest tests/test_mining_mixup.py -q` để thấy failures; implement frozen pinned DINOv3/official processor FP32 batch32, `.npy`/row mappings/caches và independent mixing RNG/checkpoint state; run lại expected PASS.
- [ ] Review/commit; actual feature/model/token verification V01/V02 và JSONL/array artifacts V04 vẫn cần server evidence, không dùng fixtures thay real reference bank lock.

### Task 9 — Strength search, official binding và frozen method pools

**Files:** `src/geobench/cli.py`, `config.py`, `artifacts.py`, relevant tests trong `tests/test_config_artifacts.py` và `tests/test_evaluation_reporting.py`.

**Interfaces:** Consumes candidate run/selection/pool locks; produces `selection_lock.json`, E1E2 official bindings và A4→A3 pool reference; không mutate candidate checkpoints.

- [ ] Thêm tests missing một trong nine runs chặn selection; mean của đủ ba seed Tail F1 quyết định winner, exact tie chọn smaller strength; overall best checkpoint vẫn authoritative.
- [ ] Thêm assertions promotion verify all three selected executions/manifest/config/init/history hashes; mismatch block/fresh-three-seed path; alias binding không rewrite source identity hoặc tạo dependency cycle.
- [ ] Chạy tests để có failures; implement selection/binding/frozen job descriptors, official matrix và source-run provenance, A4 reuse canonical A3 manifest.
- [ ] Run relevant tests expected PASS; counts 18 official identities, verified path 24 executions/fresh path 27, Base/A4 aliases không nhân đôi counts.
- [ ] Review/commit; nguồn generation/strength/pool nào đổi phải version mới, không refresh lock cho hợp output.

### Task 10 — Slurm templates, stage submission và recovery

**Files:** `scripts/slurm/{cpu.sbatch,gpu.sbatch,submit.sh}`, `tests/test_slurm_contract.py`, CLI descriptor handling.

**Interfaces:** Consumes verified `uit.json` và frozen `deployment/jobs/<stage>.json` manifests; produces scheduler job-ID ledger dưới persistent root. App commands vẫn verify own locks.

- [ ] Thêm assertions index→run mapping ổn định, profile/QoS/GRES đúng table, unknown walltime/CPU/RAM reject, incomplete/resumable stage không release success; duplicate submission không được hai writers.
- [ ] Implement thin templates/wrapper theo mục 4; planned shell check `bash -n scripts/slurm/cpu.sbatch scripts/slurm/gpu.sbatch scripts/slurm/submit.sh` expected zero errors, không run GPU ở login.
- [ ] Run `pytest tests/test_slurm_contract.py -q` expected PASS và kiểm tra scheduler parser/site flags trên UIT bằng harmless validation/smoke job, không official training.
- [ ] Verify signal forwarding/timeout/requeue với fixture job trên compute node: persistent checkpoint được giữ, restart cùng descriptor, downstream không consume incomplete lock; re-submit dependencies dùng current job IDs.
- [ ] Review/commit; submit đúng maximum effective walltime và role profiles đã khóa, không đổi common settings để vượt queue constraints.

### Task 11 — Evaluation/reporting và final protocol audit

**Files:** `src/geobench/{evaluation,reporting}.py`, `tests/test_evaluation_reporting.py`.

**Interfaces:** Consumes shared metrics và selected official checkpoint/bindings; produces evaluation/report locks theo Evaluation Spec, không train hoặc chọn lại checkpoints.

- [ ] Thêm assertions missing/duplicate prediction IDs, wrong labels/row mapping/non-finite logits bị reject; raw matrix đủ124×124/supports và `.npy` logits khớp CSV Top-1/Top-5.
- [ ] Thêm tests report aliases Base=E1E2/A4 shared evidence, percentages `.2f` nhưng raw full precision, N/A khác incomplete; no unselected tuning test rows hoặc two-seed aggregate.
- [ ] Run `pytest tests/test_evaluation_reporting.py -q` để có failures; implement inference/export/coverage/metric locks và main/ablation/per-class/CM/report outputs đúng schemas.
- [ ] Run lại expected PASS; `pytest tests -q` là regression gate sau integrated change, không chạy lặp vô cớ khi không có thay đổi/failure mới.
- [ ] Review/commit; chỉ sau actual training/evaluation mới populate numeric results và acceptance evidence, không dùng table template làm kết quả.

## 7. Triển khai theo phases và gate

| Phase | Tasks | Gate để đi tiếp |
|---|---|---|
| P0 — Contracts/data/metrics | 1–3 | Relevant fixture tests pass, strict config/locks và evaluator policy cố định |
| P1 — Runtime/generation/trainer | 4–6 | Initial UIT/env/storage inspection và staging evidence; data/E0/E1/E2 mechanics và trainer/resume fixture conformance. Task 4 final GPU probes chờ real providers đầy đủ |
| P2 — E3 realization | 7–8 | Eq./RNG/Mixup tests pass; DINOv3 access/processor và prompt/token audit có evidence/status cụ thể |
| P3 — Binding/execution/reporting | 9–11, đóng Task 4 probes | Integration tests pass, GPU probes dùng actual shared code; source/runtime/initial-state readiness được khóa trước scientific runs tương ứng |
| P4 — Benchmark execution | Stage graph + mục 5 | Stage locks theo dependencies, đủ official/tuning evidence, không mutate configs |
| P5 — Final acceptance | Evaluation/reporting | Đủ18 official per-seed outputs, main/ablation aliases đúng, full report/evaluation locks |

Tasks 5–8 có thể dùng dependency fixtures trong local tests trước actual GPU preflight, nhưng không publish scientific artifacts khi verification chưa xong. Nếu phát hiện code change sau runtime freeze, cập nhật source identity có version/trace rõ trước các runs liên quan, không trộn silently vào cùng comparison.

## 8. Checkpoint/resume và artifact management chi tiết

- **Classifier:** commit last checkpoint cuối mỗi completed epoch, chứa model/optimizer/scheduler/progress/best state/history/Python/NumPy/CPU-CUDA/DataLoader và Mixup RNG. Mid-epoch timeout khôi phục last committed epoch, không partial replay từ model weights đơn lẻ.
- **Generation:** ledger giữ slot/attempt/source/noise identities, first-valid decision và failed/infrastructure retries; files tạm không tính quota. Slot đã accepted/hash-valid không generate lại; retry không reset five-attempt cap.
- **Mining:** bank/features/centroids/confusers immutable sau freeze; resume extraction từ verified rows/shards hoặc rebuild cùng identity trước publication, không publish matrix thiếu row mapping.
- **Evaluation/report:** coverage-complete logits/predictions/metrics mới được lock; report recompute từ frozen outputs. Giữ candidate namespace bất biến, official binding là downstream sidecar.
- **Atomicity/ownership:** một writer/pool/run; outputs ghi trong staging cùng filesystem, flush/atomic replace rồi publish lock. Claims lưu Slurm/job identity và được kiểm tra trước recovery, không tự phá claim chỉ vì local PID không tồn tại.
- **Persistence:** checkpoints, JSONL diagnostics, NumPy arrays, candidate histories, logs, selection/bindings và source/env/config snapshots nằm dưới new persistent artifact root; không phụ thuộc session SSH hoặc `/tmp` trên node.
- **Storage:** shared real/E1 pools và A3→A4/official bindings dùng references, không duplicate files để đổi tên condition. Không xóa unselected tuning evidence hoặc rejected ledgers để làm mất provenance. Mọi archival/cleanup riêng phải được yêu cầu rõ, không là side effect của plan.
- **Completion:** Slurm JobID/exit code giúp vận hành nhưng không thay hash/dependency/coverage verification. Hash mismatch/data conflict/numerical failure phải block đúng stage; không sửa manifest/seed/model cho score tốt hơn.

## 9. Tiêu chí hoàn tất implementation và bàn giao

Implementation hoàn tất khi từng task có relevant checks pass, all required UIT verification evidence đã ghi, confirmed spec invariants không đổi và benchmark có thể resume từ committed state. Benchmark hoàn tất chỉ khi **actual** đủ18 official runs/evaluations, nine-run strength evidence/reuse accounting và final report locks hợp lệ; code/tests pass không thay thế scientific execution.

Bàn giao gồm source/version/environment snapshots, verified UIT deployment profile/max-walltime evidence, job manifest/ID ledger, persistent artifact root index, all spec acceptance evidence và reports. Các constraints chưa giải quyết phải ghi cụ thể thay vì claim success. Task hiện tại chỉ finalizes Evaluation Spec và viết plan này; không execute các bước trên.
