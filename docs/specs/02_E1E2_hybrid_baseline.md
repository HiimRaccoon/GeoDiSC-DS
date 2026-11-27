# 02 — E1 + E2 Hybrid Augmentation Baseline Specification

> **Workflow (2026-11-28):** Implementation completion means the production implementation exists and required local unit/integration tests pass, including fixture or mock tests where needed. Deployment verification means actual UIT environment, data, models, resources, recovery and shared-code runtime checks have passed with evidence. Scientific execution completion means the required official runs, evaluations and report locks exist and verify. Pending deployment evidence never blocks writing or locally testing independent implementation; it continues to gate the corresponding real artifact publication, jobs and scientific stages. Full scientific execution requires separate user approval. Confirmed research requirements and acceptance rules are unchanged.


**Phiên bản:** 1.2\
**Ngày:** 2026-11-26\
**Official condition ID:** `E1E2`\
**Phạm vi:** E1 Traditional Augmentation, E2 Base Diffusion và hybrid classifier baseline.\
**Trạng thái:** D01–D10 đã CONFIRMED theo xác nhận của người dùng, gồm revisions D02/D09/D10. Model/environment snapshots vẫn phải được resolve và khóa trước thực thi. Chưa sinh ảnh, tuning hoặc huấn luyện.

## 1. Mục tiêu và nguồn yêu cầu

E1 và E2 là hai module bổ sung dữ liệu cho hai nhóm thiếu ảnh khác nhau, tạo thành **một official training condition**. So sánh E0 với E1E2 trả lời RQ1 về hiệu quả augmentation trên phân bố long-tail. So sánh E1E2 với E3-A4 trả lời RQ2 về hiệu quả của proposed method trên extreme-tail.

Nguồn có thẩm quyền:

- [Markdown tổng thể](../GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), Section IV và VII.1.
- [Common Spec v1.3](00_common.md): fixed split, singleton, shared pools/budgets/strength, classifier/runtime, metrics, reproducibility, checkpoint/resume, verified reuse và artifact locking.
- [E0 Spec v1.1](01_E0_original.md): định nghĩa real training pool chuẩn và artifact identity để các condition dùng chung.

Các cấu hình dùng chung được kế thừa nguyên vẹn, không có E1E2 classifier override. Chỉ ba cơ chế được phép kế thừa từ repository cũ theo Common Spec; không import DA-Fusion configuration, QC thresholds, encoding defaults hoặc generation policy cũ.

**E2 là Base chính thức của E3 ablation.** Kết quả classifier E1E2 sau khi chọn strength được dùng trực tiếp làm Base; không có official condition E2-only, E3-A0 hoặc một lượt Base chạy lại riêng cho E3.

Đã đối chiếu Section IV/VII.1 của Markdown tổng thể và Common Spec hiện tại. Phần cấu hình tại mục 2–4 giữ đúng các giá trị nguồn; mục 11 ghi các quyết định triển khai đã được người dùng xác nhận. Không thay nghiên cứu, quota, base prompt hoặc các thông số đã khóa trong master Markdown. Xác nhận này chỉ chốt specification; trạng thái implementation và experiments được theo dõi riêng trong tasks.md.

## 2. Training membership và quota

### 2.1. Thành phần của hybrid pool

```text
Locked E0 real training pool ───────────────────────────┐
                                                      │
Medium training originals → E1 traditional outputs ────┼→ E1E2 training pool
                                                      │
Eligible tail originals  → E2 final synthetic ─────────┘

Validation / test: nguyên real splits của Common Spec
```

Gọi `R` là locked E0 real pool, `A_E1` là pool traditional đã khóa, `A_E2(s*)` là pool Base Diffusion ở selected strength:

```text
T_E1E2 = R ∪ A_E1 ∪ A_E2(s*)
```

Các tập này là membership theo sample IDs, không phải thao tác deduplicate theo content. Mỗi output có ID riêng và source provenance; mọi real anchor giữ nguyên ID, label, path và hashes từ E0. Không quét thư mục để tự thêm candidate/rejected files vào training.

| Nhóm theo locked counts | Target tổng training samples/lớp | Số ảnh bổ sung | Module |
|---|---:|---:|---|
| Head: `n_train >= 100` | `n_train` | 0 | Giữ real pool |
| Medium: `20 <= n_train <= 99` | 100 | `100 - n_train` | E1 |
| Extreme-tail: `n_train < 20`, `n_original >= 2` | `min(100, 5*n_train)` | `target - n_train` | E2 |
| Singleton: `n_original = 1` | 1 | 0 | Giữ một real anchor |

Ví dụ quota E2: `n_original=2 hoặc 3, n_train=1 → 4 synthetic`; `n_train=5 → 20 synthetic`; `n_train=19 → 76 synthetic`. Singleton có `n_train=1` nhưng không được áp quota 4 synthetic.

Quota là số **final outputs được nhận vào pool**, không phải số generation attempts. Thiếu quota ở một lớp chặn freeze; không bù bằng lớp khác, ảnh real oversampling hoặc giảm target. Counts/groups không được tính lại từ hybrid pool.

### 2.2. Source và holdout isolation

- E1/E2 chỉ lấy source từ real `train.csv` đã khóa, đúng nhóm được chỉ định. Không dùng synthetic làm source tiếp.
- Không dùng real validation/test cho source selection, prompt adaptation hoặc QC threshold tuning.
- Validation được dùng cho strength selection đúng mục 6, không phải để chọn từng ảnh synthetic theo performance.
- Không sinh final training augmentation cho head hoặc singleton, không thêm references vào hybrid membership.
- Không chỉnh sửa hoặc ghi đè cleaned originals; mọi augmentation là artifact mới.

## 3. E1 — Traditional Augmentation

### 3.1. Phạm vi và các phép biến đổi đã chốt

E1 chỉ bổ sung medium classes đến quota ở mục 2. Mỗi candidate chọn ngẫu nhiên `k ∈ {1,2}` transforms **không lặp** từ đúng tập sau:

| Transform | Tham số đã chốt |
|---|---|
| Brightness | Factor `Uniform(0.90, 1.10)` |
| Contrast | Factor `Uniform(0.90, 1.10)` |
| Gaussian blur | `sigma ~ Uniform(0.10, 0.60)`, kernel `5 × 5` |
| Gaussian noise | `sigma ~ Uniform(1,3)/255` |
| Rotation | Angle `Uniform(-3°, +3°)`, bilinear interpolation, reflection padding |

Không thêm gamma, JPEG variation hoặc operators ngoài bảng. Không horizontal/vertical flip, rotation lớn, aggressive crop, strong affine/perspective hoặc strong hue shift. Mục tiêu là giữ traffic side, road geometry, signage và geographic cues.

### 3.2. Candidate execution và provenance

Luồng E1: locked medium source → decode RGB → sample transforms/parameters → áp transforms → lưu output → kiểm tra artifact/QC → candidate decision.

Output giữ kích thước nguồn; không tạo classifier-sized pool trước rồi dùng pool đó cho diffusion. Việc resize/center letterbox cho classifier chỉ thực hiện theo Common Spec khi train/evaluate. Sampling/order và rotation canvas/numerical semantics tuân theo D01/D03 đã xác nhận, không thêm biến đổi ngoài tập cho phép.

Mỗi candidate lưu source ID/hashes, class/group, candidate/job ID, actual seed, `k`, transform IDs, sampled parameters, **actual transform order**, processing/encoding config hash, output size/file/pixel hashes và QC decision/reasons. Log này phải đủ để phân biệt outputs khác nhau dù cùng source.

Master chốt tập `k`, operators và parameter ranges; xác nhận D01 bổ sung phân phối đều và thứ tự áp cố định. Không thay các quyết định này theo training seed, strength hoặc performance.

### 3.3. Reuse E1

Sinh và khóa **một** E1 pool cho benchmark version, dùng nguyên membership/content cho mọi strength candidate, official E1E2 và E3-A1–A4, trên cả ba official training seeds. Không tạo E1 pool khác để tối ưu một strength hoặc một ablation.

## 4. E2 — Base Diffusion

### 4.1. Configuration đã chốt

| Tham số | Yêu cầu E2 |
|---|---|
| Backbone | Stable Diffusion v1.5 |
| Generation mode | Image-conditioned / img2img |
| Solver | DPM-Solver++ |
| Configured inference steps | 50 |
| CFG guidance `w` | 2.0 |
| Input/output geometry | Native cleaned resolution, reflection-pad tối thiểu `/8`, crop đúng padding |
| Generation batch size | 1 source/lần |
| Textual Inversion / LoRA | OFF / OFF |
| CADS / CCFG | OFF / OFF |
| DINOv3 / confuser mining | OFF / OFF |
| Geo-preservation | OFF; chỉ dùng base prompt dưới đây, không Geo-aware prompt/loss/filter |
| Mixup | OFF trong hybrid classifier training |
| Tham số được tuning | Chỉ strength, đúng grid `{0.25, 0.35, 0.50}` |

Base prompt được giữ nguyên, chỉ thay `{country_name}` bằng class name tương ứng mapping:

```text
A realistic street-level road photograph in {country_name}, look like picture in Google Street View.
```

Không thay bằng Geo-aware prompt hoặc artifact negative prompt của A3, không nối confusing country vào negative prompt. D06 đã xác nhận negative/unconditional prompt là empty string, không dùng prompt của repo cũ.

Scheduler realization, generation FP32/AMP OFF và conditioning được xác nhận tại D04–D06; concrete SD1.5 snapshot/library identity phải được materialize và khóa trước generation. Generation precision là quyết định D05 độc lập, không phải suy từ classifier runtime.

### 4.2. Native-resolution generation

```text
Real training source → RGB native → reflection-pad /8
        → VAE encode → add noise theo candidate/selected strength
        → SD1.5 img2img denoising với DPM-Solver++
        → VAE decode → crop đúng padding → output native → QC
```

| Cleaned source W × H | Padded W × H | Final output W × H |
|---|---|---|
| `1536 × 662` | `1536 × 664` | `1536 × 662` |
| `1030 × 703` | `1032 × 704` | `1030 × 703` |

Offsets/crop bbox theo Common Spec mục 5.2; không resize source về `896 × 384` hoặc canvas vuông. Configured 50 steps không có nghĩa mọi strength đều có 50 active denoising steps. Log actual scheduler timesteps, active step count, starting timestep, strength và scheduler config để tái lập chính xác.

### 4.3. Generation identity và resume

Khóa split/source pool, source-selection policy, job-seed derivation, prompt/tokenizer, model weights/revisions, VAE, scheduler config, generation precision/backend và output encoding trước khi sinh candidate pools. Chỉ strength được khác giữa các candidate configs; những settings còn lại phải giống nhau.

Mỗi job có identity ổn định gồm method, class, output slot/attempt, source và strength khi áp dụng. Actual seed suy ra từ augmentation seed đã khóa trong Common Spec và job key được xác nhận ở D02; không dùng training seed hoặc wall-clock time. Lưu source order và actual seeds để kiểm tra so sánh strengths không bị confound bởi policy khác nhau.

Generation resume thực hiện theo completed jobs: chỉ reuse output khi identity, file/pixel hashes và recorded QC decision khớp. Output dang dở không được tính quota hoặc publish. Lỗi hạ tầng không tự đổi source, noise seed hoặc settings cho cùng job; retry/resume được ghi history. Sau technical/duplicate reject, tạo attempt kế tiếp theo D08/D09 đã xác nhận, tối đa 5 attempts/output slot.

Khi lỗi runtime/OOM, áp common failure/versioning policy; không tự đổi các thông số generation đã khóa để hoàn tất riêng một candidate hoặc variant.

## 5. Generation và QC policy

### 5.1. Candidate lifecycle và quota accounting

```text
planned → generated → persisted/read-back → audited → accepted hoặc rejected
                                           │
                                           └→ blocked/error nếu vi phạm identity/leakage

Chỉ accepted outputs có trong frozen manifest mới là training members.
Rejected, blocked, incomplete và reference outputs không được thêm vào pool.
```

Ghi candidate ledger append-only về mặt lịch sử; việc publish summary/lock dùng atomic protocol của Common Spec. Mỗi output slot nhận candidate đầu tiên pass technical/duplicate checks trong tối đa 5 deterministic attempts, rồi dừng slot đó. Không chọn một tập con đẹp hơn bằng validation/test score, best-of-N hoặc semantic ranking. Dừng bổ sung membership khi đủ quota; hết attempts vẫn thiếu thì ghi shortage và chặn freeze.

### 5.2. Kiểm tra tính hợp lệ bắt buộc

| Kiểm tra | Yêu cầu trước pool freeze |
|---|---|
| Source provenance | Source ID và file/pixel hashes khớp real train, đúng class/group; không holdout hoặc synthetic source |
| Geometry/content format | Output decode được thành RGB, size khớp source; diffusion crop đúng bbox; không có tensor NaN/Inf trước lưu |
| Manifest integrity | Candidate ID/path duy nhất, file tồn tại, actual settings/seed đầy đủ, file/pixel hashes tính từ output đã lưu và đọc lại |
| Membership/quota | Real subset bằng E0, E1/E2 đúng lớp/quota; singleton không có final augmented output |
| Holdout collision | Audit output file SHA-256 và decoded RGB pixel hash với real validation/test; collision chặn freeze và ghi incident |

Hash RGB theo canonical representation/backend đã khóa ở Common Spec. Holdout collision audit chỉ kiểm tra exact equality để ngăn leakage, không dùng semantic similarity hoặc ảnh holdout để xếp hạng chất lượng candidate. Incident không được dùng để tối ưu prompt/generator theo nội dung test.

Output vi phạm invariant không thể accepted. Lỗi identity/leakage phải chặn publication; không tự sửa lock, relabel hoặc đổi membership thật để tiếp tục. Technical-invalid hoặc training-duplicate candidate được reject và thử attempt tiếp theo theo D08/D09; holdout collision vẫn block, không tự regenerate dựa trên holdout.

### 5.3. QC policy đã xác nhận

Nguồn mới chưa quy định một quality-filtering policy hoặc retry cap. Spec này **không thêm** semantic/geographic filtering, blur/blank thresholds, similarity cutoffs, manual review để chọn membership hoặc human approval rate.

D08 đã xác nhận **technical integrity + exact-duplicate audit**, không có quality/semantic gates. D09 cho phép tối đa 5 deterministic attempts/output slot và nhận candidate hợp lệ đầu tiên; các technical invariants bắt buộc ở mục 5.2 vẫn giữ nguyên. Áp cùng policy cho mọi E2 strength candidates, không âm thầm thay giữa tuning và official reuse.

DINOv3 vẫn OFF trong E2, không dùng DINO diagnostics/filter từ repo cũ. Không đưa geographic loss/filter hoặc A3 prompt vào E2 dưới tên QC. Những thay đổi đó sẽ làm sai Base và không thuộc phạm vi spec này.

QC/encoding/retry policy đã confirmed tại D07–D09; vẫn phải materialize config/snapshots, hoàn thành audits và đủ quota mới được freeze pool dùng cho tuning/official training.

## 6. Strength tuning

### 6.1. Thiết kế so sánh đã chốt

Tạo ba candidate tail pools tương ứng grid `s ∈ {0.25, 0.35, 0.50}`. Với mỗi candidate, classifier training membership là:

```text
T_tuning(s) = cùng E0 real pool + cùng frozen E1 pool + E2 tail pool(s)
```

Mỗi pool dùng cùng per-class targets, source eligibility, prompt, non-strength settings và QC policy. Không train một classifier chỉ trên tail rồi gọi score đó là kết quả hybrid baseline. Tuning classifier vẫn dùng toàn bộ label space và shared runtime/initialization/preprocessing/checkpoint policy của Common Spec, bao gồm center letterbox `896 × 384`; không có classifier settings riêng để chạy search nhanh hơn.

Tuning namespace/provenance tách khỏi official bindings. Search có 3 strengths × 3 common seeds = **9 candidate classifier runs**; cả ba runs ở selected strength được phép verified reuse làm ba official E1E2 identities theo mục 6.4 và Common Spec mục 9.3. Unselected runs không là official results; không đếm selected executions hai lần.

### 6.2. Metric và checkpoint selection

Metric lựa chọn strength là **Tail Macro F1 trên fixed real validation**:

```text
C_tail_val = {c : n_train(c) < 20 và support_val(c) > 0}
Tail Macro F1 = mean(F1_c for c in C_tail_val)
```

F1 từng lớp lấy từ confusion counts trên **toàn validation split**, không lọc samples hoặc logits về tail. Singleton không có validation support không vào mẫu số; lớp không support ghi N/A theo Common Spec.

Checkpoint của mỗi tuning classifier vẫn được chọn bằng **overall Validation Macro F1** và earliest-epoch tie-break của Common Spec. Từ checkpoint đó lấy Tail Macro F1 để so sánh strength. Không đổi thành chọn epoch tốt nhất theo tail metric, vì như vậy sẽ override common checkpoint policy.

D10 đã xác nhận dùng cả ba common training seeds `[42,123,2026]` cho mỗi strength. `score(s) = mean(Validation Tail Macro F1(s, seed))` qua đúng ba seeds, tại common best checkpoint của từng run. Chọn strength có mean cao nhất; hòa chính xác ở giá trị chưa làm tròn thì chọn strength nhỏ hơn. Không dùng test, pooled predictions hoặc chọn seed tốt nhất. Nếu `C_tail_val` rỗng hoặc chưa hoàn tất đủ 9 required runs, dừng thay vì chọn mặc định hoặc tính mean trên subset.

### 6.3. Lock selected strength và official transition

Sau khi đủ bằng chứng theo tuning protocol đã xác nhận:

1. Lưu score từng candidate/run, selected checkpoint hashes, evaluated tail class IDs/support và aggregation/tie-break rule.
2. Chọn duy nhất `s*` theo validation protocol đã khóa, lưu selection artifact với candidate pool/config hashes.
3. Khóa selected strength và tham chiếu **nguyên E2 candidate pool đã đánh giá** ở `s*` làm official tail pool; không regenerate một pool khác sau khi biết winner.
4. Publish hybrid pool lock cho `R + A_E1 + A_E2(s*)`, xác minh và bind cả ba selected-strength executions làm official E1E2 runs theo mục 6.4; nếu không thể reuse đầy đủ, ghi lý do và chạy fresh cả ba từ shared initial states, không fine-tune tiếp tuning checkpoints.

Selected strength được reuse cho E3-A1–A4 và A2 reference generation, không chọn lại theo variant/training seed. Tuning protocol hoặc generation policy đổi sau khi đã quan sát scores phải có version mới và ghi đầy đủ; không ghi đè search history hoặc pool lock cũ.

### 6.4. Verified reuse và official binding

Áp Common Spec mục 9.3: xác minh candidate/official membership và canonical manifest hash giống nhau, cùng shared config/environment/code/model, seed/initial state, full schedule/traversal/steps và common best-checkpoint selection, chưa evaluate test. Candidate hybrid manifest phải được freeze trước tuning và có thể được tham chiếu nguyên vẹn sau selection; không thêm phase/selection columns vào manifest để làm đổi membership hash.

Mỗi `runs/E1E2/seed_<seed>/official_binding.json` tham chiếu immutable source run và checkpoint/history hashes, kèm selected-strength lock, official pool lock, verification evidence. Không sửa candidate checkpoint identity hoặc giả lập một lần training mới. Selection hash chỉ nằm ở binding downstream, không ghi ngược vào candidate run/config đã khóa; candidate dependencies vì vậy không tạo vòng với selection.

Bind cả ba seeds như một bộ, mỗi official identity đúng một execution. Official metadata phải phân biệt `reused` với `fresh` và lưu source identity. Nếu verification thất bại, không tiếp tục test hoặc chọn rerun seed theo score; dùng fresh-three-seed path ở Common Spec. Khi reuse đủ ba runs, tổng benchmark cần 24 unique training executions: 9 search + 15 official E0/E3 executions; 18 official identities vẫn giữ nguyên, 6 unselected executions là overhead tuning. Retry/failure không đổi run accounting này.

## 7. Official E1E2 classifier execution

Official membership chỉ lấy từ frozen hybrid manifest. Mọi epoch duyệt mỗi anchor đúng một lần theo common sampler; không oversample real/synthetic hoặc điều chỉnh tỷ lệ batch theo nhóm. Không online random augmentation, Mixup/CutMix, class-weighted loss hoặc weighted sampler.

Kế thừa nguyên classifier config, FP32/AMP OFF, batch size, seeds, training schedule, metrics và best/last resume contract của Common Spec. Input là RGB `896 × 384` qua center letterbox giữ aspect ratio, tensor `[B, 3, 384, 896]`; không thay SD1.5 native-resolution input bằng classifier canvas. Pool size là size hybrid thực tế; scheduler progress dùng steps/epoch suy ra từ size này, không dùng steps của E0 hoặc cố bù để equal-compute.

Real validation/test giữ nguyên manifests của E0. Official run dùng cùng shared initial state với E0/E3 ở cùng seed. Training completion, epoch-boundary recovery, failed attempts, checkpoint freeze và final evaluation theo common contract; không retry chọn seed có score tốt hơn.

Kết quả official E1E2 qua đủ ba seeds là **Base (E2)** trong bảng E3 ablation. Đây là reuse kết quả classifier hybrid, không phải một kết quả classifier E2-only.

## 8. Artifact dependencies và reuse policy

```text
Common data/runtime locks + shared initial states + E0 real pool lock
          │
          ├→ E1 resolved policy → generate/QC → frozen E1 pool ──────────┐
          │                                                           │
          └→ E2 resolved non-strength policy + tuning protocol         │
                       │                                              │
                       ▼                                              │
                E2 candidate pools cho 3 strengths                     │
                       │                                              │
                       └────────── + E0 + cùng E1 ──────────────────────┘
                                      │
                                      ▼
                              Tuning classifiers
                                      │ real validation
                                      ▼
                           Selected-strength lock
                                      │
                     E0 + E1 + selected E2 pool
                                      │
                                      ▼
                           Official E1E2 pool/run locks
                                      │
                                      ▼
                       Base result reused trong E3 ablation
```

Shared runtime không phụ thuộc selected-strength lock; phải khóa trước tuning để tránh vòng dependency. E0 real pool lock phải có, nhưng không cần E0 classifier runs hoàn tất để sinh E1/E2 hoặc tuning.

| Artifact | Bên dùng lại | Ràng buộc |
|---|---|---|
| E0 real pool | Tuning, official E1E2, mọi E3 variant | Cùng IDs/labels/file và pixel hashes |
| Frozen E1 pool | Mọi strength, official E1E2, E3-A1–A4 | Cùng actual files/membership, không sinh lại |
| Per-class tail budgets | E2 candidates và E3-A1–A4 | Cùng targets theo locked `n_train`, cùng singleton exception |
| Selected strength | Official E2, E3-A1–A4, A2 references | Một scalar `s*` và một selection lock |
| Selected E2 pool | Ba official E1E2 seeds | Reuse đúng pool candidate đã đánh giá, không sinh theo training seed |
| Selected-strength classifier runs | Ba official E1E2 identities | Verified reuse cả ba executions, immutable source evidence và official bindings theo mục 6.4 |
| Official E1E2 results | E3 Base comparison | Reuse nguyên per-seed results và checkpoint identities; không thêm A0 |

E2 synthetic files không được coi là CADS references hoặc synthetic pool của A1/A2/A3. Các E3 variants sinh outputs theo method của chúng nhưng giữ shared budgets/settings/strength đã khóa. Cached A2 bank/confusers và pool identity A3/A4 tiếp tục theo Common Spec, không được E1E2 định nghĩa lại.

## 9. Artifacts và provenance

Các đường dẫn sau tính từ một benchmark-version artifact root, là output contract khi triển khai; chưa tạo files ở giai đoạn viết spec. Nội dung filename có thể tham chiếu shared config, không sao chép thành các defaults độc lập.

| Đường dẫn / artifact | Nội dung tối thiểu |
|---|---|
| `augmentation/E1/config.json` | Resolved sampling/order/operators/backend/encoding/QC policy và references đến common identities |
| `augmentation/E1/candidates.jsonl` | Candidate ledger, source/seed/transforms/params/output hashes/decisions/retries |
| `pools/E1/manifest.csv`, `pool.json`, `pool_lock.json` | Accepted traditional membership, per-class budgets/counts, common/E0/config/QC dependencies |
| `augmentation/E2/config.json` | Resolved non-strength SD/VAE/tokenizer/scheduler/precision/prompt/encoding/QC config và model snapshots |
| `augmentation/E2/strength_<s>/candidates.jsonl` | Generation ledger theo strength, actual schedule, source/seed/output/QC history |
| `pools/E2/strength_<s>/manifest.csv`, `pool.json`, `pool_lock.json` | Frozen candidate tail pool, quota audit và dependencies riêng của candidate |
| `pools/E1E2/candidates/strength_<s>/manifest.csv`, `pool.json`, `pool_lock.json` | Frozen hybrid membership cho từng strength trước tuning; official selected pool tham chiếu chính canonical manifest này để verified reuse |
| `tuning/E2/protocol.json` | Confirmed tuning seeds/aggregation/tie-break, fixed configs/manifests/support sets; khóa trước search |
| `tuning/E2/strength_<s>/seed_<seed>/` | Immutable candidate execution evidence cho cả ba common seeds; config/metadata/history/LR/best/last/validation metrics, không final-test outputs trong tuning namespace |
| `tuning/E2/results.json` | Full-precision per-run/per-candidate scores, checkpoint refs, status và selection evidence |
| `tuning/E2/selected_strength.json`, `selection_lock.json` | `s*`, protocol/results/candidate hashes và lý do selection theo rule đã khóa |
| `pools/E1E2/manifest.csv`, `pool.json`, `pool_lock.json` | Official union membership và explicit references đến E0/E1/selected E2/selection locks |
| `runs/E1E2/seed_<seed>/` | Ba official classifier runs với artifacts theo mục 9.2 |
| `runs/E1E2/seed_<seed>/official_binding.json` | Nếu reuse: source run/artifact hashes, verification evidence, selection/pool locks và official identity; không nhân đôi physical execution |

### 9.1. Manifest và QC records

Training manifest giữ các identity fields của Common Spec và thêm `sample_type` (`real`, `traditional`, `final_synthetic`), method, source image ID/hashes khi augmented, generation job ID/seed, method config hash, output file/pixel hashes và strength cho E2. Không tái sử dụng original `image_id` cho descendant; source và output IDs là hai trường riêng.

Pool descriptor lưu total/per-class counts, required/accepted/rejected/incomplete counts, shortage, QC policy version, content hashes và dependency hashes. QC summary chứa reasons và holdout collision incidents, không chỉ số ảnh accepted cuối cùng. Actual output encoding/config phải được lưu kể cả khi model/source/seed giống nhau.

### 9.2. Official run artifacts

Mỗi official run có `config.yaml`, `run_metadata.json`, `history.csv`, `lr_steps.csv`, `best_checkpoint.pt`, `last_checkpoint.pt`, `metrics.json` và `checkpoint_lock.json`, cùng vai trò/schema kế thừa Common Spec và E0 artifact contract. Với reused run, các training artifacts có thể được tham chiếu bằng hash qua official binding thay vì sao chép/ghi đè source evidence; final-test metrics được lưu dưới official namespace. Thay condition/pool identity bằng E1E2 ở official metadata/binding; không sửa original candidate identity và không kế thừa E0 restriction “chỉ real”.

Metadata phân biệt actual real, traditional, final synthetic và reference counts trong training; reference count phải bằng 0. Lưu selected strength/selection hash, shared E1 hash, selected E2 hash, hybrid membership hash, initial-state hash và common dependencies. Validation-at-best và final-test được ghi riêng; final-test chưa evaluate thì `null`.

Checkpoint/reports phải truy ngược được đúng selected pool và shared runtime; candidate pools không được tự thay nội dung sau khi tuning/official run đã dùng chúng. Hash mismatch chặn downstream/resume theo Common Spec, không âm thầm regenerate để “khớp seed”.

## 10. Acceptance criteria

Các tiêu chí sau là hợp đồng cho triển khai sau này, chưa phải bằng chứng đã chạy.

| ID | Tiêu chí | Bằng chứng |
|---|---|---|
| HYB-AC01 | Real subset đúng toàn bộ E0, holdouts/group/singleton không đổi | Manifest identity/hash comparisons và common split lock |
| HYB-AC02 | E1 chỉ dùng đúng medium sources/operators/ranges; không lặp transform trong candidate | Resolved E1 config và candidate transform logs |
| HYB-AC03 | E1 đủ 100 total/lớp và cùng pool dùng cho mọi strength/seed/E3 variant | Per-class quotas, E1 lock và dependency hashes |
| HYB-AC04 | E2 chỉ sinh cho eligible extreme-tail, singleton có 0 final augmentation | Source eligibility audit và per-class budgets/counts |
| HYB-AC05 | SD1.5, DPM-Solver++, 50 configured steps, CFG 2.0, batch 1; các E3 components OFF | Resolved config, model hashes và actual generation logs |
| HYB-AC06 | Native pad/crop đúng Common Spec; classifier chỉ resize theo preprocessing chung | Source/output size, offsets/bbox và model-input identity |
| HYB-AC07 | Generation có source/seed/actual params/provenance, resume không thay job identity | Candidate ledger, output hashes, retry history |
| HYB-AC08 | QC được xác nhận và khóa trước generation; invalid/holdout-colliding/incomplete outputs không thể vào frozen pool | QC config/version, read-back hashes, collision/shortage audit |
| HYB-AC09 | Tuning chỉ thay strength trong grid, dùng cùng E0/E1, quotas và classifier runtime | Candidate config comparisons, protocol lock và manifests |
| HYB-AC10 | Strength selection theo Tail Macro F1 trên real validation với common support/checkpoint policy | Tuning validation outputs, evaluated tail IDs, scores và selection evidence |
| HYB-AC11 | Đủ 9 tuning runs, select theo mean Validation Tail Macro F1 qua ba common seeds, tie-break strength nhỏ hơn; không dùng test | Confirmed D10, full-precision protocol/results và candidate checkpoint identities |
| HYB-AC12 | Official pool reuse đúng selected candidate; reused executions được verified/bind đủ ba seeds hoặc cả ba official runs là fresh | Pool/selection/run locks, official bindings, verification evidence và initial-state identities |
| HYB-AC13 | Official classifier config/checkpoint/resume/evaluation kế thừa Common Spec, không mixing/rebalancing thêm | Shared runtime hash, per-epoch traversal/step/history logs và outputs |
| HYB-AC14 | E1E2 official results là Base dùng lại cho E3; không thêm E2-only/A0 runs | Official matrix và ablation result references |
| HYB-AC15 | Tail budgets/selected strength được reuse và không thay sau khi biết scores | Locked per-class targets, selected-strength dependencies và version history |
| HYB-AC16 | D01–D10 đã confirmed; resolved configs đúng các quyết định, không tự dùng defaults cũ | Decision records và fully resolved configs trước publication |
| HYB-AC17 | Không semantic filtering, quality thresholds hoặc cấu hình DA-Fusion cũ | QC/generation config audit và confirmed D01–D10 |
| HYB-AC18 | Deterministic shuffle-cycle, paired sources/noise theo slot/attempt giữa strengths; tối đa 5 attempts/slot, accept first-valid | Source schedule/stream seeds, attempt ledger, first-accepted index và shortage audit |

Không đặt ngưỡng validation/test score tối thiểu để nghiệm thu phương pháp. Kết quả thấp hoặc không cải thiện vẫn phải báo cáo, không phải lý do để sửa generator rồi ghi đè cùng benchmark version.

## 11. D01–D10 — Confirmed decisions

### 11.1. Cơ sở và trạng thái xác nhận

Master/Common Spec quy định các invariants trong cột giữa; các chi tiết bổ sung ở cột phải đã được người dùng xác nhận bằng quyết định D01–D10 dưới đây. D02/D09/D10 dùng revisions của lần xác nhận này, các mục còn lại giữ cấu hình đã đề xuất trong v1.1.

| ID | Đã chốt trong Markdown/Common Spec | Chi tiết được chốt qua xác nhận D01–D10 |
|---|---|---|
| D01 | `k ∈ {1,2}`, transforms không lặp, đúng năm operators tại mục 3 | Xác suất chọn k/operators và thứ tự áp |
| D02 | Real-train-only sources, augmentation seed riêng, deterministic job identities/actual seeds | Source scheduling và công thức job seed/pairing giữa strengths |
| D03 | Ranges, blur kernel, rotation bilinear/reflection; common decode/identity contract | Công thức transform, border/noise/clipping/quantization và rotation canvas |
| D04 | DPM-Solver++, 50 configured steps | Scheduler realization và flags |
| D05 | SD1.5 img2img, không TI/LoRA; model/environment identity phải khóa | Generation precision/backend và VAE sampling; concrete model snapshot |
| D06 | Base prompt nguyên văn ở mục 4.1 | Empty/negative text, token overflow policy |
| D07 | Output native size, file/pixel hashes sau persist/read-back | Encoding và output naming |
| D08 | Source/identity/geometry integrity, không leakage; DINOv3/Geo-preservation OFF | Duplicate handling giữa generated outputs và real train/accepted members |
| D09 | Đủ quota mới freeze; không thay budget hoặc identity khi resume | Attempt cap, retry và candidate-selection policy |
| D10 | Grid, Validation Tail Macro F1, common checkpoint/support/runtime contract, một shared selected strength | Số tuning seeds, aggregation và strength tie-break |

**D01–D10 đều có trạng thái CONFIRMED.** Các quyết định này bổ sung realization của benchmark, không thay giá trị đã khóa trong master. Trước thực thi vẫn phải materialize model/library identities và resolved configs, không tự suy rằng xác nhận specification đồng nghĩa artifacts đã được tạo hoặc kiểm tra.

### 11.2. D01 — Sampling/order E1

**CONFIRMED:** chọn đều `k=1` hoặc `2` (xác suất `0.5` mỗi giá trị); chọn đều một tập k operators không hoàn lại trong năm operators. Áp những operators đã chọn theo thứ tự cố định của bảng mục 3.1: brightness → contrast → blur → noise → rotation. Sample riêng tham số theo đúng Uniform ranges đã chốt; không thêm transform mới.

**Lý do:** không ưu tiên một operator, không thêm weights hoặc một lần random hóa order. Giữ khả năng tái lập với ít quyết định nhất.

### 11.3. D02 — Sources và seeds

**CONFIRMED — revised:** deterministic **shuffle-cycle** source selection. Trong từng class/method, bắt đầu từ sources sắp theo relative path; mỗi cycle tạo một permutation bằng RNG seed cố định theo cycle index, dùng hết permutation rồi mới sang cycle tiếp theo. Với output slot `j` zero-based, đặt `cycle = floor(j/n_train)` và `position = j mod n_train`; source của slot là phần tử position của permutation cycle đó. Không bỏ qua hoặc đổi source schedule khi strength khác hoặc một slot cần nhiều attempts.

Seed lấy từ bốn bytes đầu SHA-256 theo big-endian của chuỗi UTF-8 `geo-e1e2-v2|314159|<method>|<class_id>|<index>|<stream>`. Với shuffle, index là cycle và stream là `source_cycle`; dùng permutation không hoàn lại trên canonical source list, khóa RNG implementation/version. Với candidate attempt, index là `5*j+a`, `a ∈ {0,1,2,3,4}`; streams tách riêng `transforms`, `vae` và `latent` theo vai trò. Lưu actual source/permutation/seed trong provenance.

Một slot giữ cùng source trong tối đa 5 attempts; mỗi attempt dùng seeds riêng suy ra deterministic từ slot/attempt index. Ba E2 strengths dùng cùng source cho slot j, cùng VAE/latent seeds và cùng initial noise tensor cho cùng `(j,a)`; **không đưa strength vào source/noise-seed keys**. Strength vẫn thuộc job/output identity và namespace. Dừng sớm sau first-valid tại mỗi strength có thể tạo accepted attempt indices khác nhau; chỉ pairing theo slot/attempt được bảo đảm, không giả định các accepted outputs luôn xuất phát từ cùng noise realization. Ghi accepted attempt index để audit; không sinh thêm sau success chỉ để ép các accepted attempts bằng nhau.

**Lý do:** phân bổ sources công bằng qua các cycle đã shuffle và giữ comparison giữa strengths có thể tái lập. Retry của một slot không làm lệch sources hoặc RNG của slot kế tiếp.

### 11.4. D03 — Numerical semantics E1

**CONFIRMED:** xử lý RGB float32 trong `[0,1]`; brightness là nhân factor; contrast là `mean(x) + factor*(x-mean(x))`, với một scalar mean trên RGB pixels của input hiện tại. Blur dùng kernel/sigma đã chốt và reflection border. Gaussian noise có mean 0, sigma đã chốt, độc lập theo pixel/channel. Clip về `[0,1]` sau mỗi transform, chỉ quantize khi lưu: `floor(255*x + 0.5)`.

Rotation quanh tâm ảnh trên canvas W × H không đổi, inverse mapping với bilinear và reflection border; không expand canvas, không crop thêm một vùng cảnh hoặc chèn constant-color corners. Khóa backend và đầy đủ center/grid conventions trong resolved config để cùng phép biến đổi cho mọi outputs.

**Lý do:** chỉ cụ thể hóa operators đã có, không thêm augmentation hay parameter ranges. Geometry pad/crop của diffusion vẫn lấy nguyên Common Spec, không thuộc lựa chọn D03.

### 11.5. D04 — DPM-Solver++ realization

**CONFIRMED:** dùng `DPMSolverMultistepScheduler` với `algorithm_type="dpmsolver++"`, `solver_order=2`, `solver_type="midpoint"`, `timestep_spacing="linspace"`, `lower_order_final=true`, `thresholding=false`, không Karras/schedule conversion bổ sung. Giữ noise schedule/prediction configuration tương thích với snapshot SD1.5 đã khóa; lưu toàn bộ resolved scheduler config thay vì phụ thuộc defaults thay đổi theo version. Configured inference steps vẫn là 50.

**Lý do:** một realization DPM-Solver++ chuẩn cho conditional generation, không thêm sampler search. Các tên tùy chọn được đối chiếu từ [tài liệu Diffusers chính thức](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver); tài liệu thư viện chỉ xác minh API, **không chứng minh đây là cấu hình paper đã chốt**.

### 11.6. D05 — Generation snapshot/runtime

**CONFIRMED:** một SD1.5 base pretrained snapshot không fine-tune, cùng bundled text encoder/tokenizer/VAE; resolve repository/revision thành immutable snapshot và hashes trước generation. Generation FP32, AMP/autocast OFF, standard attention, không xFormers, VAE slicing/tiling OFF; không tự fallback dtype/backend. VAE posterior sampling dùng RNG stream riêng ở D02, latent scaling/normalization lấy từ model/pipeline config đã khóa; decode theo cùng pipeline và crop theo Common Spec.

**Lý do:** một execution path đơn giản, không thêm tối ưu backend hoặc precision search. Generation FP32 là quyết định đã xác nhận riêng, không suy từ classifier runtime. Concrete repository/revision và library versions vẫn phải được materialize thành snapshot thực tế và khóa trước generation; không dùng mutable branch lúc chạy.

### 11.7. D06 — E2 text conditioning

**CONFIRMED:** negative/unconditional prompt là empty string `""`, giữ nguyên positive base prompt và thay country name bằng tên lớp trong locked mapping. Kiểm tra token length bằng tokenizer của snapshot; nếu vượt context length thì dừng để xác nhận xử lý, không silent truncation hoặc tự viết lại prompt. Không dùng artifact negative prompt, confuser hoặc Geo-aware wording.

**Lý do:** không thêm nội dung conditioning ngoài base prompt nguồn. Context length thực tế lấy từ tokenizer snapshot, không tạo hyperparameter search.

### 11.8. D07 — Encoding/naming outputs

**CONFIRMED:** lưu output RGB uint8 PNG lossless ở native size, bỏ EXIF/orientation metadata sau khi đã decode đúng orientation. Khóa encoder/version; không thêm JPEG encode/decode variation làm augmentation. Output ID là SHA-256 của canonical job identity gồm method/config hash, class, source, output slot/attempt và strength nếu là E2; filename dùng output ID, strength pools có namespace riêng.

**Lý do:** không phải chọn JPEG quality/subsampling và không thêm mất mát do nén. PNG không đổi cleaned JPG input hoặc SD1.5 geometry. File/pixel hashes bắt buộc tính sau persist/read-back như common contract.

### 11.9. D08 — Technical-only QC

**CONFIRMED:** chỉ áp technical checks mục 5.2 và exact-duplicate audit, không semantic filtering hoặc quality thresholds. Nếu generated output trùng file SHA-256 **hoặc** decoded RGB pixel hash với real train hoặc một accepted output trong cùng pool, ghi duplicate reason và không nhận record đó như ảnh augmentation mới. Holdout collision vẫn chặn freeze theo mục 5.2, không được chuyển thành lựa chọn regenerate dựa trên holdout content.

Đối chiếu generated duplicates riêng cho mỗi strength pool; các pools khác strengths không cần có hashes khác nhau và không được so chéo để chọn winner. Nếu outputs trùng giữa hai strengths, log để minh bạch, không tự loại candidate strength. Không loại real duplicate records đã được common split cho phép trong cùng train.

**Lý do:** ngăn quota bị lấp bằng bản sao exact mà không đưa model hoặc cutoff về chất lượng vào Base. Không DINO filtering, manual semantic selection, geographic loss/filter hoặc blur/blank thresholds.

### 11.10. D09 — Retry và shortage

**CONFIRMED — revised:** mỗi lớp có `required_c` output slots. Duyệt slots j tăng dần; mỗi slot có tối đa **5 deterministic attempts**, `a=0..4`, seeds/source theo D02. Nhận **candidate đầu tiên** pass technical và exact-duplicate checks D08; dừng slot ngay sau success, không sinh đủ năm rồi chọn ảnh. Technical-invalid/training-duplicate candidate được ghi rejected và thử attempt kế tiếp. Identity violation hoặc holdout collision vẫn block publication, không tự retry như một QC reject bình thường.

Nếu cả 5 attempts của một slot thất bại, ghi shortage và chặn freeze; không chuyển budget giữa lớp/slot hoặc mở attempt thứ sáu. Candidate content attempts tối đa là `5*required_c`/class; infrastructure retry cùng job/seed/config không phải content attempt mới. Resume giữ nguyên completed decisions và accepted index, không bắt đầu lại một bộ năm attempts để vượt cap.

Các duplicate checks phụ thuộc accepted membership nên phải xử lý/commit theo thứ tự `(class_id, slot, attempt)` cố định, độc lập worker completion order. Không semantic ranking, best-of-N hoặc selection theo validation/test. **Lý do:** cho phép phục hồi lỗi kỹ thuật có giới hạn mà không thêm một bộ chọn chất lượng.

### 11.11. D10 — Strength selection

**CONFIRMED — revised:** tuning dùng đủ common seeds `[42,123,2026]` ở mỗi strength trong `{0.25,0.35,0.50}`, tổng cộng **9 candidate classifier runs**. Dùng nguyên full common training schedule/runtime và paired initial state theo seed. Từng run chọn best checkpoint theo overall Validation Macro F1 của Common Spec, rồi lấy Validation Tail Macro F1 chưa làm tròn tại checkpoint đó.

`score(s) = (TailF1(s,42) + TailF1(s,123) + TailF1(s,2026))/3`. Chọn strength có mean lớn nhất; hòa chính xác ở giá trị chưa làm tròn thì chọn strength nhỏ hơn. Báo đầy đủ per-seed values, mean và sample std để truy vết; std không phải selection metric hoặc tie-break. Không chọn seed, không pooled-prediction scoring, không test-based selection. Chưa đủ 9 completed runs hoặc score không evaluable thì không chọn winner; resume failed run đúng identity, không thay seed.

Cho phép verified reuse cả ba completed executions tại `s*` làm official E1E2 theo mục 6.4/Common Spec mục 9.3; không train tiếp từ tuning weights, không ghi đè source artifacts và không đếm hai lần. Không thể verify đủ bộ thì fresh cả ba official runs từ shared initial states, ghi rõ accounting thực tế. **Lý do:** selection phản ánh mean qua common seeds, tận dụng execution đã tương thích và vẫn giữ nguyên official matrix/research questions.

### 11.12. Decision record và materialization

Theo xác nhận của người dùng ngày 2026-11-26: **D01, D03–D08 CONFIRMED as proposed; D02, D09, D10 CONFIRMED with revisions** trong v1.2. Không còn pending design decision trong D01–D10. Mọi resolved config phải phản ánh các giá trị này và liên kết spec/Common Spec versions tương ứng.

Model revision, file hashes, backend versions và cụ thể hóa API-compatible coordinate/RNG conventions vẫn phải được materialize trước thực thi; đây là artifact identity/numerical realization, không cho phép đổi confirmed policy hoặc dùng kết quả experiments để chọn settings. Các thay đổi chính sách về sau phải có version/decision record mới. Spec này không tự chứng minh code hoặc experiments đã hoàn tất; xem tasks.md.
