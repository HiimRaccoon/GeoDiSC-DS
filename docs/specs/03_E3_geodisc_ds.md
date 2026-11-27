# 03 — E3 Geo-DiSC-DS Specification

> **Workflow (2026-11-28):** Implementation completion means the production implementation exists and required local unit/integration tests pass, including fixture or mock tests where needed. Deployment verification means actual UIT environment, data, models, resources, recovery and shared-code runtime checks have passed with evidence. Scientific execution completion means the required official runs, evaluations and report locks exist and verify. Pending deployment evidence never blocks writing or locally testing independent implementation; it continues to gate the corresponding real artifact publication, jobs and scientific stages. Full scientific execution requires separate user approval. Confirmed research requirements and acceptance rules are unchanged.


**Phiên bản:** 1.1\
**Ngày:** 2026-11-26\
**Official condition IDs:** `E3-A1`, `E3-A2`, `E3-A3`, `E3-A4`\
**Phạm vi:** CADS, DINOv3 Visual Confuser Mining + CCFG, Geo-preservation và Real–Synthetic Mixup.\
**Trạng thái:** G01–G10 đã CONFIRMED theo xác nhận người dùng. Static inspection của official DiSC-DS đã hoàn tất; các verification/materialization requirements chưa thực hiện được tách riêng tại mục 13.3. Chưa triển khai code, sinh ảnh hoặc huấn luyện.

## 1. Nguồn yêu cầu và thiết kế ablation

Nguồn có thẩm quyền:

- [Markdown tổng thể](../GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), Section V, VI và VII: công thức, hyperparameters và thiết kế E3.
- [Common Spec v1.3](00_common.md): fixed split, singleton, fairness invariants, classifier/runtime, evaluation, reproducibility, checkpoint/resume, verified reuse và locking.
- [E0 Spec v1.1](01_E0_original.md): real training pool dùng chung, membership identity và execution/artifact contract.
- [E1E2 Spec v1.2](02_E1E2_hybrid_baseline.md): E2 Base và confirmed D01–D10, shared E1 pool, selected strength và generation/QC policy.

Các yêu cầu đã chốt được kế thừa nguyên vẹn; mục 13 ghi confirmed decisions và verification requirements riêng, không mở lại công thức, hyperparameters hoặc ablation. Các khác biệt với official DiSC-DS được ghi rõ là Geo-specific adaptations, không trình bày chúng là bản sao implementation upstream.

| Điểm so sánh | Base Diffusion | CADS | DINOv3 + CCFG | Geo-preservation | Mixup | Câu hỏi ablation |
|---|---|---|---|---|---|---|
| Base = official E1E2 | ON | OFF | OFF | OFF | OFF | Base generative baseline |
| E3-A1 | ON | ON | OFF | OFF | OFF | CADS có tăng diversity hữu ích cho extreme-tail? |
| E3-A2 | ON | ON | ON | OFF | OFF | Confuser mining + CCFG có tăng separability? |
| E3-A3 | ON | ON | ON | ON | OFF | Prompt-based Geo-preservation có cải thiện geographic fidelity? |
| E3-A4 | ON | ON | ON | ON | ON | Same-class Mixup có giảm real–synthetic domain gap? |

```text
Official E1E2 = Base (E2)
        │ + CADS
        ▼
      E3-A1
        │ + DINOv3 confuser mining và CCFG
        ▼
      E3-A2
        │ + Geo-preservation prompts/baseline convention
        ▼
      E3-A3
        │ + Real–Synthetic Mixup khi train, không sinh pool mới
        ▼
      E3-A4 = Final E3
```

Reuse official E1E2 per-seed results làm Base theo Common Spec, bao gồm verified reused selected-strength runs nếu hợp lệ. Không tạo E3-A0 hoặc classifier E2-only; không train A1 từ weights đã fine-tune của Base hoặc train A4 từ A3 checkpoint. Mỗi A1–A4 có ba classifier runs từ shared initial states theo common seeds, tổng cộng 12 E3 official identities trong matrix 18 runs.

## 2. Shared membership, quota và cấu hình nền

### 2.1. Training membership

Gọi `R` là locked E0 real training pool, `A_E1` là frozen E1 traditional pool, `S_Ak` là final tail synthetic pool của variant k:

```text
T_A1 = R ∪ A_E1 ∪ S_A1
T_A2 = R ∪ A_E1 ∪ S_A2
T_A3 = R ∪ A_E1 ∪ S_A3
T_A4 = T_A3; chỉ khác training-time Mixup
S_A4 = S_A3, cùng actual files/membership/hashes
```

Mọi real IDs/labels/file và pixel hashes bằng đúng E0. Mọi E3 variant dùng đúng cùng E1 pool với E1E2, không generate lại E1. Head giữ nguyên; medium chỉ nhận cùng E1 augmentation. Groups và `n_train` lấy trước augmentation.

Với final synthetic, chỉ target `n_train < 20` và `n_original >= 2`. Target tổng là `min(100, 5*n_train)` và final synthetic quota là `target - n_train`, giống E2 theo từng lớp. Singleton giữ một real anchor, không final synthetic, không Mixup. Không suy singleton từ `n_train=1` vì lớp có 2 hoặc 3 originals vẫn đủ điều kiện.

References có membership/role riêng, **không vào bất kỳ `T_Ak` nào**. Không dùng validation/test làm source cho final generation, references, mining hoặc real pairing.

### 2.2. E2 configuration được kế thừa

| Thành phần | Giá trị / dependency bắt buộc |
|---|---|
| Backbone/mode | Cùng locked SD1.5 snapshot, img2img; không model fine-tuning, Textual Inversion hoặc LoRA |
| Solver | Cùng DPM-Solver++ realization D04: multistep, order 2, midpoint, linspace, lower-order-final ON, thresholding OFF, không Karras conversion bổ sung |
| Configured steps / CFG | 50 / `w=2.0` |
| Strength | Duy nhất `s*` từ locked E2 selection; không tune lại trong E3 hoặc Stage 1 |
| Generation batch size | 1 source/lần; condition branches không phải thêm output samples |
| Generation runtime | FP32, AMP/autocast OFF, standard attention, không xFormers, VAE slicing/tiling OFF; cùng VAE/normalization/latent scaling D05 |
| Geometry | Native cleaned size → minimal center reflection-pad `/8` → generation → crop đúng offsets/bbox theo Common Spec |
| Output | RGB uint8 PNG lossless, native source size; encoder/version và post-read-back hashes theo D07 |
| Source scheduling | Deterministic shuffle-cycle theo D02; reference policy phải bảo đảm 20 distinct sources khi có đủ nguồn |
| Candidate acceptance | Technical-only + exact-duplicate audit D08; tối đa 5 deterministic attempts/output slot, first-valid D09; shortage block freeze |
| Classifier | Kế thừa toàn bộ common runtime; center letterbox `896 × 384`, tensor `[B,3,384,896]`, batch 128, FP32, AMP OFF |

Các options còn lại tham chiếu nguyên resolved E2 config; không tạo defaults khác cho A1/A2/A3. Chỉ thêm thành phần tương ứng ablation; A3 có prompt/baseline exception được master chỉ định tại mục 7. D01/D03 là E1 implementation đã khóa, không áp lại transforms E1 lên ảnh diffusion.

Source native/padded/output dimensions vẫn là `1536 × 662 → 1536 × 664 → 1536 × 662` hoặc `1030 × 703 → 1032 × 704 → 1030 × 703`. `896 × 384` chỉ là classifier canvas; DINOv3 dùng checkpoint-specific processor riêng.

## 3. Ký hiệu và nguyên tắc thời gian

| Ký hiệu | Ý nghĩa |
|---|---|
| `c+`, `c-` | Target country và visually confusing country khác target |
| `y+`, `y-` | Original text embeddings của target và confuser prompts |
| `ŷ+`, `ŷ-` | Text conditions sau CADS corruption/rescaling/mixing |
| `n` trong CADS | Condition noise, cùng không gian embedding; khác image latent noise |
| `x_t`, `t_sched` | Noisy latent hiện tại và actual scheduler timestep |
| `t_norm` | Active-step normalized time chỉ dùng cho CADS/dynamic temperature |
| `ε_∅` | Unconditional U-Net prediction ở A1/A2; A3/A4 thay baseline condition theo mục 7 |
| `ε_ŷ+`, `ε_ŷ-` | Raw U-Net predictions với CADS-adjusted conditions |
| `ε_y+ = ε_CADS` | CFG-combined prediction sau Eq. (5), không phải raw `ε_ŷ+` |
| `ε_CCFG`, `ε_step` | CCFG prediction và final noise prediction đưa vào scheduler |

Với `N` actual active denoising timesteps, bước `i=0..N-1`:

```text
t_norm = 1 - i/(N-1), nếu N > 1
t_norm = 0,           nếu N = 1
```

N được xác định từ actual img2img schedule ở locked strength, không mặc định bằng 50. Với N>1, bước đầu `t_norm=1`, bước cuối `t_norm=0`. Với N=1 áp đúng nhánh đặc biệt trong master. Không có active timestep thì không có generation hợp lệ; không bypass denoising để tính quota.

Chỉ CADS và dynamic CCFG temperature dùng `t_norm`. U-Net, scheduler input scaling và `scheduler.step` dùng `t_sched`/latent convention gốc của E2. Active-time normalization là Geo-specific img2img adaptation; không đổi solver schedule.

## 4. E3-A1 — CADS

### 4.1. Parameters và text conditions

| Parameter | Giá trị khóa |
|---|---:|
| CFG `w` | 2.0 |
| `τ1` | 0.5 |
| `τ2` | 0.9 |
| Condition-noise scale `s` | 0.1 |
| Rescale mixing `ψ` | 1.0 |

Ở đây `s=0.1` là **condition-noise scale**, không phải selected img2img strength `s*`. Positive prompt là nguyên base prompt của E2 cho target country; unconditional baseline là empty string theo confirmed D06. Không confuser branch, DINOv3, CCFG, Geo prompts hoặc Mixup.

### 4.2. CADS equations — Eq. (1)–(5)

Annealing schedule, dùng `t=t_norm` — Eq. (2):

```text
γ(t) = 1                       nếu t <= τ1
       (τ2-t)/(τ2-τ1)          nếu τ1 < t < τ2
       0                       nếu t >= τ2
```

Condition corruption — Eq. (1):

```text
ŷ+ = sqrt(γ(t)) * y+ + s * sqrt(1-γ(t)) * n
```

Rescale về statistics của original condition — Eq. (3):

```text
μ_in = mean(y+)
σ_in = std(y+)
y_rescaled = ((ŷ+ - mean(ŷ+))/std(ŷ+)) * σ_in + μ_in
```

Mixing — Eq. (4):

```text
ŷ+ = ψ*y_rescaled + (1-ψ)*ŷ+
ψ = 1.0  →  ŷ+ = y_rescaled
```

Trong Eq. (4), `ŷ+` ở vế phải là corrupted embedding trước rescaling. Mỗi bước lấy original `y+` làm cơ sở Eq. (1), không tích lũy perturbation bằng cách lấy embedding đã corrupt ở bước trước làm original.

CFG — Eq. (5):

```text
ε_∅  = U-Net(x_t, t_sched, unconditional condition)
ε_ŷ+ = U-Net(x_t, t_sched, CADS positive condition)
ε_y+ = ε_∅ + w*(ε_ŷ+ - ε_∅)
ε_CADS = ε_y+
```

A1 đưa `ε_step = ε_CADS` vào scheduler. Không corrupt unconditional embedding. Không thay condition noise bằng image noise hoặc perturb trực tiếp latent dưới tên CADS.

### 4.3. Confirmed numerical semantics — G01/G02

Với embedding tensor `[B,T,D]`, tính mean/std **riêng từng embedding/sample trên T và D**, không reduce qua batch. Dùng FP32, **population std** (`correction=0`), bao gồm **mọi tokens** đang có trong embedding, kể cả special/padding tokens; không áp token mask cho statistics. Original và corrupted conditions được tính statistics riêng, đúng vai trò trong Eq. (3).

Fail generation job nếu original/corrupted statistics không hữu hạn hoặc một std bằng 0; không chèn epsilon, clamp std, skip rescale hoặc thay `ψ`. Numerical failure không được publish output hợp lệ; việc ghi failed attempt/retry theo D09 vẫn giữ nguyên.

Condition noise là **standard Gaussian `N(0,I)`**, cùng shape với embedding, FP32; redraw tại **mỗi active timestep**, độc lập theo positive/confuser streams, sample, slot và attempt. Draw mới vẫn được định danh tại bước có noise coefficient bằng 0; không reuse noise của bước trước. Baseline không bị perturb. Eq. (1)–(5) và các scales giữ nguyên; các RNG streams theo confirmed G03 ở mục 8.1.

## 5. E3-A2 — Stage 1: reference bank và DINOv3 confuser mining

### 5.1. Reference generation

Stage 1 sinh **20 CADS-generated reference images/lớp cho toàn bộ 124 lớp**, tổng **2.480 accepted references**. Nguồn chỉ là real training originals. References dùng cùng locked SD1.5/strength, base country prompt và CADS A1; chưa có CCFG, Geo prompt hoặc classifier Mixup.

| Số training real sources | Source policy cho 20 reference slots |
|---|---|
| `n_train >= 20` | Random sample 20 distinct sources bằng deterministic shuffle, mỗi source sinh một reference |
| `1 <= n_train < 20` | Dùng hết sources rồi shuffle-cycle/cycle lại đến đủ 20 slots |
| Singleton | Một training real source được reuse cho 20 references với distinct noise states; chỉ có candidate centroid, không final training augmentation |

Confirmed shuffle-cycle policy D02 đáp ứng hai trường hợp đầu: lấy 20 slots đầu, không dùng số ảnh toàn dataset làm source count. Retry cùng slot giữ source để không phá distinct-source policy. Mỗi slot/attempt có noise identity riêng; Stage 1 namespace phải tách khỏi final training generation, chi tiết RNG mở rộng tại G03.

Reference generation kế thừa output/QC/retry mechanics E2, với quota reference **20/class** thay cho final training quota. Thiếu reference ở bất kỳ lớp nào chặn bank freeze/mining; không giảm `N_ref`, bỏ candidate class hoặc dùng ảnh real để bù. Candidate ledgers giữ role `reference`, source IDs/hashes, class, seeds, actual schedule và crop/output hashes.

### 5.2. DINOv3 extraction

| Thành phần | Giá trị khóa |
|---|---|
| Backbone | DINOv3 ViT-B/16, web-pretrained; không satellite weights |
| Parameters / patch size | Khoảng 86M / 16 |
| Model mode | Frozen, eval và inference mode; không fine-tuning |
| Input | Persisted native RGB references qua checkpoint-specific image processor |
| Feature | Final CLS embedding 768 chiều; không classifier logits hoặc patch-token average |
| Embedding normalization | L2 normalize từng reference embedding trước centroid averaging |

**G04 CONFIRMED:** dùng `facebook/dinov3-vitb16-pretrain-lvd1689m`, official checkpoint processor cùng revision, frozen/eval/inference mode, **FP32, autocast OFF, batch size 32**; batch cuối nhỏ hơn 32 được giữ, không có batch/dtype fallback.

Immutable revision đã resolve từ [Hugging Face model metadata](https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m): **`5931719e67bbdb9737e363e781fb0c67687896bc`**. Dùng revision này cho cả model và processor, không `main`. Model metadata khai báo access gated; pinned processor/config endpoints không truy cập được trong phiên này và weights chưa được tải. Vì vậy **revision đã resolve**, nhưng file hashes và processor H × W/resize/crop/normalization vẫn **chưa được materialize/verify** (V01).

Trước extraction, tải snapshot bằng access hợp lệ, khóa weights/config/processor hashes và Transformers/backend version, ghi actual input tensor shape từ official processor. Không đoán processor size, không thay bằng canvas EfficientNet hoặc checkpoint khác nếu chưa có access. Checkpoint-specific preprocessing giữ nguyên, không tune theo metrics.

### 5.3. Centroids, cosine similarity và confuser — Eq. (6)–(7)

Với reference r của lớp c, `h_c,r` là CLS embedding và `v_c,r` là embedding đã L2-normalize:

```text
v_c,r = h_c,r / ||h_c,r||_2
μ_c   = (1/N_ref) * Σ_r v_c,r
N_ref = 20
```

Tập đưa vào DINOv3 để tính centroid là **100% CADS references**. Real sources chỉ dùng để sinh references; không average real embeddings chung với reference embeddings.

Cosine similarity — Eq. (6):

```text
sim(μ_c, μ_c') = (μ_c · μ_c') / (||μ_c||_2 * ||μ_c'||_2)
```

Với mỗi target eligible tail, chọn confuser — Eq. (7):

```text
c- = argmax_{c' ∈ C \ {c+}} sim(μ_c+, μ_c')
C = toàn bộ 124 lớp
```

Loại chính target khỏi 123 candidates còn lại; không chỉ so với tail classes. Head, medium và singleton đều có thể là confuser. Không cần chọn confuser để sinh final training synthetic cho singleton vì singleton không đủ điều kiện target.

Không thay cosine bằng unnormalized centroid dot product. Centroid là mean của normalized embeddings; nếu lưu normalized centroid phục vụ cosine, phải giữ được original mean/normalization metadata. **G05 CONFIRMED:** embeddings/means/norms/cosine tính bằng FP32; nếu có non-finite values hoặc raw embedding/centroid norm bằng 0 thì fail extraction/mining, không tự thêm epsilon. Exact ties của FP32 similarities chọn **class ID nhỏ nhất** trong candidates khác target, trước mọi rounding. Không dùng tolerance, classifier confusion/test labels hoặc một metric khác làm tie-break.

### 5.4. Cache và reference isolation

Tạo và khóa reference bank, per-reference embeddings, centroids, similarity evidence và confuser mapping từ A2 một lần, dùng nguyên artifacts ở A3/A4. Không re-mine sau khi chuyển Geo prompt, xem validation/test hoặc đổi classifier seed.

Không đưa reference files/IDs vào classifier membership, kể cả references của eligible tail. Không lấy A1 final synthetic pool làm reference bank hoặc lấy references làm final synthetic quota. Reference membership và final-training membership phải là hai artifact roles riêng, dù đều là ảnh CADS-generated.

## 6. E3-A2 — Stage 2: CCFG và final synthetic

### 6.1. Condition branches và parameters

Target positive và confuser dùng base country prompt E2, thay country name tương ứng. Baseline A2 là empty unconditional condition. Cùng `x_t`/actual timestep được dùng cho ba predictions:

```text
ε_∅  = U-Net(x_t, t_sched, unconditional condition)
ε_ŷ+ = U-Net(x_t, t_sched, CADS target condition)
ε_ŷ- = U-Net(x_t, t_sched, CADS confuser condition)
```

| Parameter | Giá trị |
|---|---:|
| Guidance `w` | 2.0 |
| Temperature `τ` | 0.8 |
| CADS interpolation weight `α` | 0.8 |
| Dynamic temperature | `τ_hat(t) = τ*sqrt(γ(t))` |

Positive/confuser dùng cùng CADS annealing schedule, **condition noises độc lập** và original mean/std riêng cho mỗi embedding. Baseline không áp CADS. G08 static inspection đã xác minh upstream cũng perturb cả hai conditions bằng các noise draws riêng; Geo dùng explicit independent streams G02/G03 và vẫn giữ nguyên master convention. Các khác biệt numerical/schedule/guidance khác được ghi ở mục 13.2, không tự đổi convention hoặc thêm ablation để khớp upstream.

### 6.2. Adaptive scales — Eq. (8)

Trước hết tính CFG output Eq. (5):

```text
ε_y+ = ε_∅ + w*(ε_ŷ+ - ε_∅)

d_pos = ||ε_∅ - ε_y+||²_2
d_neg = ||ε_∅ - ε_ŷ-||²_2
```

**`d_pos` dùng CFG output `ε_y+`, không dùng raw prediction `ε_ŷ+`.** Theo từng sample, squared L2 được tính bằng **sum** trên latent C,H,W, không reduce qua batch:

```text
d_pos = sum_CHW((ε_∅ - ε_y+)²)
d_neg = sum_CHW((ε_∅ - ε_ŷ-)²)
```

Eq. (8), thay τ bằng dynamic `τ_hat(t)` của Eq. (10):

```text
w_pos =  2w / (1 + exp(-τ_hat*d_pos))
w_neg = -2w*exp(-τ_hat*d_neg) / (1 + exp(-τ_hat*d_neg))
```

Dạng số học ổn định, tương đương master:

```text
w_pos =  2w*sigmoid( τ_hat*d_pos)
w_neg = -2w*sigmoid(-τ_hat*d_neg)
```

Norms và scales tính bằng float32. Log `d_pos`, `d_neg`, `w_pos`, `w_neg` để nhận biết saturation; không đổi sum thành mean, đổi temperature, normalize/clamp distances hoặc rescale theo latent size để tránh saturation. Những thay đổi đó không thuộc benchmark đã chốt.

### 6.3. CCFG/interpolation — Eq. (9)–(11)

CCFG noise prediction — Eq. (9):

```text
ε_CCFG = ε_∅
         + w_pos*(ε_ŷ+ - ε_∅)
         + w_neg*(ε_ŷ- - ε_∅)
```

Dynamic temperature — Eq. (10):

```text
τ_hat(t) = τ*sqrt(γ(t))
```

Noise-level interpolation — Eq. (11):

```text
ε_step = α*ε_CADS + (1-α)*ε_CCFG
       = 0.8*ε_CADS + 0.2*ε_CCFG
ε_CADS = ε_y+
```

Interpolate noise predictions **trước** scheduler update, không chạy hai solver paths rồi trộn images/latents. Mỗi active timestep gọi `scheduler.step` đúng một lần với `ε_step`. Confuser không được nối vào artifact negative prompt.

## 7. E3-A3 — Prompt-based Geo-preservation

### 7.1. Mục tiêu và phạm vi

A3 giữ A2 và thêm Geo-preservation qua text conditioning. Cho phép lighting, weather, tone/color, textures và minor environmental details thay đổi; hạn chế drift ở traffic direction, road geometry, lane markings, road signs/script, architecture, utility poles/infrastructure, vegetation, terrain và roadside structure.

Đây là mục tiêu của prompt, không phải một bảo đảm ảnh sinh đúng địa lý hoặc một acceptance detector. Không thêm geographic consistency loss, feature constraint, semantic/geographic filtering hay quality thresholds.

### 7.2. Prompts nguyên văn từ master

Geo-aware positive prompt v1:

```text
Realistic street-level road photograph in {country_name}. Authentic local architecture, traffic direction, lane markings, road signs, utility poles, vegetation and terrain. Preserve road geometry and geographic identity. Natural variations in lighting, weather and colors.
```

Artifact negative prompt v1:

```text
illustration, cartoon, CGI, blurry, warped geometry, reversed traffic, incorrect lane markings, fake road signs, altered text, distorted vehicles, unrealistic architecture, watermark, compression artifacts
```

Confuser prompt vẫn là base prompt E2 với confusing country từ cached mapping:

```text
A realistic street-level road photograph in {country_name}, look like picture in Google Street View.
```

### 7.3. Integration convention

| Condition | A1 | A2 | A3 và A4 generation |
|---|---|---|---|
| Positive | Base target prompt + CADS | Base target prompt + CADS | Geo-aware target prompt + CADS |
| Confuser | Không dùng | Base confuser prompt + CADS | Cùng base confuser prompt + CADS |
| Baseline | Empty string, không CADS | Empty string, không CADS | Artifact negative prompt, không CADS |

Ở A3, định nghĩa baseline prediction `ε_b = U-Net(x_t,t_sched,artifact negative condition)` và thay **mọi `ε_∅` trong Eq. (5), (8), (9)** bằng `ε_b`. Không thay công thức hoặc hyperparameters khác. `d_pos` vẫn dùng CFG output sau baseline replacement, `d_neg` vẫn dùng raw CADS-confuser prediction. Cùng `ε_b` làm baseline cho cả CADS và CCFG branches tại timestep đó.

Confuser là condition độc lập, không concatenate confusing country vào artifact prompt; cached mapping không đổi theo Geo prompt. Original statistics để rescale positive lấy từ Geo-aware embedding; confuser lấy từ base-confuser embedding riêng.

### 7.4. Token length và prompt locking

Với locked SD1.5 tokenizer, kiểm tra **mọi rendered country prompt** và artifact prompt, gồm special tokens:

```text
len(tokenizer(prompt, truncation=False).input_ids) <= 77
```

**G06 CONFIRMED:** giữ nguyên master prompts và audit toàn bộ rendered base/Geo/confuser country prompts cùng artifact prompt **trước generation**. Không silent truncation hoặc sửa prompt để tự vượt audit. Nếu có prompt >77 tokens, xuất overflow report trước mọi modification và chặn generation bị ảnh hưởng; chỉ rút gọn sau khi có prompt revision được xác nhận riêng theo yêu cầu master. Lưu original template, rendered text, token IDs/count và approved prompt/config hash. Audit chưa được thực thi trong phiên viết spec; không khẳng định templates hiện tại đều fit (V02).

## 8. Sampling pipeline dùng chung cho A1–A3 và references

```text
Locked real training source + selected strength + variant conditions
        │
        ▼
Native RGB → minimal reflection-pad /8 → VAE encode/scaled latent
        │
        ▼
Add latent noise theo locked img2img strength → x_t, active timesteps
        │
        ▼
For i trong active timesteps:
    1. Tính t_norm và γ(t_norm)
    2. Chuẩn bị CADS positive condition từ original embedding
       Nếu A2/A3: chuẩn bị CADS confuser condition, noise độc lập
    3. U-Net predictions dùng actual scheduler timestep
       A1/reference: baseline + positive
       A2/A3:       baseline + positive + confuser
    4. Tính ε_CADS bằng Eq. (5)
    5. A2/A3: tính τ_hat, scales, ε_CCFG rồi ε_step bằng Eq. (8)–(11)
       A1/reference: ε_step = ε_CADS
    6. Gọi scheduler.step một lần → next scheduled latent
        │
        ▼
Final latent → VAE decode → crop exact padding → native RGB PNG
        │
        ▼
Technical/hash QC → first-valid slot acceptance → freeze artifact
```

Ở A3 baseline là artifact-negative prediction theo mục 7.3; A4 không gọi pipeline generation riêng, chỉ dùng output đã khóa của A3. Đây là img2img từ encoded real source, không khởi tạo hoàn toàn từ Gaussian image latent.

VAE/latent/source streams kế thừa mechanics E2. CADS positive/confuser noise là streams bổ sung, không được tiêu thụ chung rồi vô tình làm thay latent noise hoặc source order. Confirmed G02/G03 áp dụng như sau.

### 8.1. Paired generation streams — G03

Final generation **E2/A1/A2/A3** dùng cùng source schedule, source ID, VAE seed và initial latent-noise seed/tensor cho cùng `(class_id, slot j, attempt a)`. Reuse nguyên seed-key derivation confirmed D02 với method key **`E2`** và streams `source_cycle`, `vae`, `latent`; index attempt vẫn `5*j+a`, cycle/permutation không phụ thuộc variant. Không sửa hoặc regenerate E2 đã khóa để thay keys. Variant vẫn nằm trong job/output identity và config hash, không nằm trong shared source/VAE/latent seed keys.

Reference generation có namespace riêng **`geo-e3-ref-v1`**, cùng seed derivation mechanics, source shuffle-cycle và per-slot/attempt indexing, nhưng không dùng final namespace. Key reference là `geo-e3-ref-v1|314159|E2|<class_id>|<index>|<stream>`, index/stream theo D02. CADS RNG có key riêng **`geo-e3-cads-v1|314159|<role>|<class_id>|<slot>|<attempt>|<step>|<branch>`**, role là `reference`/`final`, branch là `positive`/`confuser`; seed lấy bốn bytes đầu SHA-256 theo big-endian của UTF-8 key. Các final variants dùng cùng positive CADS stream cho cùng step/slot/attempt; confuser dùng stream độc lập, không làm lệch positive stream khi branch chưa có ở A1. Reference CADS streams tách khỏi final.

Lưu actual keys/seeds, source permutation, initial VAE latent/noise hashes và accepted attempt index. First-valid acceptance/retry có thể khiến accepted attempts khác nhau giữa variants; pairing được bảo đảm **theo slot/attempt**, không ép accepted outputs cùng noise bằng cách chạy thêm hoặc bỏ candidate hợp lệ. A4 chỉ reuse A3 output, không có generation streams mới.

## 9. E3-A4 — Same-class Real–Synthetic Mixup

### 9.1. Pool và parameters

A4 là Final E3. Dùng nguyên `S_A3` và full training membership A3; không sinh synthetic pool mới, không biến mixed tensors thành offline dataset.

| Parameter | Giá trị khóa |
|---|---|
| Pairing | Same-class real–synthetic |
| Batch ratio | 0.5, chọn `floor(0.5*number_of_batches)` mỗi epoch |
| Eligible anchors | Chỉ final synthetic, không traditional E1 hoặc reference samples |
| Real partners | Real training originals cùng country; cho phép reuse/oversample để pairing |
| Lambda | Uniform discrete trên `{0.00,0.05,...,0.95,1.00}`, độc lập mỗi cặp |
| Labels | Giữ nguyên class label; không cross-country soft labels |

### 9.2. Mixup equations

Với một final-synthetic anchor và real partner cùng lớp:

```text
λ ~ UniformDiscrete({0.00,0.05,...,0.95,1.00})
x_mix = λ*x_real + (1-λ)*x_synthetic
y_mix = y_class
```

Đây là tập **21** giá trị, có cả endpoints. `λ=1` cho real partner, `λ=0` cho synthetic anchor; không resample lambda để loại endpoints. Không dùng Beta distribution hoặc đảo vai trò coefficients so với master.

### 9.3. Batch construction và compute invariants

1. Shuffle toàn bộ same A3/A4 training anchors mỗi epoch theo common sampler.
2. Với `L=ceil(N_anchors/128)` batches, chọn **uniform không hoàn lại** đúng `floor(0.5*L)` batch indices cho Mixup, không tăng số batches.
3. Trong mỗi selected batch, chỉ final synthetic anchors được pair với real training image cùng country, sample **uniform có hoàn lại** từ canonical real class pool; cho phép reuse real partner.
4. Đưa hai images qua cùng common RGB float `[0,1]`, resize/center-letterbox stage để tensors cùng shape; **Mixup trước ImageNet normalization**, sau đó normalize mixed tensor đúng một lần. Không thêm image transforms riêng.
5. Thay synthetic tensor bằng mixed tensor ở chính anchor position đó. Giữ nguyên original real/E1 tensors trong batch; không append partner thành anchor mới.
6. Batch không được chọn train bình thường; validation/test luôn real, không Mixup.

Head, medium và singleton train bình thường. Singleton không có eligible synthetic anchor và không tham gia pairing. Không yêu cầu partner phải là original source của synthetic; yêu cầu đã chốt là real train **cùng class**. **G07 CONFIRMED:** independent seeded RNG streams cho batch selection, real-partner sampling và lambda, tách khỏi anchor sampler/model RNG. Dẫn xuất seed từ common training seed, epoch và stream tag trong namespace `geo-e3-mixup-v1`; pair decisions thêm batch/anchor ID để không phụ thuộc worker completion order. Lưu seed/selected batch indices/partner IDs/lambdas và states trong epoch/checkpoint provenance; không dùng augmentation seed để thay training RNG.

50% là tỷ lệ batches được chọn, không phải tỷ lệ toàn bộ samples được trộn hoặc bảo đảm mọi synthetic được Mixup mỗi epoch. Selected batch không có eligible anchor vẫn giữ batch và optimizer step như bình thường; không resample batch để ép tỷ lệ ảnh. Partner oversampling chỉ là lookup để tạo mixed tensor, không thay common anchor sampler.

A3 và A4 có cùng N anchors, batch size, last-batch policy, epochs và optimizer steps: `L`/epoch, `20*L` tổng cộng. Không thêm steps để “dùng hết” real partners hoặc synthetic. Log selected batch count, eligible/mixed counts, partner/lambda evidence và RNG state để resume tái lập được.

## 10. Generation/QC, training và reuse contracts

### 10.1. Generation và pool publication

Final generation A1/A2/A3 kế thừa confirmed E2 technical-only QC và retry policy, cùng quota theo lớp; references áp cùng cơ chế với quota riêng mục 5.1. Duyệt/commit theo deterministic class/slot/attempt order, nhận first-valid trong tối đa 5 attempts, không semantic ranking hoặc best-of-N. Technical-invalid/training-duplicate output bị reject; identity/leakage incident block publication; hết attempts thiếu quota phải dừng, không mở attempt thứ sáu.

File hashes và decoded RGB pixel hashes tính sau PNG persist/read-back theo Common Spec. Không thêm DINO similarity thresholds, classifier-based acceptance, geographic filters hoặc manual semantic selection. DINOv3 chỉ phục vụ confuser mining, không quyết định final sample có đúng nhãn hay không.

Pool roles tách rõ `reference` với `final_synthetic`. **G09 CONFIRMED:** final output trùng file SHA-256 **hoặc** decoded RGB pixel hash với bất kỳ ảnh nào trong frozen reference bank bị reject với reason `reference_final_exact_duplicate`, rồi thử attempt kế tiếp trong cùng slot theo D09. Áp cho A1/A2/A3; A4 reuse nguyên A3. Không mở cap mới, không đổi source schedule hoặc regenerate reference bank để làm final output hợp lệ. Hết 5 attempts vẫn thiếu thì block freeze.

Để giữ nguyên selected E2 pool đã khóa, khi tạo reference bank cũng audit reference candidates với E2 final hashes: nếu exact duplicate thì reject **reference candidate** và retry reference slot theo cùng D09, không chỉnh sửa E2 files/membership hoặc selected strength. Như vậy bank freeze không tạo collision với Base đã có, và các E3 final pools sau đó đều kiểm tra với bank cố định; không cần mở lại confirmed E2 generation policy hoặc runs.

Frozen reference bank phải sẵn sàng để final slot acceptance/pool freeze hoàn tất audit này. A1 có thể chuẩn bị/generate provisional candidates trước bank, nhưng chưa được publish accepted training pool hoặc train official classifier trước cross-role audit. Đây là **technical QC dependency**, không đưa DINOv3/confuser vào A1 guidance hoặc thêm một research component. Bank sinh trực tiếp từ real sources bằng CADS, không phụ thuộc A1 final pool nên không tạo vòng dependency. Holdout collisions vẫn block trực tiếp; equality giữa các final ablation pools chỉ được log, không làm semantic ranking/cross-variant rejection.

### 10.2. Classifier execution

Mọi E3 run dùng common model/runtime/initial state theo seed, cùng real validation/test và metrics/support policy. A1–A3 không Mixup/CutMix; A4 chỉ khác Mixup tại mục 9. Không weighted/focal loss, class-balanced sampler hoặc extra augmentation.

Một epoch duyệt mỗi anchor đúng một lần. Reference count trong training luôn 0; cùng real/E1 pool hashes cho mọi variant, A3/A4 cùng final synthetic hash. Training history ghi actual counts/steps, validation-at-best và selected checkpoint theo Common Spec; không chọn checkpoint theo group-specific score khác.

Resume tại epoch boundary phải phục hồi common model/optimizer/scheduler/RNG/history và A4 mixing RNG/schedule state đủ tái tạo epoch tiếp theo. Không regenerate pool khi classifier resume. Generation resume theo completed slots/jobs và hashes, không chọn lại confuser hoặc reset attempt cap.

### 10.3. Reuse boundaries

| Artifact | Tạo ở đâu | Dùng lại ở đâu | Không được làm |
|---|---|---|---|
| E0 real pool | Common/E0 | E1E2 và A1–A4 | Đổi real membership theo variant |
| E1 traditional pool | E1E2 | A1–A4 và mọi training seed | Generate E1 mới cho E3 |
| Selected strength/E2 base config | E1E2 tuning/lock | A1–A4, Stage 1 references | Tune lại strength/sampler/runtime |
| Official E1E2 results | E1E2, kể cả verified reuse | Base của E3 ablation | Thêm A0/E2-only hoặc warm-start E3 từ Base weights |
| Reference bank/embeddings/centroids/confusers | A2 Stage 1 | A2 final generation, A3/A4 | Re-mine theo Geo prompt/seed hoặc đưa refs vào train |
| Final A3 synthetic pool | A3 | A4 | Sinh pool A4 mới hoặc đổi accepted membership |
| Shared classifier initialization | Common theo seed | Mọi official variant | Load trained checkpoint của variant trước làm initialization |

A1 final synthetic, A2 references và A2/A3 final synthetic không phải cùng artifact chỉ vì cùng nguồn/seed hoặc cùng native size. A3 và A4 là ngoại lệ reuse pool được chỉ định rõ, không phải yêu cầu equality content giữa mọi ablations.

## 11. Artifact dependencies và đầu ra

### 11.1. Dependency graph

```text
Shared inputs:
Common data/runtime/initial states + E0/E1 pools + E2 strength/base config
                               │
                               ├→ A1 candidates → bank-hash QC → S_A1 → A1 runs
                               │
                               └→ A2 CADS references, đủ 124 lớp
                                            │
                                            ▼
                                  DINOv3 embeddings/centroids
                                            │
                                            ▼
                                  Similarities + confuser map
                                            │
                    ┌───────────────────────┴─────────────────────┐
                    ▼                                             ▼
             A2 final generation                          A3 final generation
                    │                                    Geo prompts/baseline
                    ▼                                             │
                   S_A2                                           ▼
                    │                                            S_A3
                    ▼                                             │
             A2 classifier runs                   ┌───────────────┴─────────────┐
                                                  ▼                             ▼
                                          A3 classifier runs            A4 classifier runs
                                          No Mixup                     Same pool + Mixup
```

Tất cả generation nodes dùng E2 locked base/strength; A3/A4 dùng đúng A2 cache. A1 sampling không dùng DINOv3 features/confuser mapping hoặc trained E1E2 weights; **A1 final acceptance/freeze bổ sung dependency tới reference-bank hashes để thực hiện G09**. Thêm edge từ frozen reference bank đến exact-duplicate QC của mọi S_A1/S_A2/S_A3 trước các pool locks; không cần feature extraction/mining cho A1 QC. Official Base results cần có khi tổng hợp ablation, không phải input weights cho generation/classifier.

### 11.2. Artifact layout và nội dung

Đường dẫn tính từ benchmark-version artifact root, chỉ là specification output contract; chưa tạo các files này.

| Artifact | Nội dung tối thiểu |
|---|---|
| `augmentation/E3/shared_config.json` | Inherited E2 config/model/environment/strength hashes, CADS parameters, source/output/QC policy và confirmed E3 decisions |
| `augmentation/E3/A1/config.json`, `candidates.jsonl` | A1 resolved conditions, jobs/seeds/source/attempt/output hashes, actual active schedule và CADS traces |
| `augmentation/E3/A2/references/manifest.csv`, `bank_lock.json` | 2.480 accepted CADS refs, 20/class, source/crop/seed/prompt hashes, role reference, generation dependencies |
| `augmentation/E3/A2/features/embeddings.npy`, `centroids.npy`, `feature_index.json`, `feature_lock.json` | FP32 NumPy normalized CLS vectors/mean centroids, model/processor identity và row/reference/class mapping |
| `augmentation/E3/A2/similarities.npy`, `confusers.json`, `mining_lock.json` | FP32 cosine matrix, candidate label order và Top-1 mapping, exact-tie evidence, feature/bank dependencies |
| `augmentation/E3/A2/config.json`, `candidates.jsonl` | Final A2 generation config, cached confuser refs, target/negative condition seeds, raw/scaled guidance diagnostics |
| `augmentation/E3/A3/prompts.json`, `token_audit.json` | Verbatim/approved prompt templates, rendered texts, tokenizer/token IDs/counts, shortening decisions nếu cần |
| `augmentation/E3/A3/config.json`, `candidates.jsonl` | A3 generation/prompt/baseline config, same cached mapping, source/job/output/attempt evidence |
| `augmentation/E3/<stage_or_variant>/guidance.jsonl` | Per-attempt/per-active-timestep diagnostics G10, t_sched/t_norm/γ, distances/scales/interpolation và condition identities |
| `augmentation/E3/upstream_comparison.json` | Official repository commit/files đã inspect, findings G08 và confirmed Geo adaptation mapping |
| `pools/E3-A<k>/manifest.csv`, `pool.json`, `pool_lock.json`, `k=1,2,3` | Full real/E1/final-synthetic membership, per-class quota audit, references excluded, dependency hashes |
| `pools/E3-A4/pool.json`, `pool_lock.json` | Tham chiếu trực tiếp A3 canonical training manifest/final pool; không tạo independent synthetic membership |
| `runs/E3-A<k>/seed_<seed>/` | Official classifier config/metadata/history/LR/best/last/metrics/checkpoint locks theo Common Spec |
| `runs/E3-A4/seed_<seed>/mixup_config.json`, `mixup_history.*` | Locked policy/RNG identity, per-epoch selected batch evidence, pairing/lambda trace và committed summary |

**G10 CONFIRMED:** feature artifacts là NumPy `.npy`, dtype float32, không object arrays/pickle; `embeddings.npy` shape `[2480,768]`, `centroids.npy` `[124,768]`, `similarities.npy` `[124,124]`. `feature_index.json` ghi mỗi reference ID/hash/class/row và thứ tự class IDs; cùng metadata/arrays được hash-lock. Mean centroid được giữ nguyên, không ghi đè bằng normalized copy không có provenance.

Guidance diagnostics là **JSONL cho từng active timestep của từng attempt** ở A1, A2, A3 và reference generation. Mỗi row có job/role/variant/class/slot/attempt/step identity, N, actual scheduler timestep, t_norm, γ, branch/noise identities, baseline type, FP32 statistics/norms, `d_pos`, `d_neg`, `τ_hat`, `w_pos`, `w_neg`, α và scheduler update index khi áp dụng. CCFG-only fields ở A1/references là `null`, không giả số 0. Không serialize NaN/Infinity như JSON numbers; numerical error được ghi status/reason với invalid fields `null`, attempt không thể accepted. Diagnostics không tạo saturation cutoff hoặc filtering mới.

Tách JSONL theo job/namespace để resume giữ được prefix đã commit và không nhân đôi timestep rows; pool lock gắn hashes của completed traces/candidate decisions. Mixup history vẫn theo per-epoch/pair evidence, không phải guidance traces.

### 11.3. Required provenance và locks

Generation records phải lưu source image ID/hashes, original/padded/output sizes và crop bbox, stage/variant/role, class/slot/attempt, actual source/latent/VAE/CADS stream seeds, prompt/tokenizer/config hashes, actual timesteps/N và QC decision/reason.

A2/A3 records thêm confuser ID/mapping hash, condition identities và guidance diagnostics đủ phát hiện nhầm Eq. (8)/saturation. Reference feature metadata phải truy ngược mỗi embedding về đúng reference file/hash và checkpoint/processor; không chỉ lưu một matrix không có labels/order.

Run identities chứa common dependencies, variant config, shared E0/E1/strength hashes, variant synthetic membership, cache/mapping/prompt dependencies khi dùng và shared initial-state hash. A4 thêm Mixup config/RNG identity, đồng thời chứng minh canonical manifest bằng A3. Snapshot cache/pool được khóa trước classifier training, không cập nhật khi xem metrics.

Atomic publication và checksum verification theo Common Spec: publish đủ upstream artifacts rồi lock cuối cùng, xác minh trước downstream/resume; mismatch/shortage chặn thực thi. Không đổi upstream file và sửa hash cho khớp trong cùng benchmark version.

## 12. Acceptance criteria

Đây là điều kiện cần chứng minh khi triển khai, chưa phải tuyên bố đã chạy kiểm tra hoặc model đạt hiệu quả.

| ID | Tiêu chí | Bằng chứng |
|---|---|---|
| E3-AC01 | Đúng cumulative ablation; Base reuse official E1E2, không A0 | Variant flags/config comparisons và Base run references |
| E3-AC02 | Shared E0/E1 pools, groups/singleton và per-class E2 tail budgets giữ nguyên | IDs/labels/file/pixel hashes, pool/quota locks |
| E3-AC03 | Cùng selected strength, SD1.5/DPM-Solver++/steps/CFG/runtime/geometry, không tune lại | Inherited resolved config và dependency hashes |
| E3-AC04 | Eq. (1)–(5) và CADS parameters đúng; baseline không CADS | Condition/γ/statistics/CFG evidence và numerical realization đã xác nhận |
| E3-AC05 | Active t_norm đúng endpoints/N=1, actual timesteps không bị thay | Schedule traces, actual N và t_norm/γ mapping |
| E3-AC06 | Reference bank đủ 124 × 20, training sources only, đúng distinct/cycle policy | Source/permutation/slot manifests, bank counts và hashes |
| E3-AC07 | Reference set 100% CADS synthetic; singleton có centroid nhưng không final augmentation | Role/membership audit và per-class reference/final counts |
| E3-AC08 | Confirmed DINOv3 checkpoint/revision, official processor, frozen FP32 batch 32, final CLS 768 | Verified snapshot/processor hashes/input shape, embedding/centroid identities |
| E3-AC09 | FP32 Eq. (6)/(7) chọn Top-1 trong 123 lớp khác target; exact ties lấy class ID nhỏ nhất, fail invalid/zero norms | NumPy cosine matrix, class ordering, target/confuser map và numerical audit |
| E3-AC10 | `d_pos` dùng CFG output; squared L2 sum CHW/sample, raw confuser cho `d_neg` | Eq. (8) trace, float32 norms/scales; không batch reduction hoặc mean substitution |
| E3-AC11 | Positive/confuser CADS noises độc lập, rescale statistics riêng | Condition seeds/RNG streams và resolved convention evidence |
| E3-AC12 | Eq. (9)–(11) đúng, dynamic τ, α=0.8, một scheduler.step/active timestep | Guidance/scheduler traces; không latent/image interpolation thay noise interpolation |
| E3-AC13 | A3 chỉ thêm prompt-based Geo; artifact-negative baseline đúng Eq. (5)/(8)/(9) | Rendered conditions/token audit/config hashes và cached confuser identity |
| E3-AC14 | Giữ master prompts, audit mọi token lengths trước generation; overflow được báo trước khi có sửa đổi | Locked tokenizer audit; mọi shortening cần xác nhận revision riêng, không silent truncation |
| E3-AC15 | A2 bank/features/centroids/confusers cache nguyên ở A3/A4; refs không vào training | Cache dependency hashes và reference/final membership audit |
| E3-AC16 | A4 dùng đúng A3 pool, không new synthetic; cùng anchors/epochs/optimizer steps | Canonical manifest/content hashes và per-run counts/step logs |
| E3-AC17 | A4 chọn floor(0.5*L) batches, chỉ eligible synthetic thay bằng same-class mix | Batch/pair/role evidence, real partner IDs thuộc train cùng label |
| E3-AC18 | λ uniform discrete 21 values, batch selection uniform không hoàn lại, partners cùng class có hoàn lại, Mixup trước normalization | Independent seeded mixing RNG/config, batch/partner/lambda/label audit và resume state |
| E3-AC19 | Real/E1/head/medium/singleton không bị mix; oversampling chỉ phục vụ partner lookup | Anchor traversal và membership/pairing audit |
| E3-AC20 | Technical-only QC, first-valid ≤5 attempts/slot, đủ quotas; không semantic filters | Candidate/attempt/shortage/leakage reports và pool locks |
| E3-AC21 | 12 E3 official runs đúng common seeds/init/runtime/evaluation/checkpoint policy | Run matrix/identity/history/best locks và per-seed/aggregate outputs |
| E3-AC22 | Resume giữ pool/cache/attempt/RNG identity, A4 mixing state đủ tái lập | Generation và classifier resume histories, lock verification |
| E3-AC23 | G01–G10 implemented đúng confirmed policy; verification chưa thực hiện không được ghi thành đã pass | Decision records, resolved configs và evidence V01–V04 trước bước liên quan |
| E3-AC24 | G01 population std FP32 trên T,D/all tokens, Gaussian G02 redraw mỗi timestep; fail invalid/zero std | Per-embedding/branch statistics và seeded noise identities |
| E3-AC25 | Paired source/VAE/initial noise theo slot/attempt E2/A1/A2/A3; reference/CADS namespaces riêng | Source permutation/seed/initial latent hashes, accepted attempt indices |
| E3-AC26 | Reference–final exact duplicate bị reject/retry dưới cùng cap D09 | Frozen reference-bank hash dependency, collision reasons và slot attempt ledger |
| E3-AC27 | G08 source comparison và Geo adaptations được ghi rõ; không thay master equations để khớp upstream | Pinned official source comparison và unchanged formula/config checks |
| E3-AC28 | Feature arrays `.npy` FP32 và JSONL per-timestep đủ identity/row mapping/hash locks | Artifact schemas/shapes, guidance traces và resume consistency evidence |

Không đặt ngưỡng Accuracy/F1 hoặc geographic fidelity score để nghiệm thu. Saturation, singleton centroid ít cảnh, few-shot evaluation và prompt-based preservation là các giới hạn phải ghi nhận, không phải lý do để tự đổi công thức hoặc lọc lại pool sau khi thấy kết quả.

## 13. Confirmed decisions và verification requirements

### 13.1. G01–G10 — CONFIRMED

Các quyết định dưới đây đã được người dùng xác nhận; không còn pending policy trong G01–G10 và không có tuning parameters mới. Verification/materialization còn thiếu không được ghi thành đã hoàn tất chỉ vì policy đã confirmed.

| ID | Quyết định đã confirmed | Đặc tả chi tiết |
|---|---|---|
| G01 | Per-embedding mean/std trên token/features, FP32, population std, include all tokens; fail invalid/zero std | Mục 4.3 |
| G02 | Standard Gaussian cùng embedding shape, redraw mỗi timestep; positive/confuser streams độc lập | Mục 4.3 và 8.1 |
| G03 | Pair sources/VAE/initial latent noise E2/A1/A2/A3 theo slot/attempt; namespace reference và CADS riêng | Mục 8.1 |
| G04 | `facebook/dinov3-vitb16-pretrain-lvd1689m`, official processor, frozen FP32, batch 32; revision `5931719e67bbdb9737e363e781fb0c67687896bc` | Mục 5.2; snapshot content verification V01 chưa hoàn tất |
| G05 | FP32 cosine, smallest class ID cho exact ties; fail invalid/zero norms | Mục 5.3 |
| G06 | Giữ master prompts, audit tất cả token lengths trước generation; report overflow trước khi sửa | Mục 7.4; token audit V02 chưa chạy |
| G07 | Uniform batch selection không hoàn lại; uniform same-class partners có hoàn lại; Mixup trước normalization, independent seeded RNG | Mục 9.3 |
| G08 | Inspect official source, ghi Geo-specific differences và giữ nguyên master convention | Source inspection hoàn tất tại mục 13.2; chưa runtime reproduction |
| G09 | Reject reference–final exact duplicates, retry cùng slot theo D09, cap không đổi | Mục 10.1 |
| G10 | NumPy `.npy` FP32 features và JSONL guidance diagnostics cho mỗi active timestep | Mục 11.2 |

### 13.2. G08 — Official DiSC-DS source inspection

Đã inspect repository [hvcl/DiSC_DS](https://github.com/hvcl/DiSC_DS) tại immutable commit **`5d0585a9a007c07f1742d5fd187528872d6c8e64`**. Các nguồn được đọc: [README](https://github.com/hvcl/DiSC_DS/blob/5d0585a9a007c07f1742d5fd187528872d6c8e64/README.md), [sampler.py](https://github.com/hvcl/DiSC_DS/blob/5d0585a9a007c07f1742d5fd187528872d6c8e64/sampling/sampler.py), [main.py](https://github.com/hvcl/DiSC_DS/blob/5d0585a9a007c07f1742d5fd187528872d6c8e64/sampling/main.py), [utils.py](https://github.com/hvcl/DiSC_DS/blob/5d0585a9a007c07f1742d5fd187528872d6c8e64/sampling/utils.py) và danh sách dataset CSVs. Đây là **static source inspection**, không phải đã chạy upstream sampler hoặc chứng minh numerical parity.

| Chủ đề | Quan sát từ source đã pin | Hành vi Geo phải giữ |
|---|---|---|
| Negative-condition perturbation | Sampler gọi noise perturbation riêng cho positive và negative khi γ khác 1, baseline không bị perturb | Giữ corrupt cả hai conditions, riêng statistics/streams; G02 redraw chuẩn Gaussian mỗi timestep. Phần không redraw ở γ=1 của upstream không được dùng để đổi G02 |
| CADS statistics/rescaling | Helper lấy global statistics từ original tensor trước perturbation, dùng std mặc định của PyTorch; nhánh rescale không thực hiện normalization bằng statistics của corrupted embedding rồi restore original std như Eq. (3) master | Dùng nguyên Eq. (3)/(4) master và G01 per-embedding population std FP32/all tokens; đây là Geo numerical realization, không copy helper upstream |
| Annealing direction | Sampler đưa index i tăng dần vào schedule, với hai mốc được nhân configured steps; γ ở đầu bằng 1 rồi giảm | Geo dùng decreasing active `t_norm`, γ đầu bằng 0 và cuối bằng 1 khi N>1, theo master; giữ N=1 special case. Không đổi active normalization để khớp upstream |
| Positive guidance distance | Distance phía positive được tính từ raw conditional prediction so với baseline trong source | Geo Eq. (8) dùng **CFG-combined `ε_y+`**, squared L2 sum CHW/sample; ghi rõ khác upstream, không sửa về raw prediction |
| Dynamic temperature | Upstream có bước ψ-blending/clamping sau temperature schedule | Với ψ=1 và τ=0.8, giá trị tương ứng nằm trong [0,0.8]; Geo giữ đúng Eq. (10), không thêm clamp hoặc tham số mới |
| Noise interpolation | Default coefficient phía refined/CCFG là 0.2, phần còn lại là ordinary CADS/CFG | Effective weights khớp Eq. (11): 0.8 CADS + 0.2 CCFG; không suy nhầm tên biến upstream thành α=0.2 |
| Backbone/geometry/strength | Entry point load SD1.4; preprocessing resize/crop ảnh về square 512; sampler không có selected-strength argument hay Geo active-step contract | Geo giữ confirmed SD1.5, native reflection pad/crop, locked E2 strength và E2 scheduler realization; không import defaults upstream |
| Confuser/mining/Mixup | Published entry point đọc food class-similarity CSV; các files đã inspect không triển khai DINOv3 reference mining hoặc classifier Mixup | Geo A2 reference/DINOv3 pipeline và A4 same-class Mixup là adaptations theo master, không khẳng định có sẵn trong upstream sampler |
| Reproducibility | Helper nhận seed nhưng chỉ cấu hình cuDNN flags; không thể xem đó là cơ chế đủ để phục hồi RNG/paired jobs | Dùng confirmed Common/E1E2 và G02/G03/G07 seeded streams/checkpoint provenance |

Các nhận xét trên hỗ trợ xác minh convention và báo cáo khác biệt; chúng **không thay equations, hyperparameters, prompts, sampler hoặc ablation flags**. Những phần Geo khác upstream phải được ghi là adaptations trong implementation metadata/report, không tuyên bố là reproduction nguyên xi của repository. Không dùng kết quả runtime/test để chọn xem phiên bản công thức nào có score tốt hơn.

### 13.3. Verification chưa hoàn tất — không phải policy cần xác nhận lại

| ID | Trạng thái hiện tại | Bằng chứng phải có trước bước liên quan |
|---|---|---|
| V01 — DINOv3 snapshot/processor | **Revision đã resolve; file contents chưa verified.** Metadata public có SHA/gated status, nhưng pinned config/processor endpoints không truy cập được trong phiên này; chưa tải weights | Snapshot đúng revision bằng access hợp lệ, file SHA-256 cho model/config/processor, actual official-processor resize/crop/normalization/H×W và FP32 batch-32 extraction identity. Không fallback model/processor |
| V02 — Prompt token audit | **Chưa chạy** vì chưa materialize locked SD1.5 tokenizer trong task viết spec | Token IDs/counts gồm special tokens cho toàn bộ rendered prompts, overflow report nếu có. Nếu overflow thì chặn generation liên quan và cần xác nhận prompt revision riêng **trước khi sửa**, không coi G06 là chấp thuận sẵn một shortened template |
| V03 — Numerical/runtime conformance | **Chưa có code hoặc runtime evidence**; G08 mới là static inspection | Khi triển khai: evidence Eq. (1)–(11), G01 zero-std failure, G02 redraw/independence, G03 seed/source/noise pairing, G05 tie/failure handling, G07 pre-normalization mixing và G09 retry cap đúng confirmed spec; không claim upstream parity |
| V04 — Artifact/lock materialization | **Chưa tạo execution artifacts**; source revision G08 và model revision G04 đã ghi, nhưng actual outputs/config locks chưa có | Immutable E2/Geo model/library identities, frozen reference hash index trước final acceptance, `.npy` row mappings/dtypes/shapes, per-timestep JSONL, Mixup RNG state và resume/pool locks. Không publish pool/run chỉ từ policy confirmation |

Model access, token overflow và runtime evidence là verification requirements cụ thể, không mở lại G01–G10 hoặc cho phép tuning thêm. Mọi thay đổi chính sách sau này cần version/decision record mới. Spec này không triển khai code, tải model weights, rút gọn prompts hoặc khởi chạy experiments.
