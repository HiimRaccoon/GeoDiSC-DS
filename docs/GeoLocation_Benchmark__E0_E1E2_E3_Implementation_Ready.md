# GeoLocation Benchmark Specification

> **Workflow (2026-11-28):** Implementation completion means the production implementation exists and required local unit/integration tests pass, including fixture or mock tests where needed. Deployment verification means actual UIT environment, data, models, resources, recovery and shared-code runtime checks have passed with evidence. Scientific execution completion means the required official runs, evaluations and report locks exist and verify. Pending deployment evidence never blocks writing or locally testing independent implementation; it continues to gate the corresponding real artifact publication, jobs and scientific stages. Full scientific execution requires separate user approval. Confirmed research requirements and acceptance rules are unchanged.

## E0 vs E1+E2 vs E3 – Our Method: Geo-DiSC-DS

**Project:** CS406 – GeoLocation / GeoGuessr Country Classification\
**Dataset:** GeoGuessr, 49,997 JPG images, 124 country/territory classes\
**Primary task:** 124-class country classification from street-level images\
**Primary metric:** Macro F1\
**Downstream classifier:** EfficientNet-B0\
**Last updated:** 2026-11-26

---

# I. Tổng quan 3 benchmark chính

Benchmark chính của dự án gồm **3 training conditions**:

| Benchmark | Classes ≥100 | Classes 20–99 | Classes <20 | Vai trò |
|---|---|---|---|---|
| **E0 – Original** | Original | Original | Original | Baseline không augmentation |
| **E1 + E2 – Hybrid Augmentation Baseline** | Original | **Traditional Augmentation (E1)** | **Base Diffusion (E2)** | Baseline augmentation |
| **E3 – Our Method: Geo-DiSC-DS** | Original | **Traditional Augmentation giống E1** | **Geo-DiSC-DS** | Proposed method |

Các nhóm trên dựa vào `n_train`. Ngoại lệ chung: class có `n_original = 1` luôn giữ nguyên một ảnh real trong train, không sinh final training synthetic trong E1+E2 hoặc E3.

Sơ đồ tổng quát:

```text
                         GeoGuessr
                  49,997 images / 124 classes
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
             E0             E1 + E2            E3
          ORIGINAL       HYBRID BASELINE     OUR METHOD
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                       EfficientNet-B0
                         same split
                        same training
                         same seeds
                              │
                              ▼
                         same test set
```

## 1. Research questions

### RQ1 – Augmentation có giúp long-tailed GeoLocation không?

```text
E0
vs
E1 + E2
```

### RQ2 – Geo-DiSC-DS có xử lý extreme-tail tốt hơn Base Diffusion không?

```text
E1 + E2
vs
E3-A4
```

Trong E1+E2 và E3:

- classes `≥100`: giống nhau;
- classes `20–99`: cùng dùng E1;
- classes `<20` có `n_original >= 2`: E2 là Base Diffusion, còn E3 bổ sung dần CADS, CCFG, Geo-preservation và Mixup;
- singleton classes (`n_original = 1`): giữ nguyên một ảnh real trong mọi training condition.

---

# II. Cấu hình chung cho toàn bộ benchmark

## 1. Split policy

Split được tạo **một lần trước mọi augmentation** và giữ cố định cho E0, E1+E2 và E3.

| Số ảnh original của class | Train | Validation | Test |
|---|---:|---:|---:|
| `n >= 10` | khoảng 80% | khoảng 10% | khoảng 10% |
| `3 <= n <= 9` | `n - 2` | 1 | 1 |
| `n = 2` | 1 | 0 | 1 |
| `n = 1` | 1 | 0 | 0 |

Quy tắc:

- augmentation chỉ sinh từ **training originals**;
- synthetic/traditional augmented images chỉ nằm trong training set;
- validation và test chỉ chứa ảnh real;
- không dùng validation/test làm nguồn augmentation;
- mọi benchmark dùng cùng một fixed split.

### 1.1. Singleton class policy

`n_original` là tổng số ảnh real của class trước khi split; `n_train` là số ảnh real trong training split trước augmentation.

| Điều kiện | Train | Validation | Test | Final training augmentation |
|---|---:|---:|---:|---|
| `n_original = 1` | 1 | 0 | 0 | Không |
| `n_original = 2` | 1 | 0 | 1 | Có, theo E2/E3 |
| `n_original = 3` | 1 | 1 | 1 | Có, theo E2/E3 |
| `n_original >= 4` | Theo split policy | Theo split policy | Theo split policy | Theo nhóm `n_train` |

Quy tắc áp dụng nhất quán cho E0, E1+E2 và E3-A1 đến E3-A4:

- Giữ singleton class trong training set và giữ classifier output đủ 124 classes.
- Không sinh final synthetic training images cho singleton class; không sử dụng singleton class trong real–synthetic Mixup.
- Không đồng nhất `n_original = 1` với `n_train = 1`: class có 2 hoặc 3 ảnh original vẫn đủ điều kiện diffusion augmentation.
- Trong Stage 1 của E3-A2 trở đi, singleton class vẫn có 20 CADS-generated reference images và centroid để làm ứng viên confusing country. Reference images chỉ dùng cho confuser mining, không đưa vào training set.
- Centroid của singleton class chỉ xuất phát từ một cảnh real, nên độ đại diện địa lý có thể hạn chế.

## 2. Downstream classifier cố định

| Parameter | Giá trị |
|---|---|
| Model | `EfficientNet-B0` |
| Weights | `IMAGENET1K_V1` |
| Fine-tuning | Full backbone |
| Classifier | `1280 → 124` |
| Dropout | 0.2 |
| Input | `896 × 384` landscape (W × H), tensor `[B, 3, 384, 896]` |
| Preprocessing | Preserve aspect ratio + center letterbox + ImageNet normalization |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Epochs | 20 |
| Scheduler | 1 epoch linear warmup → cosine |
| Warmup start factor | 0.1 |
| Minimum LR | `1e-6` |
| Loss | CrossEntropyLoss |
| Label smoothing | 0 |
| Class weights | Không |
| Weighted sampler | Không |
| Focal loss | Không |
| Default Mixup/CutMix | Không, trừ E3-A4 |
| Batch size | **128 (fixed)** |
| Precision | **FP32** |
| AMP / autocast | **OFF** |
| Training seeds | `[42, 123, 2026]` |
| Split seed | 42 |
| Augmentation seed | 314159 |

Batch size được **chốt cố định = 128** cho toàn bộ benchmark.

```text
batch_size = 128
```

Không dùng batch-size fallback trong final benchmark.

Center-letterbox formulas và artifact locking theo [Common Spec](specs/00_common.md). Các tham số training khác giữ nguyên; SD1.5 native-resolution generation và DINOv3 processor độc lập với classifier input.

## 3. Metrics

### Primary

```text
Macro F1
```

#### Evaluation support policy (Validation và Test)

Với mỗi evaluation split `S`, xác định tập class có ít nhất một ground-truth real sample:

```text
C_eval(S) = {c ∈ C : support_S(c) > 0}
Macro F1(S) = sum(F1_c(S) for c in C_eval(S)) / |C_eval(S)|
F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
```

- Chỉ tính trung bình trên các class có ground-truth support trong chính split đang đánh giá; không gán F1 = 0 cho class không có ground-truth chỉ để đưa vào mẫu số.
- Singleton classes không có mặt trong validation/test nên không được tính trực tiếp vào Macro F1 của hai split.
- Classifier vẫn dự đoán trên đủ 124 classes, không mask logits của class không có evaluation support. Dự đoán nhầm sang singleton class vẫn làm giảm Accuracy và F1 của class ground-truth tương ứng.
- Tập evaluated classes được xác định từ ground truth của fixed split, không phụ thuộc vào prediction, benchmark hoặc training seed.
- Báo cáo số class có support trên từng split và từng nhóm; per-class metrics của class không có support được ghi N/A.

#### Group-wise Macro F1 và Recall

Nhóm head/medium/extreme-tail được xác định bằng `n_train` trước augmentation. Với mỗi nhóm `G`:

```text
C_eval_G(S) = C_eval(S) ∩ G
Group Macro F1(S) = mean(F1_c(S) for c in C_eval_G(S))
Group Macro Recall(S) = mean(Recall_c(S) for c in C_eval_G(S))
```

Tính TP/FP/FN trên toàn bộ evaluation split (không lọc các sample ngoài nhóm trước khi tính per-class metrics). Nếu nhóm không có class được hỗ trợ, báo N/A. Tail Macro F1 áp dụng cùng policy ở validation (strength tuning) và test (final benchmark).

### Secondary / diagnostic

- Top-1 Accuracy
- Top-5 Accuracy
- Weighted F1
- Balanced Accuracy
- Per-class F1
- Per-class Recall
- Confusion Matrix
- Group-wise metrics:
  - head (`>=100`)
  - medium (`20–99`)
  - extreme-tail (`<20`)

Mỗi benchmark chạy 3 training seeds và báo:

```text
mean ± sample standard deviation
```

---

# III. E0 – Original Baseline

## 1. Mục tiêu

E0 đo performance khi **không can thiệp vào long-tail distribution**.

```text
Original training set
        ↓
EfficientNet-B0
        ↓
124-country classifier
```

## 2. Data policy

```text
>=100    → giữ nguyên
20–99    → giữ nguyên
<20      → giữ nguyên
```

Không có:

- traditional offline augmentation để rebalance;
- diffusion generation;
- weighted loss;
- oversampling;
- class-balanced sampler;
- Mixup/CutMix.

---

# IV. E1 + E2 – Hybrid Augmentation Baseline

E1 và E2 là hai module augmentation cho hai mức thiếu dữ liệu khác nhau và được **gộp thành một training condition**.

```text
>=100
→ Original

20–99
→ E1 Traditional Augmentation

<20 và n_original >= 2
→ E2 Base Diffusion

n_original = 1
→ Original only
```

---

## 1. E1 – Traditional Augmentation cho classes 20–99

### 1.1. Target

Với:

```text
20 <= n_train <= 99
```

augment đến:

```text
target = 100 images/class
```

### 1.2. Augmentation operators

Mỗi candidate chọn ngẫu nhiên:

```text
k ∈ {1, 2}
```

transform không lặp.

| Transform | Parameter |
|---|---|
| Brightness | `Uniform(0.90, 1.10)` |
| Contrast | `Uniform(0.90, 1.10)` |
| Gaussian blur | `sigma ~ Uniform(0.10, 0.60)`, kernel `5×5` |
| Gaussian noise | `sigma ~ Uniform(1,3)/255` |
| Rotation | `Uniform(-3°, +3°)` |
| Rotation interpolation | Bilinear |
| Rotation padding | Reflection |

Không dùng:

- horizontal flip;
- vertical flip;
- rotation lớn;
- aggressive crop;
- strong affine/perspective;
- strong hue shift.

Mục tiêu là tránh phá traffic side, road geometry, signage và các geographic cues khác.

---

## 2. E2 – Base Diffusion

E2 là **base generative augmentation** cho các extreme-tail classes có `n_train < 20` và `n_original >= 2`. Singleton classes (`n_original = 1`) không sinh final training synthetic.

E2 đồng thời là **Base của E3**. Khi làm E3 ablation, kết quả của E2 được reuse trực tiếp; không cần tạo thêm E3-A0.

### 2.1. Target count

Với extreme-tail classes (`n_train < 20`), target tổng số ảnh training và số final synthetic cần sinh:

```text
if n_original == 1:
    target_c = n_train = 1
    n_synthetic = 0
else:
    target_c = min(100, 5 × n_train)
    n_synthetic = target_c - n_train
```

Ví dụ:

```text
n_original=1,  n_train=1  → target=1   → 0 synthetic
n_original=2,  n_train=1  → target=5   → 4 synthetic
n_original=3,  n_train=1  → target=5   → 4 synthetic
n_original>=2, n_train=5  → target=25  → 20 synthetic
n_original>=2, n_train=10 → target=50  → 40 synthetic
n_original>=2, n_train=19 → target=95  → 76 synthetic
```

Target count và singleton exception được giữ giống nhau giữa E2 và E3-A1 đến E3-A4. Quy tắc này không áp dụng cho Stage 1 reference set.

### 2.2. Paper-aligned Base Diffusion configuration

E2 được cấu hình bám theo generation setup của DiSC-DS paper ở các parameter mà paper công bố.

| Parameter | E2 Final |
|---|---|
| Diffusion backbone | **Stable Diffusion v1.5** |
| Generation mode | Image-conditioned diffusion / img2img-style |
| Sampler | **DPM-Solver++** |
| Configured inference steps | **50** |
| CFG guidance `w` | **2.0** |
| Textual Inversion | OFF |
| LoRA | OFF |
| CADS | OFF |
| CCFG | OFF |
| DINOv3 | OFF |
| Mixup | OFF |
| Diffusion generation batch size | **1** |
| Tuned hyperparameter | **strength only** |

Điểm quan trọng:

```text
E2
=
Base Diffusion
=
starting point của E3 ablation
```

### 2.3. Native-resolution pipeline

GeoLocation giữ native resolution của cleaned images.

```text
1536×662 cleaned image
        ↓
reflection-pad tối thiểu để /8
        ↓
1536×664
        ↓
SD1.5 image-conditioned diffusion
        ↓
crop padding
        ↓
1536×662 synthetic
```

Với loại ảnh còn lại:

```text
1030×703
   ↓
1032×704
   ↓
SD1.5
   ↓
crop
   ↓
1030×703
```

Tóm tắt:

```text
cleaned native image
       ↓
minimal reflection pad /8
       ↓
VAE encode
       ↓
add noise / denoise
       ↓
SD1.5 + DPM-Solver++
       ↓
VAE decode
       ↓
crop exact padding
       ↓
synthetic image cùng size gốc
```

`896×384` chỉ dùng ở classifier preprocessing; diffusion vẫn dùng native cleaned resolution và crop padding như trên.

### 2.4. Strength – hyperparameter duy nhất được tune ở E2

E2 chỉ fine-tune **một hyperparameter duy nhất** dựa trên Tail Macro-F1 trên Validation.
Tail Macro F1 = mean F1 của các classes có n_train < 20
                và có ground-truth real sample trong validation set.
Áp dụng evaluation support policy tại II.3; test dùng cùng quy tắc với ground truth của test split.

```text
strength = {0.25, 0.35, 0.5}
```

Mục tiêu là tìm một mức noise đủ để tạo variation nhưng vẫn giữ geographic structure của ảnh nguồn.

Các parameter còn lại của E2 được khóa:

```text
Stable Diffusion v1.5
DPM-Solver++
50 configured inference steps
CFG w = 2.0
native resolution
diffusion generation batch size = 1
```

Quy trình:

```text
E2 strength search
      ↓
evaluate trên fixed validation protocol
      ↓
chọn 1 strength tốt nhất
      ↓
LOCK
      ↓
reuse exact same strength cho:
E3-A1
E3-A2
E3-A3
E3-A4
```

Sau khi strength được chọn ở E2, **không tune lại strength trong E3**.

Nhờ vậy:

```text
E2 → A1
```

chỉ thêm CADS;

```text
A1 → A2
```

chỉ thêm DINOv3 Visual Confuser Mining + CCFG;

```text
A2 → A3
```

chỉ thêm Geo-preservation;

```text
A3 → A4
```

chỉ thêm Real/Synthetic Mixup.

### 2.5. Base prompt

E2 không dùng Geo-preservation prompt quá mạnh, để contribution của E3-A3 vẫn được isolate.

Prompt cơ bản:

```text
A realistic street-level road photograph in {country_name}, look like picture in Google Street View.
```


### 2.6. Final E1+E2 training condition

```text
Original >=100
      +
Traditional augmented 20–99
      +
Base Diffusion augmented <20 (n_original >= 2)
      +
Singleton original images (n_original = 1)
      ↓
EfficientNet-B0
```

---

# V. E3 – Our Method: Geo-DiSC-DS

E3 là proposed method.

E3 giữ:

```text
>=100 → Original
20–99 → Traditional Augmentation giống E1
n_original = 1 → Original only
```

và phát triển trực tiếp từ Base Diffusion E2 cho nhánh `n_train < 20` và `n_original >= 2`. Singleton classes không sinh final training synthetic ở bất kỳ ablation nào.

```text
E2 Base
   │
   │ + CADS
   ▼
E3-A1
   │
   │ + DINOv3 Visual Confuser Mining
   │ + CCFG
   ▼
E3-A2
   │
   │ + Geo-preservation
   ▼
E3-A3
   │
   │ + Real/Synthetic Mixup
   ▼
E3-A4
```

---

## 1. Base – E2

Base của E3 chính là **E2 đã trình bày ở IV.2**.

Không tạo một `E3-A0` riêng.

```text
Base of E3
=
E2 Base Diffusion
```

Base configuration:

```text
Stable Diffusion v1.5
DPM-Solver++
50 configured inference steps
CFG w = 2.0
native resolution
no CADS
no CCFG
no DINOv3
no Geo-preservation nâng cao
no Mixup
```

Kết quả E2 được reuse trực tiếp làm điểm xuất phát cho E3 ablation.

---

## 2. E3-A1 – + CADS

A1 giữ toàn bộ E2 và thêm **Condition-Annealed Diffusion Sampler (CADS)**.

### 2.1. Mục tiêu

Tăng:

```text
intra-class diversity
```

mà vẫn duy trì class conditioning.

### 2.2. Cấu trúc

```text
E2 Base
   ↓
CADS
   ↓
more diverse synthetic samples
```

### 2.3. CADS parameters tham khảo từ paper

| Parameter | Paper value | E3-A1 |
|---|---:|---:|
| Guidance scale `w` | **2.0** | **2.0** |
| `τ1` | **0.5** | **0.5** |
| `τ2` | **0.9** | **0.9** |
| Condition-noise scale `s` | **0.1** | **0.1** |
| Rescale factor `ψ` | **1.0** | **1.0** |
| Configured inference steps | **50** | **50** |
| Solver | **DPM-Solver++** | **DPM-Solver++** |

Condition corruption – Equation (1):

```text
ŷ+ = sqrt(γ(t)) y+ + s sqrt(1-γ(t)) n
```

Annealing schedule – Equation (2):

```text
γ(t) = 1                       if t <= τ1
       (τ2-t)/(τ2-τ1)          if τ1 < t < τ2
       0                       if t >= τ2
```

#### Active timestep normalization

GeoLocation sử dụng img2img với strength thấp.
Nếu chuẩn hóa timestep theo toàn bộ diffusion schedule,
CADS có thể không thêm condition noise.

Do đó, Geo sử dụng active denoising schedule:

```text
N = number of active denoising timesteps
i = current denoising step index, starting from 0

t_norm = 1 - i / (N - 1), if N > 1
t_norm = 0,               if N = 1

γ = gamma(t_norm)
```

- t_norm = 1 tại bước denoising đầu tiên.
- t_norm = 0 tại bước cuối cùng.
- Chỉ dùng t_norm cho CADS và dynamic CCFG temperature.
- U-Net và DPM-Solver++ vẫn sử dụng scheduler timestep gốc.
- Áp dụng nhất quán cho A1–A4 và Stage 1 reference generation.

Đây là Geo-specific adaptation của CADS cho img2img,
không phải thiết lập được xác nhận trực tiếp từ paper.

#### 2.3.1. CADS condition rescaling – Equation (3)

Sau khi positive condition được perturb bởi CADS, condition được rescale để đưa mean và standard deviation về gần distribution của positive condition gốc:

```text
y_rescaled =
    ((ŷ+ - mean(ŷ+)) / std(ŷ+)) * σ_in + μ_in
```

Trong đó:

```text
μ_in = mean của original positive condition y+
σ_in = standard deviation của original positive condition y+
```

Bước này giúp giữ diversity do condition perturbation tạo ra nhưng hạn chế việc embedding bị lệch quá xa khỏi statistical distribution ban đầu.

#### 2.3.2. Mixing bằng ψ – Equation (4)

Final CADS condition được tính bằng:

```text
ŷ+ = ψ · y_rescaled + (1 - ψ) · ŷ+
```

Với cấu hình hiện tại:

```text
ψ = 1.0
```

Do đó:

```text
ŷ+ = y_rescaled
```

Tức là pipeline sử dụng hoàn toàn rescaled condition sau bước CADS perturbation.

`ψ` điều khiển trade-off giữa stability và diversity:

```text
ψ → 1.0
→ ưu tiên rescaled condition
→ ổn định hơn

ψ → 0.0
→ ưu tiên noisy condition gốc
→ diversity cao hơn nhưng risk drift lớn hơn
```

#### 2.3.3. Classifier-Free Guidance (CFG) – Equation (5)

Sau khi có CADS-conditioned positive condition, noise prediction được kết hợp với unconditional prediction bằng CFG:

```text
ε_y+ = ε_∅ + w(ε_ŷ+ - ε_∅)
```

Với:

```text
w = 2.0
```

Trong đó:

```text
ε_∅   = unconditional noise prediction
ε_ŷ+  = noise prediction conditioned on the CADS-adjusted positive condition
```

CFG giúp kéo quá trình denoising về đúng target class sau khi CADS đã tăng diversity của condition.

Tóm tắt flow:

```text
positive prompt
      ↓
text condition y+
      ↓
CADS perturbation
      ↓
noisy condition ŷ+
      ↓
condition rescaling
      ↓
ψ mixing
      ↓
final CADS condition + noisy latent x_t
      ↓
U-Net noise predictions
      ↓
unconditional ε_∅ and conditional ε_ŷ+
      ↓
CFG with w = 2.0
      ↓
DPM-Solver++ scheduler.step
      ↓
next scheduled latent
      ↓
repeat for all active timesteps
      ↓
final latent x_0
      ↓
VAE Decoder
      ↓
crop exact padding
      ↓
synthetic image
```

### 2.4. Ablation question

```text
E2 vs E3-A1
```

> CADS có cải thiện extreme-tail recognition nhờ tăng intra-class diversity không?

---

## 3. E3-A2 – + CCFG

A2 giữ CADS và thêm **Contrastive Classifier-Free Guidance (CCFG)**.

DINOv3 Visual Confuser Mining nằm **bên trong A2**, vì CCFG cần xác định negative/confusing country.

### 3.1. Cấu trúc tổng quát

```text
Tail country c+
      ↓
CADS reference generation
      ↓
Reference images
      ↓
DINOv3 ViT-B/16
      ↓
L2-normalized embeddings
      ↓
country centroid
      ↓
cosine similarity
      ↓
Top-1 confusing country c-
      │
      ├──────────────┐
      │              │
Target country c+    │
      │              │
      └──────┬───────┘
             ↓
            CCFG
             ↓
toward target + away from confuser
             ↓
Synthetic image
```

### 3.2. DINOv3 Visual Confuser Mining

**Model:** DINOv3 ViT-B/16, web-pretrained.

| Parameter | Giá trị |
|---|---|
| Backbone | DINOv3 ViT-B/16 |
| Parameters | ~86M |
| Patch size | 16 |
| Embedding dimension | 768 |
| Fine-tuning | Không |
| Similarity | Cosine similarity |
| Confuser | Top-1 nearest country |

Không dùng satellite weights vì dataset là ground-level Street View.

Feature extraction policy:

```text
RGB reference image
      ↓
DINOv3 checkpoint-specific image processor
      ↓
Frozen DINOv3 ViT-B/16
      ↓
Final CLS embedding (768 dimensions)
      ↓
L2 normalization
      ↓
Country centroid
```

- Model sử dụng eval mode và inference mode.
- Không fine-tune DINOv3.
- Không sử dụng classifier logits hoặc patch-token average.
- Khóa checkpoint revision và image processor configuration.
- DINOv3 preprocessing độc lập với EfficientNet-B0 preprocessing.

#### 3.2.1. Reference set phải là 100% CADS-generated synthetic images

Để bám đúng logic Stage 1 của DiSC-DS paper, **DINOv3 không tính centroid từ tập `real + CADS refs`**.

Thay vào đó:

```text
real images của class
        ↓
chỉ đóng vai trò source / seed cho CADS
        ↓
VAE Encoder
        ↓
latent z
        ↓
add noise theo selected strength / starting timestep
        ↓
CADS-conditioned denoising
        ↓
VAE Decoder
        ↓
N_ref synthetic reference images
        ↓
100% CADS reference set
        ↓
DINOv3
        ↓
N_ref embeddings
        ↓
L2 normalization
        ↓
mean
        ↓
country centroid μ_c
```

Tức là tập đưa vào DINOv3 để xây centroid:

```text
reference_set(c)
=
N_ref CADS-generated synthetic images
```

Không dùng:

```text
real images
+
CADS synthetic refs
```

để tính centroid.

Lý do:

1. bám sát cách paper tạo synthetic reference set trước khi feature extraction;
2. mọi class được biểu diễn bằng cùng một loại reference distribution;
3. tránh trường hợp class này có centroid từ phần lớn ảnh real nhưng class khác lại có centroid từ phần lớn synthetic;
4. comparison giữa các country centroids nhất quán hơn.

#### 3.2.2. Candidate bank gồm toàn bộ classes

Chỉ các class có `n_train < 20` và `n_original >= 2` là **target classes cần tìm negative/confuser để sinh final synthetic training images**. Singleton classes vẫn nằm trong candidate bank với 20 CADS-generated references nhưng không cần chọn confuser cho chính chúng.

Tuy nhiên candidate bank phải bao gồm:

```text
all 124 classes
```

vì confusing country gần nhất của một tail class có thể là medium/head class.

Với **mọi class trong 124 classes**, kể cả class có rất nhiều ảnh real:

```text
real source pool
      ↓
sample / cycle source images
      ↓
VAE Encoder
      ↓
latent z
      ↓
add noise theo selected strength / starting timestep
      ↓
CADS-conditioned denoising
      ↓
VAE Decoder
      ↓
N_ref synthetic reference images
      ↓
DINOv3
      ↓
μ_candidate
```

Ví dụ với USA:

```text
USA real source pool
      ↓
VAE Encoder
      ↓
latent z
      ↓
add noise theo selected strength / starting timestep
      ↓
CADS-conditioned denoising
      ↓
VAE Decoder
      ↓
N_ref synthetic refs
      ↓
DINOv3
      ↓
μ_USA
```

Các CADS reference images này **chỉ dùng cho confuser mining**.

Toàn bộ reference images không được đưa vào classifier training,
kể cả reference images của extreme-tail classes.

```text
124 classes × 20 references = 2,480 reference images
```

Reference bank và confuser mapping được cache từ A2
và reuse trong A3/A4.

#### 3.2.3. Target tail class

Với một target country `c+` có `n_train < 20` và `n_original >= 2`:

```text
real tail source images
      ↓
VAE Encoder
      ↓
latent z
      ↓
add noise theo selected strength / starting timestep
      ↓
CADS-conditioned denoising
      ↓
VAE Decoder
      ↓
N_ref synthetic refs
      ↓
DINOv3
      ↓
μ_target
```

##### 3.2.3.1. Country feature representation

Với mỗi country `c`, `N_ref` CADS-generated reference images được DINOv3 encode thành các feature vectors:

```text
v_1, v_2, ..., v_N_ref
```

Trong đó mỗi v_i là CLS embedding đã L2-normalize.

Mean feature vector (country centroid) được tính bằng:

```text
μ_c = (1 / N_ref) · Σ_i v_i
```

Với:

```text
N_ref = 20
```

`μ_c` được dùng làm visual representation của country `c`.

##### 3.2.3.2. Cosine similarity between countries – Equation (6)

Với target country `c` và một candidate country khác `c'`, visual similarity được tính bằng cosine similarity:

```text
sim(μ_c, μ_c')
=
(μ_c · μ_c') / (||μ_c|| ||μ_c'||)
```

Trong đó:

```text
μ_c   = centroid của target country
μ_c'  = centroid của candidate country
```

Confusing country cho target class `c+` được chọn bằng Equation (7):

```text
c- = argmax_{c' ∈ C \ {c+}} sim(μ_c+, μ_c')
```

Trong đó:

```text
C  = tập tất cả 124 country classes
c+ = target tail country
c- = visually confusing country được chọn
```

Do đó:

```text
μ_target
      ↓
cosine similarity với
123 country centroids còn lại
      ↓
highest similarity
      ↓
Top-1 confusing country c-
      ↓
dùng làm negative condition trong CCFG
```

#### 3.2.4. Phân biệt reference synthetic và final training synthetic

Hai loại ảnh synthetic này có mục đích khác nhau:

```text
Stage 1 – Reference synthetic
CADS-generated
→ chỉ dùng cho DINOv3
→ xây country centroid
→ tìm confusing class
```

và:

```text
Stage 2 – Final training synthetic
target c+
+
confuser c-
+
CADS
+
CCFG
(+ Geo-preservation ở A3)
→ dùng để augment training set
```

Reference images của Stage 1 không được đưa vào training set. Singleton classes chỉ có reference images ở Stage 1, không sinh final training synthetic ở Stage 2.

#### 3.2.5. N_ref

`N_ref` là số **CADS-generated reference images cho mỗi class** dùng để xây country centroid.

Hiện tại:

```text
N_ref = 20
```

Mọi class trong candidate bank phải dùng cùng một giá trị:

```text
N_ref refs / class
```

để đảm bảo cách xây centroid nhất quán.

##### Case 1 – Large class: United States

Suppose the United States has:

    n_real = 12,014
    N_ref = 20

Since `n_real >= N_ref`, randomly sample **20 distinct real source images**:

    USA_real_01
    USA_real_02
    ...
    USA_real_20

Each real source image is passed through CADS once:

    USA_real_01 → CADS → USA_ref_01
    USA_real_02 → CADS → USA_ref_02
    ...
    USA_real_20 → CADS → USA_ref_20

The reference pipeline is:

    20 distinct real source images
            ↓
    CADS
            ↓
    20 CADS-generated reference images
            ↓
    DINOv3 ViT-B/16
            ↓
    20 feature embeddings
            ↓
    L2 normalization
            ↓
    mean
            ↓
    μ_USA

Therefore, the United States centroid is constructed from **20 CADS-generated references originating from 20 different real scenes**.

##### Case 2 – Tail class: `n_real < N_ref`

If a class has fewer than 20 real images:
    1 <= n_real < 20

all available real source images are used and repeatedly reused with a **shuffle-cycle / round-robin policy** until 20 CADS-generated reference images are obtained.

General procedure:

    all available real source images
            ↓
    shuffle source order
            ↓
    cycle through sources repeatedly
            ↓
    each selected source → CADS
            ↓
    generate one new reference image
            ↓
    repeat until total references = N_ref
            ↓
    20 CADS-generated reference images
            ↓
    DINOv3 ViT-B/16
            ↓
    20 feature embeddings
            ↓
    L2 normalization
            ↓
    mean
            ↓
    μ_class


Examples (where `n_real = n_train` for Stage 1 source selection):

    n_real = 1
    → reuse the single source 20 times
    → generate 20 CADS references with distinct noise states
    → if n_original = 1: candidate centroid only; no final training synthetic

    n_real = 3
    → repeatedly cycle through 3 sources
    → approximately 6–7 CADS references per source
    → total = 20

    n_real = 8
    → repeatedly cycle through 8 sources
    → approximately 2–3 CADS references per source
    → total = 20


### 3.3. CCFG

Sau khi xác định target country c+ và DINOv3-selected confuser c-,
CCFG hướng quá trình generation về target và tránh confuser.

#### 3.3.1. Adaptive guidance scales – Equation (8)

```text
d_pos = ||ε_∅ - ε_y+||²₂
d_neg = ||ε_∅ - ε_ŷ-||²₂

w_pos = 2w / (1 + exp(-τ * d_pos))

w_neg = -2w * exp(-τ * d_neg)
        / (1 + exp(-τ * d_neg))
```

Positive guidance kéo về target class.
Negative guidance đẩy ra xa confusing class.

Noise prediction notation:

```text
ε_∅   = U-Net(x_t, t, unconditional condition)
ε_ŷ+  = U-Net(x_t, t, CADS positive condition)
ε_ŷ-  = U-Net(x_t, t, CADS confuser condition)

ε_y+  = ε_∅ + w * (ε_ŷ+ - ε_∅)
```

Theo ký hiệu Eq. (5) và Algorithm 1 của paper,
ε_y+ là CFG output, khác với raw U-Net prediction ε_ŷ+.

Squared L2 norm được tính riêng cho từng sample:

```text
d_pos = sum((ε_∅ - ε_y+)²)
d_neg = sum((ε_∅ - ε_ŷ-)²)
```

Reduce trên các chiều latent C, H, W,
không reduce qua batch dimension.

Stable numerical implementation:

```text
w_pos =  2w * sigmoid(τ_hat * d_pos)
w_neg = -2w * sigmoid(-τ_hat * d_neg)
```

Tính norm và guidance scales bằng float32.
Theo dõi d_pos, d_neg, w_pos và w_neg để phát hiện
sigmoid saturation.

Không tự ý thay squared L2 sum bằng mean
nếu chưa thực hiện controlled experiment.

Geo implementation convention:

- Positive và confusing-country conditions sử dụng
  cùng CADS annealing schedule.
- Condition noise được sample độc lập cho hai condition.
- Rescaling sử dụng mean/std riêng của mỗi embedding.
- Unconditional condition không áp dụng CADS.

Negative-condition perturbation là quy ước triển khai
cần được kiểm chứng với official DiSC-DS implementation.

#### 3.3.2. CCFG noise prediction – Equation (9)

```text
ε_CCFG =
    ε_∅
    + w_pos * (ε_ŷ+ - ε_∅)
    + w_neg * (ε_ŷ- - ε_∅)
```

Confusing country không được nối thủ công
vào artifact negative prompt.

### 3.4. CCFG / DiSC parameters tham khảo từ paper

| Parameter | Paper value | E3-A2 |
|---|---:|---:|
| Guidance scale w | 2.0 | 2.0 |
| CCFG temperature τ | 0.8 | 0.8 |
| CADS weight α | 0.8 | 0.8 |
| Dynamic temperature | τ√γ(t) | same |

#### 3.4.1. Dynamic temperature – Equation (10)

```text
τ_hat(t) = τ * sqrt(γ(t))
```

Thay τ bằng τ_hat(t) trong Equation (8)
tại mỗi denoising timestep.

#### 3.4.2. Linear interpolation – Equation (11)

```text
ε_step = α * ε_CADS + (1 - α) * ε_CCFG

       = 0.8 * ε_CADS + 0.2 * ε_CCFG
```

Kết hợp intra-class diversity của CADS
và inter-class separability của CCFG
ở mức noise prediction.

#### 3.4.3. Algorithm 1 – Geo-DiSC-DS Sampling

Adapted from DiSC-DS Algorithm 1 for img2img:

```text
Real training image
    ↓
Minimal reflection padding /8
    ↓
VAE Encoder
    ↓
Scaled latent z
    ↓
Add noise according to locked strength
    ↓
Initial noisy latent x_t

For each active scheduler timestep:

    1. Compute active t_norm and γ(t_norm) – Eq. (2)
    2. Prepare CADS positive and confuser conditions
       – Eq. (1), (3), (4)
    3. Compute dynamic τ_hat(t) – Eq. (10)
    4. U-Net noise predictions:
       ε_∅, ε_ŷ+, ε_ŷ-
    5. Compute CADS + CFG – Eq. (5)
    6. Compute adaptive guidance scales – Eq. (8)
    7. Compute CCFG prediction – Eq. (9)
    8. Combine CADS and CCFG – Eq. (11)
    9. DPM-Solver++ scheduler.step:
       x_t → next scheduled latent

End loop
    ↓
Final latent x_0
    ↓
VAE Decoder
    ↓
Crop exact padding
    ↓
Final synthetic training image
```

Geo sử dụng img2img thay vì khởi tạo hoàn toàn
từ Gaussian noise như Algorithm 1 của paper.

50 là số inference steps được cấu hình;
số active denoising steps phụ thuộc strength.

CADS condition noise khác với image latent noise.
CADS và dynamic CCFG temperature sử dụng t_norm.
U-Net và scheduler sử dụng timestep gốc.

Mỗi active timestep chỉ gọi scheduler.step một lần,
sau khi kết hợp CADS và CCFG noise predictions.

### 3.5. Ablation question

```text
E3-A1 vs E3-A2
```

> Khi đã có diversity từ CADS, DINOv3-based confuser mining + CCFG có giúp tăng inter-class separability không?

---

## 4. E3-A3 – Full Geo

A3 là **full Geo-DiSC generation method**.

```text
E3-A2
+
Geo-preservation
```

Tức:

```text
E2 Base
+
CADS
+
DINOv3 Visual Confuser Mining
+
CCFG
+
Geo-preserving generation
```

### 4.1. Mục tiêu Geo-preservation

Diffusion được phép thay đổi:

- lighting;
- weather;
- tone/color;
- textures;
- minor environmental details.

Nhưng phải hạn chế drift ở:

- traffic direction;
- road geometry;
- lane markings;
- road-sign structure;
- visible script/text characteristics;
- architecture;
- utility poles/infrastructure;
- vegetation;
- terrain;
- roadside structure.

Objective:

```text
Diversity ↑
Separability ↑

while

Geographic fidelity preserved
```

### 4.2. Geo-aware positive prompt – v1

```text
Realistic street-level road photograph in {country_name}. Authentic local architecture, traffic direction, lane markings, road signs, utility poles, vegetation and terrain. Preserve road geometry and geographic identity. Natural variations in lighting, weather and colors.
```

### 4.3. Artifact negative prompt – v1

```text
illustration, cartoon, CGI, blurry, warped geometry, reversed traffic, incorrect lane markings, fake road signs, altered text, distorted vehicles, unrealistic architecture, watermark, compression artifacts
```

Confusing country vẫn đi qua **DINOv3 + CCFG**, không nối thủ công vào negative prompt.

#### 4.3.1. Prompt integration policy

A3 là prompt-based Geo-preservation.

Không sử dụng geographic consistency loss,
feature constraint hoặc geographic filtering.

```text
Positive condition:
    Geo-aware target-country prompt

Confuser condition:
    Base prompt with DINOv3-selected confusing country

Baseline condition:
    Artifact negative prompt
```

Trong A3, thay unconditional baseline prediction bằng
artifact-negative-conditioned prediction trong Eq. (5), (8), (9).

Đây là Geo-specific integration convention.
Confusing country vẫn là một condition độc lập.

Không thay đổi confuser mapping đã cache từ A2.

Kiểm tra token length bằng SD1.5 tokenizer cho tất cả
country prompts, bao gồm special tokens.

```text
len(tokenizer(prompt, truncation=False).input_ids) <= 77
```

Nếu vượt giới hạn, rút gọn prompt trước khi generation.
Không silent truncation.

### 4.4. Ablation question

```text
E3-A2 vs E3-A3
```

> Geo-preservation có làm DiSC-style generation phù hợp hơn với country geolocation không?

---

## 5. E3-A4 – Mixed

A4 là **Final E3** dùng trong main benchmark.

Generation của A4 giống hệt A3.

Không sinh một synthetic pool mới.

Điểm thêm duy nhất:

```text
Real training samples
        +
A3 Geo-DiSC synthetic samples
        ↓
Real–Synthetic Mixup
        ↓
EfficientNet-B0
```

### 5.1. Mục tiêu

Giảm:

```text
real Street View distribution
        ≠
diffusion-generated distribution
```

và giảm nguy cơ classifier học shortcut:

```text
synthetic-looking
→ tail class
```

thay vì geographic cues thật.

### 5.2. Mixup policy

Theo DiSC-DS paper:

- Mixup được áp dụng ở **50% số batch trong mỗi epoch**.
- Trong các batch sử dụng Mixup, **synthetic samples được pair với real samples**.
- Nếu số lượng real samples không đủ, real samples được **oversample** để có thể sử dụng toàn bộ synthetic samples.

Training schedule:

    50% batches/epoch → real-synthetic Mixup
    50% batches/epoch → normal training

#### GeoLocation adaptation

Trong E3-A4, sử dụng **same-class real–synthetic Mixup** để giảm domain gap giữa ảnh real và synthetic mà không làm mờ ranh giới giữa các country. Singleton classes không có final synthetic samples, nên chỉ tham gia training bằng ảnh real và không được pair trong Mixup.

Ví dụ:

    Real Vietnam
          +
    Synthetic Vietnam
          ↓
        Mixup
          ↓
    Label = Vietnam

Cấu hình được chốt:

    same_class_pairing = ON
    mixup_batch_ratio = 0.5
    lambda_sampling = uniform discrete
    lambda_values = [0.00, 0.05, 0.10, ..., 0.90, 0.95, 1.00]
    real_oversampling = ON when needed
    label = unchanged class label

Với một cặp real–synthetic cùng class:

    x_mix = λ · x_real + (1 - λ) · x_synthetic

Do hai ảnh thuộc cùng một class:

    y_mix = y_class

không cần tạo soft label giữa hai country khác nhau.

Hệ số `λ` được sample độc lập cho từng cặp từ tập:

    {0.00, 0.05, 0.10, ..., 0.90, 0.95, 1.00}

Trong đó:

    λ → 1.0  : ảnh mixed gần real hơn
    λ → 0.0  : ảnh mixed gần synthetic hơn
    λ ≈ 0.5  : real và synthetic đóng góp gần tương đương

Same-class pairing là **GeoLocation-specific adaptation**, không phải thiết lập được DiSC-DS paper chỉ định rõ.


#### 5.2.1. Mixup batch construction

- Shuffle toàn bộ training pool mỗi epoch.
- Chọn floor(0.5 × number_of_batches) batches để Mixup.
- Trong selected batches, chỉ final synthetic samples
  đủ điều kiện real–synthetic Mixup.
- Với mỗi eligible synthetic sample, sample một real
  training image cùng country.
- Cho phép reuse real source khi cần.
- Thay synthetic sample bằng mixed sample.
- Các original real samples trong batch giữ nguyên.
- Head, medium và singleton samples train bình thường.
- Không thêm reference images vào training.
- Không tăng batch size hoặc optimizer steps do pairing.

A3 và A4 sử dụng cùng synthetic pool,
cùng số anchor samples và cùng training epochs.

50% là tỷ lệ batches được chọn,
không phải tỷ lệ chính xác của toàn bộ images được Mixup.

### 5.3. Ablation question

```text
E3-A3 vs E3-A4
```

> Mixup có giảm real–synthetic domain gap và cải thiện generalization trên real validation/test images không?

---

# VI. E3 Ablation Summary

| Variant | Base Diffusion E2 | CADS | DINOv3 + CCFG | Geo-preservation | Mixup | Vai trò |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Base (E2)** | ✅ | ❌ | ❌ | ❌ | ❌ | Base generative baseline |
| **E3-A1 – + CADS** | ✅ | ✅ | ❌ | ❌ | ❌ | Diversity-aware |
| **E3-A2 – + CCFG** | ✅ | ✅ | ✅ | ❌ | ❌ | Diversity + separability |
| **E3-A3 – Full Geo** | ✅ | ✅ | ✅ | ✅ | ❌ | Full proposed generation |
| **E3-A4 – Mixed** | ✅ | ✅ | ✅ | ✅ | ✅ | Final proposed pipeline |

```text
E2 Base
   ↓ + CADS
E3-A1
   ↓ + DINOv3 confuser mining + CCFG
E3-A2
   ↓ + Geo-preservation
E3-A3
   ↓ + Real/Synthetic Mixup
E3-A4 = Final E3
```

---

# VII. Parameter Summary

## 1. Base parameters khóa xuyên suốt E2 → E3-A4

| Parameter | Value |
|---|---|
| Diffusion backbone | **Stable Diffusion v1.5** |
| Solver | **DPM-Solver++** |
| Configured inference steps | **50** |
| CFG guidance `w` | **2.0** |
| Input | **native cleaned resolution** |
| Padding | minimal reflection pad `/8` |
| Output | crop exact padding về original size |
| Diffusion generation batch size | **1** |
| Classifier training batch size | **128 (fixed)** |
| Textual Inversion | OFF |
| LoRA | OFF |

Các parameter này không thay đổi giữa E2 và các E3 ablations. Final diffusion augmentation chỉ áp dụng cho `n_train < 20` và `n_original >= 2`; singleton classes giữ nguyên. Stage 1 candidate bank vẫn đủ 124 classes với 20 references/class.

## 2. Components được thêm dần

| Stage | Added parameters/components |
|---|---|
| **E2 Base** | Base diffusion only |
| **E3-A1** | CADS: `τ1=0.5`, `τ2=0.9`, `s=0.1`, `ψ=1.0` |
| **E3-A2** | DINOv3 ViT-B/16 + Top-1 confuser + CCFG `τ=0.8`, `α=0.8`, dynamic `τ` |
| **E3-A3** | Geo-preservation constraints/prompt |
| **E3-A4** | Real-synthetic Mixup, 50% batches/epoch |
