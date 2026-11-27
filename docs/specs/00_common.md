# 00 — Common Specification: GeoLocation Benchmark

**Phiên bản:** 1.3\
**Ngày:** 2026-11-26\
**Phạm vi:** Quy tắc chung cho E0, E1 + E2 và E3-A1 đến E3-A4.\
**Trạng thái:** Đặc tả để triển khai; chưa tạo split, sinh ảnh hoặc huấn luyện.

## 1. Nguồn yêu cầu và thứ tự ưu tiên

Nguồn chính là [GeoLocation Benchmark Specification](../GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), đặc biệt Section I, II, IV.2.3 và VII.1. Khi có mâu thuẫn, cấu hình và chính sách trong Markdown mới được ưu tiên hơn repository cũ.

Repository tham khảo: [GeoLocation_DataPreprocessing](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing), tại commit [`1806ea80efe535119fbd987e1d860a7a5793ef13`](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing/commit/1806ea80efe535119fbd987e1d860a7a5793ef13). Chỉ kế thừa ba cơ chế:

| Cơ chế | Nội dung kế thừa | Nguồn tham khảo |
|---|---|---|
| Reproducibility | Seed các RNG, deterministic runtime, môi trường/weights có định danh cố định, shared initialization theo seed | [Đặc tả cũ §11.2](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing/blob/1806ea80efe535119fbd987e1d860a7a5793ef13/docs/specs.md), [experiment.py](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing/blob/1806ea80efe535119fbd987e1d860a7a5793ef13/src/geoguessr_benchmark/experiment.py) |
| Checkpoint và Resume | Best/last checkpoint, phục hồi tại epoch boundary, lưu RNG và xác minh run identity | [training.py](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing/blob/1806ea80efe535119fbd987e1d860a7a5793ef13/src/geoguessr_benchmark/training.py) |
| Artifact Locking | Canonical serialization, SHA-256, atomic publication, xác minh dependency trước khi sử dụng | [artifacts.py](https://github.com/HiimRaccoon/GeoLocation_DataPreprocessing/blob/1806ea80efe535119fbd987e1d860a7a5793ef13/src/geoguessr_benchmark/artifacts.py) |

Không chuyển nguyên pipeline hoặc lock cũ sang benchmark mới. Cleaning đã hoàn thành; đầu vào là `data/cleaned_dataset/`. Không triển khai lại cleaning, không yêu cầu chạy lại cleaning manifest của repository cũ, không dùng split/membership/checkpoint của benchmark cũ.

Các lựa chọn bổ sung để yêu cầu có thể triển khai được được ghi rõ trong tài liệu này: cách làm tròn split, thứ tự phân bổ ảnh, định dạng artifact và quy tắc checkpoint. Chúng không thay đổi chính sách benchmark mới. Thuật toán augmentation, prompt, CADS, CCFG và Mixup sẽ nằm trong các spec riêng; tài liệu này chỉ nêu các ràng buộc chung mà chúng phải tuân thủ.

## 2. Mục tiêu và các điều kiện so sánh

Phân loại một ảnh street-level vào **124 quốc gia/vùng lãnh thổ**. Classifier dùng chung là **EfficientNet-B0**; primary metric là **Macro F1 trên các lớp có ground-truth support** trong evaluation split.

| Điều kiện | Head: `n_train >= 100` | Medium: `20 <= n_train <= 99` | Extreme-tail: `n_train < 20`, không phải singleton | Singleton |
|---|---|---|---|---|
| E0 — Original | Original | Original | Original | Original only |
| E1 + E2 — Hybrid baseline | Original | E1 traditional augmentation | E2 Base Diffusion | Original only |
| E3 — Geo-DiSC-DS | Original | Cùng E1 | E3 theo ablation tương ứng | Original only |

E1 và E2 là hai module của **một** training condition. E3-A4 là final E3 trong main comparison. Mỗi condition/variant được huấn luyện độc lập với cùng ba training seeds.

- **RQ1:** Augmentation có giúp bài toán long-tail không? So sánh E0 với E1 + E2.
- **RQ2:** Geo-DiSC-DS có cải thiện extreme-tail so với Base Diffusion không? So sánh E1 + E2 với E3-A4.

Cùng seed phải dùng cùng classifier initial state. Mọi condition dùng cùng real validation/test, class mapping, group assignments, classifier preprocessing và training/checkpoint policy. Số optimizer steps có thể khác vì training pool có số ảnh khác; không mô tả đây là so sánh equal-compute.

### 2.1. Fairness invariants bắt buộc khóa

| Phạm vi | Invariant |
|---|---|
| Shared E0 real pool | Mọi condition/variant chứa đúng cùng toàn bộ real training originals của E0, cùng IDs, labels và content hashes; không bỏ, thay thế hoặc bổ sung real anchors theo condition |
| Shared E1 pool | E1 + E2 và E3-A1–A4 dùng đúng một pool E1 traditional augmented đã khóa, cùng membership và file/pixel hashes; không generate lại theo variant hoặc training seed |
| Tail augmentation budgets | E2 và E3-A1–A4 có cùng quota **theo từng lớp**: singleton giữ 1 real, 0 final synthetic; extreme-tail còn lại có `target_c = min(100, 5*n_train(c))`, `n_synthetic_c = target_c - n_train(c)`; thiếu quota chặn pool freeze, không tự giảm budget |
| Shared selected strength | Một strength được chọn từ E2 validation tuning, khóa và dùng nguyên giá trị cho official E2, E3-A1–A4 và A2 reference generation; không tune lại theo ablation hoặc training seed |
| Cached A2 references | Reference bank A2 gồm 20 CADS references/lớp × 124 lớp, cùng embeddings/centroids và confuser mapping được cache, khóa và dùng lại ở A3/A4; không regenerate/re-mine theo variant hoặc training seed; references không vào training pool |
| A3/A4 pool và compute | A4 dùng chính xác final synthetic pool của A3, cùng membership và content hashes; cùng real/E1 pools, số training anchors, batch size, `drop_last`, 20 epochs và optimizer steps. Mixup pairing không thêm anchors hoặc optimizer steps |

Với `N` anchors chung của A3/A4, mỗi run có `ceil(N/128)` optimizer steps/epoch và `20*ceil(N/128)` steps tổng cộng. Real samples được reuse để pairing ở A4 không được tính thành anchors bổ sung. Tính đồng nhất pool phải được xác minh bằng manifest/content hashes, không chỉ bằng counts hoặc cùng generation seed.

### 2.2. Experimental matrix chính thức

| Official condition | Seed 42 | Seed 123 | Seed 2026 | Số classifier runs |
|---|---|---|---|---:|
| E0 — Original | E0/42 | E0/123 | E0/2026 | 3 |
| E1 + E2 — Hybrid baseline | E1E2/42 | E1E2/123 | E1E2/2026 | 3 |
| E3-A1 | A1/42 | A1/123 | A1/2026 | 3 |
| E3-A2 | A2/42 | A2/123 | A2/2026 | 3 |
| E3-A3 | A3/42 | A3/123 | A3/2026 | 3 |
| E3-A4 — Final E3 | A4/42 | A4/123 | A4/2026 | 3 |
| **Tổng: 6 conditions × 3 seeds** | **6** | **6** | **6** | **18** |

18 là số **official classifier run identities**, không bao gồm generation/reference jobs hoặc resume/retry của cùng run. E2 Base là kết quả của hybrid baseline E1 + E2; không tạo thêm E3-A0 hoặc condition E2 độc lập. Strength search dùng đúng ba common training seeds cho mỗi strength trong grid, tổng cộng 9 candidate classifier runs; chọn strength bằng mean Validation Tail Macro F1 qua ba seeds, hòa ở giá trị chưa làm tròn thì chọn strength nhỏ hơn. Checkpoint từng candidate vẫn theo overall Validation Macro F1 của common checkpoint policy.

Cho phép **verified reuse** cả ba selected-strength candidate runs làm ba official E1E2 runs theo mục 9.3; không đếm chúng hai lần hoặc chọn reuse theo score của từng seed. Tuning namespaces/provenance vẫn riêng và được bảo toàn; hai strengths không được chọn không có official bindings. Khi reuse đủ ba selected runs, tổng số training jobs duy nhất cho đủ benchmark là `9 search + 15 official còn lại = 24`, gồm 18 official identities và 6 unselected tuning jobs bổ sung, không tính failed attempts/retries. Nếu không reuse, vẫn phải đủ 18 official identities và báo rõ số fresh jobs thực tế. Shared classifier runtime phải được khóa trước cả tuning lẫn official runs (mục 6).

```text
Cleaned real dataset
        │
        ▼
Tạo và khóa MỘT split mới
        │
        ├── Train real → training pool theo condition → classifier riêng
        ├── Validation real → chọn checkpoint / tuning được cho phép
        └── Test real → đánh giá cuối cùng sau khi khóa checkpoint

Mọi ảnh đưa vào classifier → center letterbox 896 × 384 → EfficientNet-B0
                            tensor [B, 3, 384, 896]
Ảnh đưa vào diffusion     → native size + reflection padding /8
```

## 3. Dataset và danh tính mẫu

### 3.1. Đầu vào chuẩn

Dataset đã cleaning, gồm **49.997 JPG / 124 thư mục lớp** tại `data/cleaned_dataset/<class_name>/`. Không chỉnh sửa hoặc ghi đè ảnh real trong quá trình benchmark.

Kích thước ảnh ghi theo **W × H**. Bộ cleaned đã được kiểm kê header trong workspace: 49.942 ảnh `1536 × 662`, 55 ảnh `1030 × 703`. Đây là thông tin đầu vào, không phải kết quả model.

Trước split, lập inventory từ chính cleaned dataset: relative path, class name/ID, width, height, file size, SHA-256 của file (`sha256`) và decoded RGB pixel hash (`decoded_pixel_sha256`). Nếu số ảnh/lớp khác baseline hoặc file không đọc được, dừng để xác định phiên bản dữ liệu; không âm thầm bỏ mẫu. Đây là bước kiểm kê đầu vào, không phải bước cleaning mới.

### 3.2. Định danh cố định

- `relative_path`: đường dẫn tính từ dataset root, dùng dấu `/`, giữ nguyên chữ hoa/thường và tên file; không dùng absolute path làm danh tính.
- `image_id = SHA256(UTF8(relative_path))`.
- `class_to_idx`: sắp tên lớp theo Unicode code point, không theo locale; gán ID liên tiếp `0..123` và lưu một lần.
- `n_original(c)`: số ảnh real của lớp trước split.
- `n_train(c)`: số ảnh real của lớp trong training split trước mọi augmentation.
- `group(c)`: head, medium hoặc extreme-tail theo `n_train`, khóa trước augmentation.

Dataset fingerprint được tính từ inventory canonical đã sắp theo relative path, bao gồm class mapping và content hashes, loại trừ timestamps/absolute paths. Di chuyển dataset root không làm đổi membership; thay nội dung file làm fingerprint không còn tương thích.

### 3.3. Exact duplicate audit trước split freeze

`sha256` băm nguyên file bytes. `decoded_pixel_sha256` băm biểu diễn canonical: prefix UTF-8 `RGB8` và một byte NUL, tiếp theo width và height dạng unsigned 64-bit big-endian, rồi RGB uint8 bytes theo row-major. Decode ảnh, áp EXIF orientation nếu có, chuyển RGB; không resize, normalize hoặc encode lại trước khi hash. Khóa decoder/version và quy tắc chuyển RGB trước inventory để pixel hash có thể tái lập.

Hai ảnh là exact duplicates nếu trùng **một trong hai** hash. Gom thành connected components theo quan hệ này, bao gồm liên kết bắc cầu. Lưu duplicate-group IDs, thành viên, labels và hai loại hash trong audit artifact. Nếu một component chứa nhiều labels, dừng để xử lý dataset identity; không tự relabel hoặc bỏ ảnh.

Mọi component cùng label phải nằm trọn trong một split, vẫn giữ đủ ảnh và quota ở mục 4.1. Nếu không tồn tại cách phân bổ thỏa cả quota và duplicate grouping, không được freeze split; báo component/quota gây xung đột để xử lý như một thay đổi dataset/policy có phiên bản. Không tự deduplicate hoặc nới quota để tiếp tục.

Trước khi publish split lock, audit lại cả ba cặp train/validation/test: không giao `image_id`, file SHA-256 **hoặc** decoded RGB pixel hash. Lưu báo cáo audit và hash của nó vào common data lock. Audit thất bại chặn mọi bước downstream. Kiểm tra này phát hiện exact duplicates; không khẳng định đã loại near-duplicates hoặc các ảnh cùng vị trí địa lý.

## 4. Tạo fixed split mới

### 4.1. Quota theo lớp

Tạo split mới từ inventory cleaned; không import hoặc phục hồi split cũ. `split_seed = 42`, độc lập với training seeds. Split chỉ được tạo một lần trước augmentation và dùng lại ở mọi condition, mọi seed, mọi lần resume.

Để cụ thể hóa yêu cầu “khoảng 80/10/10”, **quyết định của Common Spec** với `n >= 10` là dùng round-half-up cho số ảnh validation và test:

```text
q = max(1, floor(n / 10 + 0.5))
train_count = n - 2*q
val_count   = q
test_count  = q
```

| `n = n_original(c)` | Train | Validation | Test |
|---|---:|---:|---:|
| `n >= 10` | `n - 2*q` | `q` | `q` |
| `3 <= n <= 9` | `n - 2` | 1 | 1 |
| `n = 2` | 1 | 0 | 1 |
| `n = 1` | 1 | 0 | 0 |

Ví dụ: `1 → 1/0/0`, `2 → 1/0/1`, `3 → 1/1/1`, `9 → 7/1/1`, `10 → 8/1/1`, `15 → 11/2/2`, `100 → 80/10/10`. Không dùng banker's rounding hoặc làm tròn riêng ba tỷ lệ khiến tổng khác `n`.

### 4.2. Thứ tự phân bổ có thể tái lập

Quy ước mới `split_policy_version = geolocation_common_split_v2_exact_duplicates` (giữ domain string bên dưới để ổn định thứ tự ảnh cơ sở):

1. Với từng ảnh, tính khóa SHA-256 từ UTF-8 của chuỗi `geolocation-common-split-v1|42|<relative_path>`.
2. Trong mỗi lớp, sắp tăng dần theo chuỗi hex của khóa; nếu trùng khóa, dùng relative path để phá hòa.
3. Phân bổ cơ sở: lấy `train_count` ảnh đầu làm train, `val_count` ảnh tiếp làm validation, còn lại làm test. Nếu không chia cắt duplicate component nào, giữ nguyên phân bổ này.
4. Nếu có component bị chia cắt, phân bổ lại **trong lớp đó** theo nguyên component, với singleton component cho ảnh không trùng. Sắp components theo số thành viên giảm dần, rồi theo vị trí thành viên đầu tiên trong thứ tự bước 2. Chọn assignment vector nhỏ nhất theo thứ tự từ điển với `train < val < test` trong các assignments thỏa chính xác quota. Quy tắc chọn dựa trên nghiệm khả thi hoàn chỉnh; một bước greedy thất bại không đủ để kết luận vô nghiệm.
5. Chạy duplicate leakage audit mục 3.3. Chỉ khi audit thành công mới lưu/freeze manifests theo `(class_id, relative_path)`, cùng quota, thuật toán, duplicate assignments và fingerprint.

Đây là split mới theo policy của Markdown mới. Domain string và đầu vào danh tính đã được xác định cho benchmark này; không dùng hash key/raw path/lock của repository cũ. Không tạo lại split theo training seed.

### 4.3. Invariants và singleton

- Mỗi ảnh real thuộc đúng một split; train/validation/test không giao nhau về `image_id`, file SHA-256 hoặc decoded RGB pixel hash, hợp của chúng đúng bằng inventory.
- Mỗi lớp có ít nhất một real training image; tổng quota từng lớp bằng `n_original`.
- Validation và test chỉ chứa ảnh real.
- Augmentation, reference generation và real–synthetic pairing chỉ lấy nguồn từ `train.csv`; lưu source IDs để xác minh.
- Không dùng ảnh validation/test làm input sinh ảnh hoặc reference mining.
- Không chuyển augmented/synthetic samples vào validation/test.
- Nhóm lớp được xác định từ `n_train`, không đổi theo số ảnh augmented.

**Singleton (`n_original = 1`)** giữ một ảnh real trong train và giữ output class của classifier. Không sinh final training augmentation và không pair Mixup. Trong E3-A2 trở đi, singleton vẫn có 20 CADS references và centroid để làm candidate confuser; references chỉ phục vụ mining, không vào classifier training.

Lớp có `n_original = 2` hoặc `3` cũng có `n_train = 1` nhưng **không phải singleton**; vẫn đủ điều kiện diffusion augmentation. Không suy singleton từ `n_train`.

## 5. Hợp đồng kích thước ảnh và preprocessing

### 5.1. Classifier

**Input:** ảnh RGB, canvas **896 × 384 (W × H)**, tensor **`[B, 3, 384, 896]`**. Cùng preprocessing cho ảnh real, traditional augmented, final synthetic và real validation/test.

Giữ aspect ratio bằng resize rồi letterbox; không kéo giãn ảnh nguồn. Để khóa hình học thống nhất, Common Spec chọn các quy ước vận hành sau:

- `scale = min(896/W, 384/H)`; `W_resized = min(896, max(1, floor(W*scale + 0.5)))`, `H_resized = min(384, max(1, floor(H*scale + 0.5)))`.
- Resize bilinear với antialias, center letterbox lên canvas `896 × 384`. Padding: `left = floor((896-W_resized)/2)`, `right = 896-W_resized-left`, `top = floor((384-H_resized)/2)`, `bottom = 384-H_resized-top`; phần lẻ ở phải/dưới.
- Tensor RGB float trong `[0,1]`; fill letterbox bằng ImageNet mean `[0.485, 0.456, 0.406]` trước normalization.
- Normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Khóa backend/version xử lý ảnh vì interpolation có thể khác giữa thư viện. Không thêm online random augmentation vào preprocessing chung.

Đây là các quyết định cụ thể hóa yêu cầu preserve-aspect-ratio + letterbox của Markdown mới. Không có preprocessing model nào được tự động thay bằng default transform của pretrained weights nếu default đó khác hình học trên.

### 5.2. Diffusion native resolution

Giữ native cleaned resolution. SD1.5 nhận source image qua VAE; chỉ reflection-pad tối thiểu để W và H chia hết cho 8, không resize về classifier canvas `896 × 384` hoặc một canvas vuông.

```text
Cleaned RGB ở native size
        ↓ minimal reflection padding /8
Padded RGB → VAE encode → img2img noise/denoise → VAE decode
        ↓ crop đúng padding đã thêm
Synthetic RGB có kích thước bằng ảnh cleaned nguồn
```

| Source W × H | Padded W × H | Latent C × H × W | Final synthetic W × H |
|---|---|---|---|
| `1536 × 662` | `1536 × 664` | `4 × 83 × 192` | `1536 × 662` |
| `1030 × 703` | `1032 × 704` | `4 × 88 × 129` | `1030 × 703` |

Kích thước latent suy ra từ hệ số giảm không gian `/8` của SD1.5 VAE. Generation batch size là 1 source/lần; việc ghép guidance conditions trong U-Net không làm đổi số ảnh đầu ra yêu cầu.

Quy ước padding của Common Spec: center reflection-pad, left/top lấy floor của nửa phần thiếu, phần lẻ ở right/bottom. Lưu `original_size`, `padded_size`, offsets `(left, top, right, bottom)` và crop bbox half-open `[left, top, left+W, top+H]` trong provenance. Kiểm tra decode đúng canvas và crop đúng source size trước khi publish artifact. Backend normalization/encode/decode phải được ghi trong resolved generation config.

Chính sách này dùng chung cho final synthetic E2/E3 và reference generation. Reference images phải có vai trò riêng trong manifest để không bị đưa nhầm vào training pool.

### 5.3. Feature model khác

DINOv3 không dùng mặc định classifier canvas `896 × 384`. Checkpoint và processor của DINOv3 thuộc spec E3; kích thước preprocessing thực tế phải được resolve và khóa trước khi tạo reference embeddings. Common Spec không tự đặt một input size chưa có trong Markdown nguồn.

## 6. Cấu hình model và training chung

| Tham số | Giá trị cố định |
|---|---|
| Classifier | EfficientNet-B0 |
| Pretrained weights | `IMAGENET1K_V1`; khóa bytes/hash thực tế |
| Fine-tuning | Full backbone |
| Classifier head | `1280 → 124` |
| Dropout | `0.2` |
| Input | RGB `896 × 384`; tensor `[B, 3, 384, 896]`, center letterbox giữ aspect ratio |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Epochs | `20` |
| Scheduler | 1 epoch linear warmup, start factor `0.1`, sau đó cosine |
| Minimum LR | `1e-6` |
| Loss | CrossEntropyLoss |
| Label smoothing | `0` |
| Class weights / focal loss | Không |
| Weighted sampler | Không |
| Default Mixup/CutMix | Không; chỉ E3-A4 có chính sách Mixup riêng |
| Batch size | **128 cố định**; không batch-size fallback |
| Training seeds | **`[42, 123, 2026]`** |
| Split seed | `42` |
| Augmentation seed | `314159` |

Ba training seeds trên được chọn và khóa tại Common Spec, trùng bộ seeds của Markdown mới để giữ nhất quán. Áp dụng đủ ba seeds cho tất cả main conditions và E3 ablations; không thay seed sau khi xem kết quả.

### 6.1. Shared classifier runtime đã chốt

Các lựa chọn dưới đây cụ thể hóa những runtime parameters còn mở; áp dụng cho cả strength-tuning classifier runs và 18 official runs. Đây là quyết định của Common Spec mới, không phải tự động import defaults của repository cũ.

| Runtime parameter | Giá trị khóa |
|---|---|
| Device | Một CUDA GPU/run, không DDP/DataParallel; cùng GPU model và runtime profile |
| Precision | FP32 model/input/forward/backward/optimizer; AMP/autocast OFF, GradScaler không dùng; TF32 OFF |
| Tensor layout / compilation | Contiguous NCHW; channels-last OFF; `torch.compile` OFF |
| AdamW defaults | `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=false`, `maximize=false`, `foreach=false`, `fused=false`, `capturable=false`, `differentiable=false` |
| Parameter groups | Một group gồm toàn bộ trainable parameters, cùng LR/weight decay; không thêm bias/norm exemptions |
| Loss reduction | `mean`; loss/metrics theo epoch được tổng hợp theo số samples thực tế |
| Gradient accumulation / clipping | `gradient_accumulation_steps=1`; không gradient clipping; `zero_grad(set_to_none=true)` mỗi batch |
| Training sampling | Shuffle một permutation toàn bộ anchors/epoch, không replacement, không weighted sampler; pairing A4 tuân theo invariant mục 2.1 |
| DataLoader | `num_workers=4`, `pin_memory=true`, `persistent_workers=false`, `prefetch_factor=2`, giữ thứ tự batches do sampler tạo |
| Last batch | `drop_last=false` cho train/validation/test; batch cuối có thể nhỏ hơn 128 |
| Validation / test | Không shuffle; `model.eval()` + inference mode; không làm thay đổi training RNG |
| Training mode | `model.train()`; dropout và BatchNorm hoạt động theo training mode của full backbone |
| Image backend | Pillow RGB decode/EXIF transpose; chuyển float tensor trước resize, dùng torchvision tensor bilinear resize với `antialias=true`; versions được khóa cùng environment |

Các settings deterministic và worker RNG ở mục 8 tiếp tục áp dụng. OOM phải dừng run, không tự giảm batch size, workers, precision hoặc resolution để hoàn tất riêng một condition.

**Scheduler convention cố định:** với training pool có `N` anchors, `L = ceil(N/128)` optimizer steps/epoch. Tại optimizer step `k` (zero-based, `0..20*L-1`), đặt `u=k/L`; áp LR **trước** optimizer update:

```text
0 <= u < 1:
    lr = 3e-4 * (0.1 + 0.9*u)
1 <= u <= 20:
    lr = 1e-6 + 0.5*(3e-4 - 1e-6)*(1 + cos(pi*(u-1)/19))
```

Scheduler tiến một lần sau mỗi optimizer step; không step thêm ở validation/epoch boundary và không phụ thuộc validation score. `lr(0)=3e-5`, `lr(1 epoch)=3e-4`, `lr(20 epochs)=1e-6` tại terminal boundary sau update cuối; LR của update cuối lấy `u=20-1/L`. Checkpoint lưu progress đủ để resume đúng `k` và `L`.

Trước **bất kỳ strength-tuning run hoặc official run nào**, materialize và khóa resolved shared classifier config, preprocessing/backend, checkpoint/evaluation policy, environment/code identity, pretrained weights và ba initial states. Package versions, GPU/driver/CUDA identity và file hashes phải có giá trị thực tế, không để mutable revision hoặc trường chưa resolve. Không tìm runtime settings theo tuning results hoặc từng condition. Shared runtime lock không phụ thuộc selected strength để tránh vòng phụ thuộc; selected-strength lock được tạo sau tuning và là dependency của official augmented pools.

### 6.2. Generation parameters chung

Các tham số generation chung xuyên suốt E2 → E3-A4:

| Tham số | Giá trị |
|---|---|
| Backbone | Stable Diffusion v1.5 |
| Mode | Image-conditioned / img2img |
| Solver | DPM-Solver++ |
| Configured inference steps | `50` |
| CFG guidance `w` | `2.0` |
| Source / output resolution | Native cleaned size, minimal reflection-pad `/8`, crop padding |
| Generation batch size | `1` |
| Textual Inversion / LoRA | OFF / OFF |

50 là số steps **cấu hình**, số active denoising steps phụ thuộc strength. Strength được chọn ở E2 theo validation protocol của spec E1E2 rồi khóa để tái sử dụng trong E3. Không kế thừa sampler, guidance, strength, precision hoặc QC/acceptance policy từ repository cũ. Common Spec không định nghĩa thuật toán sinh ảnh riêng của mỗi nhánh.

## 7. Evaluation contract dùng chung

### 7.1. Ground-truth support và label space

Với mỗi split `S` là validation hoặc test:

```text
C_eval(S) = {c : support_S(c) > 0}
F1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
Macro F1(S) = mean(F1_c for c in C_eval(S))
```

Tập `C_eval(S)` lấy từ ground truth của fixed real split và khóa trước training. Không đưa lớp support=0 vào mẫu số với F1=0. Per-class metrics của lớp không có support ghi `N/A` (JSON dùng `null`, không dùng NaN).

Classifier luôn dự đoán đủ **124 lớp**, không mask logits của lớp thiếu evaluation support. Dự đoán nhầm sang singleton vẫn làm sai sample và giảm score của lớp ground truth. Macro F1 validation và test có thể dùng hai tập evaluated classes khác nhau; báo cáo support và danh sách lớp của từng split.

### 7.2. Group-wise và secondary metrics

Với nhóm `G` đã khóa theo `n_train`:

```text
C_eval_G(S) = C_eval(S) ∩ G
Group Macro F1     = mean(F1_c for c in C_eval_G(S))
Group Macro Recall = mean(Recall_c for c in C_eval_G(S))
```

Tính TP/FP/FN từ **toàn bộ evaluation split**, sau đó mới lấy trung bình các lớp thuộc nhóm. Không lọc sample ngoài nhóm hoặc hạn chế logits trước khi tính. Nhóm không có support ghi `N/A`. Tail Macro F1 dùng cùng policy ở validation tuning và final test.

Secondary metrics: Top-1 Accuracy, Top-5 Accuracy trên toàn bộ 124 logits, Weighted F1, Balanced Accuracy (mean recall trên `C_eval(S)`), per-class F1/Recall và confusion matrix đủ `124 × 124`. Báo số lớp có support và số sample cho từng split/nhóm.

### 7.3. Aggregation và test isolation

Với mỗi condition/variant, báo đủ ba seeds, **mean ± sample standard deviation (`ddof=1`)**. Không gộp predictions của ba models thành một score thay thế báo cáo qua seeds. Giữ số thực chưa làm tròn cho checkpoint selection và aggregation.

Validation phục vụ chọn checkpoint và tuning được spec nhánh cho phép. Chỉ mở final test cho checkpoint đã chọn và khóa; không chọn epoch, seed, strength hoặc sửa generator theo test score. Chi tiết prediction artifacts/reporting sẽ nằm trong `04_evaluation.md`; các quy tắc support và label space tại đây là bắt buộc.

## 8. Reproducibility

### 8.1. RNG và môi trường

Kế thừa cơ chế tái lập của repository cũ, với scope mở rộng cho mọi condition/ablation mới:

- Seed Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader generator và workers. Khởi động interpreter với `PYTHONHASHSEED` tương ứng trước khi chạy.
- Worker seed lấy từ DataLoader initial seed modulo `2**32`; không phụ thuộc thời gian hoặc thứ tự filesystem.
- Bật deterministic algorithms; cuDNN benchmark OFF, cuDNN deterministic ON, TF32 OFF. Đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước CUDA initialization.
- Nếu operation không hỗ trợ deterministic trong profile đã khóa, dừng và xử lý trước main runs; không âm thầm tắt chế độ deterministic.
- Validation/test dùng RNG riêng hoặc capture/restore để không làm thay đổi RNG training sampler.
- Lưu Python/package versions, OS, GPU model, driver, CUDA/cuDNN, code commit hoặc source archive hash, model revisions/weight hashes và resolved configs.

Dùng cùng single-GPU runtime profile cho các classifier runs. Tái lập được cam kết trong môi trường đã khóa; không cam kết bitwise equality giữa các GPU/library versions khác nhau.

Split seed, augmentation seed và training seed là ba vai trò riêng. Augmentation pool được sinh một lần theo generation configuration đã khóa rồi dùng lại cho ba training seeds. Seed từng generation job phải suy ra deterministic từ augmentation seed và job identity ổn định; spec nhánh xác định job key và lưu actual seed, source IDs, settings. Resume không tạo lại seed theo số lần retry.

### 8.2. Shared initialization

Tạo và khóa một classifier initial state cho mỗi training seed từ cùng ImageNet weights, bao gồm classifier head đã khởi tạo. Mọi condition/variant cùng seed load chính xác state đó trước training. Lưu initial-state hash trong run identity để kiểm tra sự ghép cặp.

Shared initialization không đòi các training pools có cùng số batch hoặc augmentation có cùng RNG consumption. Các khác biệt được cho phép phải nằm trong config/manifest của condition tương ứng, không làm thay shared defaults.

## 9. Checkpoint và Resume

### 9.1. Checkpoint policy

Kế thừa best/last và epoch-boundary recovery. Train đủ 20 epochs, validation cuối mỗi epoch bằng eval/inference mode, không early stopping.

- `best_checkpoint.pt`: epoch có Validation Macro F1 trên `C_eval(val)` cao nhất; hòa ở giá trị chưa làm tròn thì lấy epoch sớm nhất.
- `last_checkpoint.pt`: trạng thái ở cuối epoch đã hoàn tất gần nhất, là điểm resume chuẩn.
- Không dùng test metric để chọn checkpoint; lưu selected epoch và best-score history.

Checkpoint phải chứa: schema version, run identity, completed epoch, global optimizer step, model state, optimizer state, scheduler state/progress, best score/epoch và đủ dữ liệu khôi phục best checkpoint, training history, RNG states của Python/NumPy/PyTorch CPU/CUDA/DataLoader và RNG của logic training riêng nếu có. Nếu resolved precision profile có scaler thì lưu scaler state.

### 9.2. Quy tắc resume và lỗi giữa chừng

Run identity tối thiểu gồm benchmark version, condition/variant, training seed, shared/method config hashes, dataset/split/class/group hashes, training pool hash, initial-state hash, environment/code/model identity và augmentation dependency hashes.

Chỉ resume khi toàn bộ identity tương thích. Không thay split, batch size, epochs, preprocessing, dữ liệu hoặc runtime profile giữa run. Phục hồi optimizer/scheduler/RNG rồi bắt đầu epoch tiếp theo, không chỉ load model weights như một run mới.

Nếu bị ngắt giữa epoch, bỏ phần tiến độ chưa commit và chạy lại epoch đó từ last checkpoint hợp lệ. Nếu chưa có checkpoint, khởi động lại cùng seed/initial state. Không hỗ trợ mid-epoch resume trong common contract này.

Publish last checkpoint bằng ghi file tạm cùng filesystem, flush rồi atomic replace. Last checkpoint là nguồn phục hồi nhất quán, phải giữ đủ best state/history để tái tạo các file phụ nếu bị ngắt lúc publish. Ghi failure/resume history; retry không làm mất lịch sử hoặc chọn một seed khác. Checkpoint hỏng/không tương thích phải báo lỗi, không âm thầm bỏ qua và ghi đè run.

### 9.3. Verified reuse của selected-strength runs

Chỉ sau khi đủ 9 candidate runs, chọn và khóa một strength theo mục 2.2, mới được bind ba candidate runs của strength đó thành official E1E2 identities tương ứng seed. Reuse là công nhận một full training execution đã hoàn tất, **không phải** dùng tuning checkpoint để khởi tạo và train tiếp một official run.

Trước binding phải xác minh đầy đủ: cùng dataset/split/class/groups/support sets; cùng real/E1/selected-E2 members, labels, file/pixel hashes và canonical training manifest hash; cùng resolved classifier/generation configs, code/environment/model identity; cùng seed/initial-state hash; đủ common epochs, traversal và optimizer/scheduler steps; checkpoint/history/RNG artifacts hợp lệ và best epoch đúng common validation rule. Candidate chưa được evaluate trên test. Không dùng hash tương đương giả định chỉ vì counts hoặc strength giống nhau.

Giữ nguyên immutable candidate run identity, checkpoints và logs. Lưu official binding/promotion record riêng với official condition/seed, source run identity, source artifact hashes, selection lock, official pool lock và verification evidence. Khác biệt namespace/phase hoặc selection metadata được gắn sau training không được làm thay training semantics; không sửa checkpoint identity để trông như một fresh run.

Candidate runs chỉ phụ thuộc candidate pool và tuning protocol đã khóa trước search. Selection lock tham chiếu candidate evidence, sau đó official binding tham chiếu cả selection lock và source run; không ghi ngược selection hash vào candidate artifacts để tạo vòng dependency. Mỗi official identity có đúng một source execution. Xác minh/bind cả ba seeds như một bộ trước final test; nếu không thể reuse đầy đủ thì ghi lý do và chạy fresh cả ba official E1E2 seeds từ shared initial states, không chọn seed cần rerun theo score. Final test chỉ dùng các frozen best checkpoints được official binding hoặc fresh-run lock chỉ định.

## 10. Artifact Locking

### 10.1. Các nhóm artifact cần khóa

| Phạm vi lock | Nội dung |
|---|---|
| Common data | Dataset inventory/fingerprint với file/pixel hashes, duplicate components/assignments và leakage audit, mapping, split manifests, split policy/seed, counts, groups và evaluated class lists |
| Shared runtime | Common spec hash, resolved classifier config, preprocessing/checkpoint/evaluation policy, environment, code và model weight identity |
| Method / pool | Spec/config nhánh, generation provenance và actual seeds, output file/pixel hashes, training membership; shared E0/E1 pools, per-class tail budgets, selected-strength lock, cached A2 references/embeddings/centroids/confusers, A3/A4 pool identity và optimizer-step invariants |
| Run | Condition/variant, seed, các dependency hashes, initial state, best/last state và completion metadata; official binding/verification evidence nếu reuse selected-strength execution |
| Final evaluation | Frozen selected-checkpoint hash, split hash, predictions và metric outputs |

Split manifests mới gồm `train.csv`, `val.csv`, `test.csv`; mỗi row có tối thiểu `image_id, relative_path, class_name, class_id, split, sha256, decoded_pixel_sha256, duplicate_group_id, is_synthetic=false`. `class_groups.csv` lưu `n_original`, `n_train`, singleton flag và group. `split_metadata.json` lưu allocation policy, seed, per-class/per-split counts, duplicate audit identity và evaluation support sets.

Training pool manifest của mỗi condition phải phân biệt `real`, `traditional`, `final_synthetic`. Reference synthetic là artifact riêng, không phải training membership. Ghi source IDs và source hashes cho ảnh augmented/generated để xác minh nguồn là training original.

### 10.2. Hash, publication và dependency

- File contents dùng SHA-256; manifests dùng UTF-8 và thứ tự row/field cố định. JSON canonical có key order ổn định; không đưa timestamp vào semantic fingerprint.
- Ghi nhận cả config hash và actual output hashes; seed/config giống nhau không đủ để xem hai pool là cùng artifact.
- Publish output hoàn tất trước, kiểm tra đầy đủ rồi atomic-publish lock cuối cùng. Artifact dang dở không được xem là frozen chỉ vì đã có file trên disk.
- Trước generation, training, resume hoặc evaluation, xác minh required upstream artifacts tồn tại, đúng hash và đúng schema/identity.
- Shared split lock là bất biến. Method/pool locks độc lập để thêm E3 ablation mà không thay identity của E0 hoặc mở khóa các pool đã chốt.
- Shared runtime lock phải tồn tại trước strength tuning; các official pool/run locks phải xác minh fairness invariants ở mục 2.1. Lưu explicit dependency hashes cho shared pools/strength/reference cache, không chỉ copy giá trị config sang từng nhánh.
- Không dùng một stage sequence hard-code cho chỉ hai conditions/sáu runs như benchmark cũ; lock phải thể hiện dependencies thực tế của benchmark mới.

### 10.3. Mismatch và thay đổi phiên bản

Thiếu file, checksum mismatch, sai source membership hoặc identity khác phải **dừng trước khi tiêu thụ artifact**. Không tự sửa lock cho khớp file, không regenerate split, không overwrite pool frozen để tiếp tục run.

Nếu chủ động thay dữ liệu, policy hoặc config đã khóa, tạo benchmark/pool/run version mới và ghi dependency mới. Giữ artifact cũ để truy vết; không gộp kết quả khác shared policy vào cùng bảng so sánh mà không công bố. Checkpoint đang chạy là mutable progress artifact được publish an toàn; khi run hoàn tất, khóa selected best checkpoint trước final test.

## 11. Điều kiện nghiệm thu Common Specification khi triển khai

Các điều kiện sau là hợp đồng cho implementation sau này, chưa phải tuyên bố đã chạy kiểm tra:

1. Input là cleaned dataset đúng identity; không có bước cleaning lại hoặc phụ thuộc raw dataset/cleaning lock cũ.
2. Split mới thỏa quota ở các biên `n=1,2,3,9,10,15`, có membership cố định và được khóa trước augmentation.
3. Singleton, `n_train=1` nhưng không singleton, và group boundaries `19/20/99/100` được xử lý đúng.
4. Cả 124 output classes được giữ; support=0 và group-wise metrics tuân theo mục 7.
5. Classifier input đúng `[B,3,384,896]`, center letterbox canvas `896 × 384` giữ aspect ratio theo mục 5.1; diffusion giữ native content, pad `/8` rồi crop đúng ảnh nguồn; DINOv3 giữ processor riêng.
6. Tất cả experiments dùng seeds `[42,123,2026]`, shared initialization theo seed và batch size 128 không fallback.
7. Resume phục hồi model/optimizer/scheduler/RNG từ epoch đã commit và từ chối identity mismatch.
8. Artifact locks phát hiện file/config/membership thay đổi và chặn downstream runs trước khi sử dụng.
9. Shared classifier runtime mục 6.1 được resolve và khóa trước strength tuning lẫn main experiments; mọi khác biệt có chủ đích nằm trong spec/config của nhánh.
10. Duplicate components giữ nguyên trong một split; không có file/pixel hash giao giữa các splits, cross-label duplicate hoặc bài toán quota vô nghiệm phải chặn freeze.
11. Fairness locks xác minh shared E0/E1 pools, tail budgets, selected strength, A2 cache và cùng A3/A4 synthetic membership/optimizer steps.
12. Official matrix có đúng 6 conditions × 3 seeds = 18 classifier run identities; tuning có đủ 3 strengths × 3 seeds và selection theo mean Validation Tail Macro F1. Chỉ cả ba verified selected-strength executions được bind làm official E1E2 theo mục 9.3, không đếm hai lần; unselected runs vẫn là tuning-only.

Common Spec quy định các hợp đồng dùng chung; realization và confirmed decisions của E1E2 nằm trong `02_E1E2_hybrid_baseline.md`. Các spec nhánh và implementation artifacts phải tham chiếu đúng common version, không thay các invariants đã khóa. Chưa triển khai code hoặc chạy experiments.
