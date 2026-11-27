# 01 — E0 Original Baseline Specification

> **Workflow (2026-11-28):** Implementation completion means the production implementation exists and required local unit/integration tests pass, including fixture or mock tests where needed. Deployment verification means actual UIT environment, data, models, resources, recovery and shared-code runtime checks have passed with evidence. Scientific execution completion means the required official runs, evaluations and report locks exist and verify. Pending deployment evidence never blocks writing or locally testing independent implementation; it continues to gate the corresponding real artifact publication, jobs and scientific stages. Full scientific execution requires separate user approval. Confirmed research requirements and acceptance rules are unchanged.


**Phiên bản:** 1.1\
**Ngày:** 2026-11-26\
**Condition ID:** `E0`\
**Phạm vi:** Original-only classifier baseline.\
**Trạng thái:** Đặc tả để triển khai; chưa chạy training hoặc tạo experiment artifacts.

## 1. Mục tiêu và quan hệ với Common Spec

E0 đo hiệu quả phân loại quốc gia khi giữ nguyên phân bố long-tail của real training split. Đây là original baseline cho RQ1 khi so sánh với E1 + E2 và là mốc tham chiếu trong main benchmark cùng final E3.

Nguồn yêu cầu:

- [Markdown tổng thể](../GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), Section I và Section III — E0 Original Baseline.
- [Common Specification v1.3](00_common.md), áp dụng đầy đủ và là nguồn cấu hình dùng chung.

Tài liệu này chỉ bổ sung hợp đồng riêng cho E0. Không có E0 override cho classifier, pretrained weights, preprocessing, optimizer, scheduler, precision, batch size, training seeds, số epochs, checkpoint selection hoặc metrics đã khóa ở Common Spec.

| Nội dung kế thừa | Nguồn có thẩm quyền trong Common Spec v1.3 |
|---|---|
| Official matrix, paired comparison và shared real pool | Mục 2.1–2.2 |
| Cleaned input, file/pixel identity, duplicate audit và fixed split | Mục 3–4 |
| Classifier preprocessing | Mục 5.1 |
| Classifier config và resolved runtime | Mục 6 và 6.1 |
| Evaluation support, label space và aggregation | Mục 7 |
| RNG, environment và shared initialization | Mục 8 |
| Best/last checkpoint và resume | Mục 9 |
| Artifact identity, locks và mismatch handling | Mục 10 |

E0 chiếm một condition trong official matrix; chạy một classifier run cho mỗi training seed đã khóa tại Common Spec, tổng cộng **ba official E0 runs**. Resume/retry không tạo thêm official run identity. E0 không có hyperparameter search hoặc strength-tuning run riêng.

## 2. Training membership

### 2.1. Nguồn membership duy nhất

Gọi `R_train` là toàn bộ rows của real `train.csv` trong common split lock đã qua duplicate leakage audit. Membership E0 được định nghĩa chính xác:

```text
E0_training_pool = R_train
E0_sample_type   = real, cho mọi training row
E0_augmented_count = 0
```

E0 đọc membership từ locked manifest, không quét thư mục ảnh để tự suy ra dataset hoặc tạo lại split. Mọi row phải có `is_synthetic=false`, trỏ tới cleaned real input có file/pixel hashes đúng với common inventory. Nếu pool metadata dùng trường `sample_type`, mọi row đều có giá trị `real`.

Pool E0 tham chiếu trực tiếp manifest `train.csv` chung; không cần sao chép ảnh hoặc tạo một bản CSV khác làm nguồn membership độc lập. E0 pool descriptor lưu manifest path/hash và identity của common data lock để xác minh quan hệ này.

### 2.2. Bảo toàn phân bố và danh tính

- Với mọi lớp `c`, `E0_count(c) = n_train(c)` trong locked common groups/counts.
- Không thêm, bỏ hoặc thay thế training rows ở bất kỳ nhóm head, medium, extreme-tail hoặc singleton nào.
- Giữ nguyên class IDs, labels, source paths và file/pixel hashes; không sửa ảnh real trên disk.
- Giữ singleton và các lớp có `n_train=1` theo Common Spec, không gộp hoặc loại khỏi classifier label space.
- Các duplicate records được common split cho phép trong cùng train vẫn giữ nguyên membership; E0 không deduplicate lại hoặc đổi quota.
- Validation/test dùng nguyên real manifests đã khóa, không đưa sample của hai tập này vào training pool.

Counts phải được lấy từ manifests thực tế sau split, không hard-code tổng training size dự kiến từ repository cũ.

### 2.3. Shared E0 real pool cho các condition khác

E0 pool là real-anchor pool dùng chung trong fairness invariant của Common Spec. Real subset của mọi condition augmented phải bằng E0 theo IDs, labels và content hashes, không chỉ bằng số ảnh.

Khóa E0 real membership độc lập với các augmentation artifacts. E1 + E2 và E3 tham chiếu khóa này khi tạo pool của chúng; việc thêm condition hoặc sinh ảnh sau đó không được thay E0 pool hoặc làm mất hiệu lực các E0 runs đã hoàn tất trên cùng common identity.

## 3. E0-specific restrictions

E0 không can thiệp vào phân bố training ngoài phép shuffle đã khóa trong common runtime:

| Phạm vi | Ràng buộc E0 |
|---|---|
| Offline augmentation | Không tạo hoặc đưa traditional augmented images vào membership |
| Generative augmentation | Không diffusion generation, không final synthetic hoặc reference synthetic trong training |
| Online augmentation | Không random image transforms; chỉ dùng deterministic classifier preprocessing của Common Spec |
| Rebalancing | Không oversampling, undersampling, replacement sampling hoặc class-balanced/weighted sampler |
| Loss | Không class weights, focal loss hoặc loss correction theo tần suất lớp; dùng nguyên common loss |
| Sample mixing | Không Mixup, CutMix, real–synthetic pairing hoặc thay training anchors bằng mixed samples |
| Model/config | Không chọn cấu hình riêng cho E0, không thay pretrained initialization hoặc freeze backbone khác common config |
| Selection | Không chọn seed, epoch hoặc retry dựa trên test score; checkpoint selection theo Common Spec |

Resize, letterbox và normalization bắt buộc của classifier là preprocessing dùng chung, không được tắt để diễn giải “Original” thành ảnh raw không qua model preprocessing. “Original” ở E0 là cleaned real image trước augmentation, không phải raw dataset trước cleaning.

E0 không load generator, DINOv3, selected strength, reference bank hoặc confuser mapping như training dependencies. Các artifacts này có thể tồn tại trong cùng benchmark root nhưng không được làm thay đổi E0 execution hoặc membership.

## 4. Điều kiện bắt đầu và dependency contract

Chỉ được bắt đầu E0 training khi các dependencies sau đã được publish và xác minh:

1. Common data lock hợp lệ: inventory/fingerprint, mapping, split manifests, counts/groups, duplicate components/assignments, leakage audit và evaluation support sets.
2. E0 real pool descriptor chứng minh membership bằng đúng common `train.csv`, mọi content hash còn khớp và không có non-real sample.
3. Shared classifier runtime lock đã resolve đầy đủ theo Common Spec, gồm environment/code, preprocessing, checkpoint/evaluation policy và pretrained weights.
4. Ba shared initial states đã khóa; run dùng đúng initial state tương ứng seed trong official matrix.
5. E0 spec hash, resolved condition descriptor và run identity đã được xác định; existing run directory nếu có phải tương thích trước khi ghi dữ liệu mới.

E0 có thể chạy khi các dependencies trên sẵn sàng, không cần chờ E1/E2/E3 pools hoặc kết quả strength tuning. Augmentation dependencies trong E0 identity được ghi rõ là `not_applicable` hoặc danh sách rỗng theo schema; không dùng hash của pool augmented để ràng buộc E0.

Xử lý failure tuân thủ Common Spec. Classifier preprocessing là center letterbox giữ aspect ratio về `896 × 384` theo Common Spec mục 5.1. Không tạo runtime fallback riêng cho E0 hoặc điều chỉnh common settings theo kết quả E0 validation.

## 5. Training execution

```text
Common data/runtime locks + E0 real pool + shared initial state
                              │
                              ▼
                    Xác minh run identity
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
          Fresh E0 run               Compatible resume
          Load initial state         Restore last committed epoch
                 └────────────┬─────────────┘
                              ▼
          Shuffle toàn bộ real anchors cho epoch tiếp theo
                              │
                              ▼
          Common preprocessing → classifier training updates
                              │
                              ▼
          Real validation → common checkpoint selection
                              │
                              ▼
          Commit last state, best state và epoch history
                              │
                 Lặp đến hết common training schedule
                              │
                              ▼
          Freeze selected best checkpoint → final evaluation
```

### 5.1. Fresh run và epoch traversal

Mỗi seed là một run độc lập. Fresh run load đúng shared initial state, khởi tạo optimizer/scheduler theo shared config, seed RNG và DataLoader theo Common Spec. Không bắt đầu từ checkpoint đã fine-tune của seed hoặc condition khác.

Mỗi completed epoch phải duyệt **mỗi row của E0 training manifest đúng một lần** bằng permutation không replacement; batch cuối được xử lý theo common last-batch policy. Việc các batch khác nhau giữa seeds không làm thay training membership.

Log số training samples đã tiêu thụ và optimizer steps của mỗi epoch. Tổng committed steps phải đúng schedule suy ra từ E0 pool size và common runtime; không bù thêm steps để bằng augmented conditions. LR progression dùng E0 steps/epoch theo công thức Common Spec, không dùng pool size của E1 + E2 hoặc E3.

### 5.2. Validation và checkpoint

Validation cuối mỗi epoch trên fixed real validation split bằng preprocessing và evaluation contract chung. RNG validation không được làm thay training sequence. E0 dùng Validation Macro F1 theo common support policy để chọn best checkpoint và tie-break; không có tiêu chí chọn checkpoint riêng theo head/tail hoặc train accuracy.

Publish best/last state và history theo common atomic checkpoint protocol. Lưu validation-at-best tách khỏi final-test metrics. Hoàn tất đủ common epochs mới đánh dấu training completed; không dừng sớm vì score thấp hoặc không tăng.

### 5.3. Resume và failure

Thực hiện nguyên epoch-boundary resume của Common Spec: kiểm tra identity, phục hồi đầy đủ model/optimizer/scheduler/RNG/history và chạy epoch tiếp theo. Nếu ngắt giữa epoch, replay epoch chưa commit từ last state hợp lệ; không tính những updates bị bỏ vào completed-step totals.

Ghi lại failed attempts và resume events dưới cùng E0/seed identity. Không reset best-score history khi retry, không đổi seed, data hoặc config để hoàn tất run. Run đã hoàn tất và selected checkpoint đã khóa không được train thêm hoặc overwrite để cải thiện kết quả.

### 5.4. Kết thúc training và bàn giao evaluation

Trước khi freeze selected checkpoint, đối chiếu run identity, đủ completed epochs, committed sample/step counts, validation history và best-epoch selection. Khóa checkpoint hash cùng dependencies rồi bàn giao cho final evaluation theo Common Spec mục 7 và spec evaluation khi được viết.

Chỉ evaluate fixed real test với checkpoint đã chọn. Không dùng last checkpoint hoặc một seed tốt nhất để thay official results. E0 final report phải bao gồm cả ba official seeds và aggregation chung; chưa đủ ba seeds thì báo trạng thái incomplete, không công bố như baseline hoàn chỉnh.

## 6. Artifacts và provenance

Các đường dẫn dưới đây tính từ root artifact của một benchmark version; đây là cấu trúc đầu ra cần có khi triển khai, chưa phải các file được tạo ở giai đoạn viết spec.

### 6.1. E0 pool artifacts

| Artifact | Nội dung tối thiểu |
|---|---|
| Common real `train.csv` | Manifest chuẩn được tham chiếu trực tiếp, schema và hashes theo Common Spec |
| `pools/E0/pool.json` | Condition ID, common data identity, manifest reference/hash, total/per-class real counts, non-real counts bằng 0 và membership identity |
| `pools/E0/pool_lock.json` | Spec/descriptor hashes, common dependency hashes và kết quả xác minh pool bằng common training originals |

E0 không xuất augmented manifest hoặc thư mục ảnh mới. Không tạo bản sao real dataset để làm membership độc lập với split đã khóa.

### 6.2. Per-seed run artifacts

Run root: `runs/E0/seed_<seed>/`, với `<seed>` lấy từ official matrix của Common Spec. Mỗi run có tối thiểu:

| Artifact | Nội dung / mục đích |
|---|---|
| `config.yaml` | Snapshot resolved config thực tế được kế thừa từ shared lock, E0 membership reference và seed; không chứa override trái Common Spec |
| `run_metadata.json` | Run identity, condition/seed, pool/data/config/environment/model/initial-state/spec hashes, counts, status, UTC timestamps, failure/resume history |
| `history.csv` | Một row/completed epoch: sample-weighted train/validation loss, validation metrics, LR, consumed samples và committed optimizer steps |
| `lr_steps.csv` | Global step, epoch/step position và LR thực tế đã dùng, đủ đối chiếu shared scheduler convention |
| `last_checkpoint.pt` | Trạng thái epoch-boundary để phục hồi theo common checkpoint schema |
| `best_checkpoint.pt` | Selected validation checkpoint, cùng best epoch/score và identity |
| `metrics.json` | Validation-at-best; final-test ghi riêng khi đã evaluate, chưa evaluate thì để `null` |
| `checkpoint_lock.json` | Selected checkpoint hash, selected epoch/score và các dependency hashes sau khi training hoàn tất |

Run metadata cần ghi rõ actual real count, traditional count, final synthetic count và reference count trong training. Ba count non-real phải bằng 0; không suy ra chỉ từ tên condition.

Final evaluation xuất predictions, per-class/group metrics, confusion matrix và aggregation theo evaluation contract chung. Metadata của chúng phải liên kết selected-checkpoint hash với real evaluation split hash; E0 không định nghĩa lại metric formulas hoặc thay output schema của spec evaluation sau này.

### 6.3. Lock và trạng thái

- E0 pool lock độc lập với mutable training progress; common data/runtime lock vẫn là upstream authority.
- Mỗi run có trạng thái rõ: chưa chạy, đang chạy, failed, training completed hoặc final evaluation completed; lưu selected checkpoint freeze status riêng.
- `training completed` yêu cầu đủ schedule và checkpoint/history hợp lệ; `final evaluation completed` còn yêu cầu final-test artifacts đúng identity.
- Thiếu file/hash mismatch/membership mismatch phải chặn training, resume hoặc evaluation tương ứng. Không sửa lock tự động để chấp nhận output khác.
- Cross-condition comparison phải truy ngược được cùng E0 real pool hash và cùng initial-state hash theo seed. Không dùng E0 metrics từ benchmark version hoặc split khác.

## 7. Acceptance criteria

Đây là các điều kiện cần chứng minh khi triển khai; chưa phải kết quả kiểm tra hiện tại.

| ID | Tiêu chí | Bằng chứng cần có |
|---|---|---|
| E0-AC01 | Membership bằng đúng toàn bộ locked real train, từng lớp giữ `n_train` | Manifest reference/hash, pool descriptor, so sánh IDs/labels/file và pixel hashes |
| E0-AC02 | Không có augmented, final synthetic, reference synthetic hoặc holdout sample trong training | Non-real counts bằng 0, membership validation và common split leakage audit đã khóa |
| E0-AC03 | Không đổi group, không loại singleton hoặc deduplicate lại trong E0 | Per-class counts/group references khớp Common Spec, đủ label mapping |
| E0-AC04 | Không rebalance, weighted loss/sampler, online random transforms hoặc sample mixing | Resolved config và execution metadata phù hợp restrictions mục 3 |
| E0-AC05 | Mọi shared config/preprocessing/runtime invariant được kế thừa nguyên vẹn, classifier center letterbox `896 × 384`, tensor `[B,3,384,896]` | Common lock hashes, resolved snapshot, không có E0 override |
| E0-AC06 | Ba official E0 runs dùng đúng seeds và paired initial states | Run identities khớp official matrix, initial-state hashes theo seed |
| E0-AC07 | Mỗi completed epoch thăm từng real anchor đúng một lần và đủ steps theo E0 pool size | Traversal audit theo IDs/permutation digest gắn manifest, consumed-sample counts, history và LR/step logs |
| E0-AC08 | Checkpoint được chọn chỉ theo common validation policy, đủ common epochs | Validation history, best/last checkpoint metadata và selected-checkpoint lock |
| E0-AC09 | Resume đúng epoch boundary, không mất committed history hoặc đổi identity | Resume/failure log, checkpoint state và restored scheduler/RNG progress; mismatch bị từ chối |
| E0-AC10 | E0 độc lập với augmentation jobs và không consume artifacts của chúng | Dependency graph/identity chỉ yêu cầu common artifacts, E0 pool và shared initialization |
| E0-AC11 | Test chỉ dùng frozen best checkpoint; label/support/group policy đúng Common Spec | Evaluation dependency hashes và outputs theo common evaluation contract |
| E0-AC12 | Baseline hoàn chỉnh có đủ ba seeds, aggregate theo Common Spec | Per-seed outputs và aggregate identity; không chọn seed tốt nhất hoặc loại run score thấp |

Nghiệm thu E0 dựa trên tính đúng của membership, execution và provenance, không đặt ngưỡng Accuracy/Macro F1 tối thiểu. Kết quả thấp vẫn là baseline hợp lệ nếu thỏa các hợp đồng trên. Spec này không triển khai training code, không tạo split mới lần nữa và không mở rộng thuật toán của E1/E2/E3.
