# 04 — Evaluation and Reporting Specification

> **Workflow (2026-11-28):** Implementation completion means the production implementation exists and required local unit/integration tests pass, including fixture or mock tests where needed. Deployment verification means actual UIT environment, data, models, resources, recovery and shared-code runtime checks have passed with evidence. Scientific execution completion means the required official runs, evaluations and report locks exist and verify. Pending deployment evidence never blocks writing or locally testing independent implementation; it continues to gate the corresponding real artifact publication, jobs and scientific stages. Full scientific execution requires separate user approval. Confirmed research requirements and acceptance rules are unchanged.


**Phiên bản:** 1.1\
**Ngày:** 2026-11-26\
**Phạm vi:** Validation, final test, support-aware metrics và reporting cho sáu official conditions.\
**Trạng thái:** EVAL-D01/D02 đã CONFIRMED. Chưa thực hiện inference, tính metrics hoặc tạo kết quả benchmark; verification khi triển khai được ghi tại mục 10.2.

## 1. Nguồn yêu cầu và phạm vi kế thừa

Nguồn có thẩm quyền:

- [Markdown tổng thể](../GeoLocation_Benchmark__E0_E1E2_E3_Implementation_Ready.md), Section I, II.3 và VI: research questions, metric set, support policy và ablation.
- [Common Spec v1.3](00_common.md): split/class/group identities, classifier preprocessing/runtime, evaluation contract, checkpoint selection, three-seed aggregation, official matrix và verified reuse.
- [E0 Spec v1.1](01_E0_original.md): baseline membership, per-seed execution và artifact contract.
- [E1E2 Spec v1.2](02_E1E2_hybrid_baseline.md): mean Validation Tail Macro F1 strength selection, official bindings và Base result reuse.
- [E3 Spec v1.1](03_E3_geodisc_ds.md): A1–A4, shared artifacts, A3/A4 invariants và confirmed G01–G10.

Tài liệu này cụ thể hóa cách tính và lưu **đúng các metrics đã chốt**, không thêm metric, tuning parameter hoặc thay evaluation policy. Không định nghĩa lại training/generation algorithms. Validation và test là hai real splits đã khóa; dùng cùng policy nhưng có thể có evaluated class sets khác nhau.

## 2. Evaluation protocol

### 2.1. Phân biệt ba vai trò evaluation

| Vai trò | Dữ liệu | Checkpoint/selection | Đầu ra được phép dùng cho |
|---|---|---|---|
| Epoch validation | Fixed real validation | Cuối mỗi training epoch; chọn best bằng overall Validation Macro F1 theo Common Spec | Monitoring và checkpoint selection |
| E2 strength selection | Fixed real validation | Tại common best checkpoint của từng candidate/seed; mean Tail Macro F1 qua ba seeds | Chọn duy nhất strength từ grid đã khóa |
| Final benchmark test | Fixed real test | Chỉ frozen best checkpoint thuộc official identity/binding hợp lệ | Main/ablation reporting; không lựa chọn lại model hoặc method |

Không dùng training pool, augmented images hoặc references làm evaluation membership. E0/E1E2/E3 dùng cùng manifests, labels, groups và evaluation support sets. Mỗi real evaluation record được tính đúng một lần; không deduplicate nội dung trong cùng split hoặc loại ảnh khó để thay số mẫu đã khóa.

### 2.2. Inference dùng chung

- Kế thừa Common Spec: RGB, preserve aspect ratio + center letterbox `896 × 384`, ImageNet normalization; tensor `[B,3,384,896]`.
- Cùng classifier/runtime FP32, AMP OFF, batch size 128 và last-batch policy đã khóa. Không đổi cấu hình để evaluate riêng một condition.
- `model.eval()` và inference mode; không optimizer update, không cập nhật BatchNorm statistics và không ảnh hưởng training RNG.
- Validation/test không shuffle, không random augmentation, không Mixup/CutMix hoặc real–synthetic pairing; A4 được evaluate trên real images bình thường.
- Mỗi sample trả logits trên **đủ 124 lớp** theo locked class mapping; không mask logits của lớp không có support.
- Xác minh sample coverage, output shape và finite logits trước publication. Thiếu/duplicate prediction row, sai label/order hoặc non-finite output là lỗi evaluation, không được bỏ sample rồi tiếp tục tính score.

**EVAL-D01 CONFIRMED:** Top-1 và Top-5 dùng cùng ranking của 124 logits: **logit giảm dần, class ID tăng dần khi logits bằng nhau chính xác**. Tương đương sort key `(-logit, class_id)`; không rounding/tolerance trước ranking, không ground-truth tie-break hoặc library-dependent ordering. Áp cho mọi epoch validation, tuning và official inference, kể cả tie tại ranh giới Top-5.

### 2.3. Evaluation dependencies

Trước inference, xác minh common data/split/class/group/support hashes, preprocessing/runtime identity và checkpoint/run identity. Final test còn yêu cầu run đã hoàn tất training, best-checkpoint lock và official binding nếu reuse E1E2.

Các pool/generation/cache dependencies được truy ngược qua run locks theo spec nhánh; evaluation không cần chạy lại generator, DINOv3 hoặc confuser mining. Verification requirements còn thiếu trong spec E3 không được ghi thành đã pass chỉ từ một score classifier.

```text
Fixed real evaluation manifest + class/groups/support lock
                              │
Frozen official best checkpoint / verified binding
                              │
                              ▼
              Common preprocessing và inference
                              │
                              ▼
             Per-sample logits + labels + Top-1/Top-5
                              │
                              ▼
          Full 124 × 124 confusion matrix trên toàn split
                              │
                              ▼
          Overall / supported groups / per-class metrics
                              │
                              ▼
          Per-seed results → mean ± sample std → reports
```

## 3. Support-aware metric definitions

### 3.1. Confusion counts và support

Với evaluation split S có N records, label space `C={0,...,123}`, confusion matrix M có **row=true class, column=Top-1 predicted class**:

```text
M[a,b] = số samples có true_class=a và predicted_class=b
support_S(c) = sum_b M[c,b]
TP_c = M[c,c]
FN_c = sum_b M[c,b] - TP_c
FP_c = sum_a M[a,c] - TP_c
C_eval(S) = {c ∈ C : support_S(c) > 0}
```

Support lấy từ ground truth manifest, không phụ thuộc model predictions, condition hoặc seed. Khi triển khai, row sums của M phải bằng locked supports, tổng matrix bằng N. Không tạo support bằng synthetic samples hoặc bằng việc model dự đoán một class.

### 3.2. Per-class F1 và Recall

Với class có support >0:

```text
Recall_c(S) = TP_c / (TP_c + FN_c)
F1_c(S) = 2*TP_c / (2*TP_c + FP_c + FN_c)
```

Mẫu số của hai biểu thức đều dương khi support >0. Class có support nhưng không dự đoán đúng nhận F1/Recall bằng **0**, không phải N/A.

Class có support=0 nhận `evaluable=false`, F1/Recall **N/A**; JSON dùng `null`, CSV dùng ô trống kèm status rõ, không NaN. Vẫn lưu raw TP/FP/FN và support, không xóa class khỏi mapping hoặc matrix. Một class không support có thể có FP ở cột của nó; N/A không có nghĩa prediction nhầm bị bỏ qua.

Singleton không xuất hiện trong validation/test theo fixed split nên không được evaluate trực tiếp. Prediction sang singleton vẫn làm sample sai, tạo FN cho true class và ảnh hưởng các metrics tương ứng. Lớp có hai originals có thể có test support nhưng không validation support; không suy support test từ support validation.

### 3.3. Overall metrics đã khóa

Primary metric:

```text
Macro F1(S) = (1/|C_eval(S)|) * Σ_{c∈C_eval(S)} F1_c(S)
```

Secondary metrics:

```text
Top-1 Accuracy(S) = Σ_c M[c,c] / N

Top-5 Accuracy(S) = (1/N) * Σ_i 1{true_class_i ∈ Top5(logits_i)}

Weighted F1(S) = (1/N) * Σ_{c∈C_eval(S)} support_S(c)*F1_c(S)

Balanced Accuracy(S) = (1/|C_eval(S)|) * Σ_{c∈C_eval(S)} Recall_c(S)
```

Balanced Accuracy là unadjusted mean Recall trên supported classes đúng Common Spec; không thay bằng một corrected/chance-adjusted variant. Macro F1 không dùng toàn bộ 124 classes làm mẫu số nếu một số lớp thiếu support. Không dùng weighted/class-frequency averaging để thay primary metric.

Mọi accuracy/F1/Recall lưu numeric values trong `[0,1]`, chưa làm tròn. N và `|C_eval(S)|` phải dương cho một overall evaluation hợp lệ; nếu manifest/evidence không đáp ứng thì báo invalid/incomplete, không tạo một score 0 giả.

### 3.4. Group-wise metrics

Groups G khóa từ số real training originals **trước augmentation**:

| Group | Điều kiện |
|---|---|
| Head | `n_train >= 100` |
| Medium | `20 <= n_train <= 99` |
| Extreme-tail | `n_train < 20` |

Với mỗi nhóm:

```text
C_eval_G(S) = C_eval(S) ∩ G
Group Macro F1(S) = mean(F1_c(S) for c in C_eval_G(S))
Group Macro Recall(S) = mean(Recall_c(S) for c in C_eval_G(S))
group_support(S) = Σ_{c∈C_eval_G(S)} support_S(c)
```

Per-class counts bắt buộc lấy từ **full confusion matrix trên toàn split**, sau đó mới chọn classes để average. Không lọc samples ngoài nhóm, bỏ predictions ngoài nhóm hoặc hạn chế logits trước khi tính; FP từ true classes ngoài nhóm vẫn được tính.

Nếu `C_eval_G(S)` rỗng, group F1/Recall là N/A/null, ghi `num_evaluable_classes=0` và support=0. Không báo group score=0 hoặc average nhóm này vào overall Macro F1. Nếu các nhóm đều có support, overall Macro F1 là class-weighted combination theo số evaluated classes, không phải trung bình đều ba group scores.

Validation Tail Macro F1 cho strength tuning chính là group Macro F1 extreme-tail với **validation support set**, không là metric mới hoặc một policy khác.

## 4. Checkpoint selection và test isolation

### 4.1. Checkpoint policy kế thừa

Mọi classifier run, kể cả strength candidates, dùng full common training schedule và chọn epoch có **overall Validation Macro F1 cao nhất** trên `C_eval(val)`. Hòa ở giá trị chưa làm tròn thì lấy epoch sớm nhất. Không early stopping, không chọn last checkpoint mặc định hoặc epoch có Tail Macro F1/test accuracy cao nhất.

History ghi overall validation scores theo epoch, selected epoch/score và checkpoint hash. Validation metrics tại selected epoch được ghi dưới `validation_at_best`; không ghi test metrics đè lên các trường này. Khi đối chiếu selection, dùng raw scores trước presentation rounding.

`last_checkpoint.pt` phục vụ resume theo Common Spec, không phải một model ứng viên thêm cho final comparison. Resume/retry không reset best history hoặc tạo quyền chọn một epoch/seed khác sau khi thấy test.

### 4.2. Strength selection và verified reuse

E2 search có ba strengths × ba common seeds. Với mỗi strength, lấy Tail Macro F1 tại **common best checkpoint** của từng seed rồi tính mean qua đủ ba seeds; chọn mean lớn nhất, hòa thì strength nhỏ hơn. Không chọn checkpoint bằng tail score, không average best-tail epochs hoặc dùng test cho strength selection.

Chỉ sau đủ nine-run evidence và selected-strength lock mới được verified bind cả ba selected executions làm official E1E2 theo Common Spec mục 9.3/E1E2 mục 6.4. Evaluator phải resolve official binding tới đúng immutable source checkpoint/history/config và xác minh hashes, không đổi candidate identity để giả lập fresh run.

Unselected strength runs không có final benchmark test evaluation hoặc official report rows. Không đủ reuse evidence thì đi đúng fresh-three-seed path đã chốt; không chọn rerun riêng seed score thấp. Reused execution được tính đúng một official identity/seed.

### 4.3. Final test gate

Final test chỉ được dùng sau khi checkpoint của run đã chọn và khóa, đủ training completion/dependency evidence. Với reused E1E2, hoàn tất verification/binding của cả ba seeds trước final test như Common Spec; không test candidates để quyết định có reuse hay không.

Không tự thêm global gate buộc mọi condition phải hoàn tất cả 18 runs trước một per-run test hợp lệ: các specs hiện tại cho phép run đủ điều kiện được bàn giao evaluation. Tuy nhiên một bảng comparison hoàn chỉnh phải có đủ ba seeds của tất cả conditions trong bảng; benchmark hoàn chỉnh có đủ 18 official identities.

Không dùng test predictions, confusion matrix hoặc per-class failures để đổi generator, prompts, QC, strength, cached mapping, Mixup policy, training/runtime hoặc selected checkpoint trong cùng benchmark. Kết quả thấp vẫn phải được ghi nhận.

Persist predictions/logits/metrics với identity cố định để tổng hợp và sửa presentation từ artifacts, không inference lại nhằm chọn score tốt hơn. Nếu inference bị ngắt/lỗi hạ tầng, chỉ retry cùng checkpoint/config/split identity, ghi failure history và không overwrite một kết quả khác mà không có incident/version provenance. Thay đổi evaluation implementation phải giữ bằng chứng cũ và kiểm tra conformance với policy đã khóa, không được biến thành tuning trên test.

## 5. Aggregation qua ba seeds

Với mỗi condition K, split S và một metric m đã định nghĩa ở mục 3, có đúng ba values `m_42`, `m_123`, `m_2026`:

```text
mean_m = (m_42 + m_123 + m_2026)/3
sample_std_m = sqrt(Σ_r (m_r - mean_m)^2 / (3-1))
report = mean_m ± sample_std_m
```

Áp dụng cho overall metrics, group F1/Recall và per-class F1/Recall khi có support. Dùng **sample standard deviation (`ddof=1`)**, không population std, standard error hoặc confidence interval thay cho std. Không làm tròn từng seed trước aggregation.

**EVAL-D02 CONFIRMED:** hiển thị metrics theo **percentage với hai chữ số thập phân**. Aggregate cell hiển thị `100*mean_m ± 100*sample_std_m`, cả hai vế format `.2f`, trong cột ghi đơn vị `%`; per-seed metric cũng nhân 100 và format `.2f` khi render. Raw per-seed/mean/std artifacts giữ nguyên full-precision values trong `[0,1]`; không dùng số đã render cho checkpoint selection, strength selection hoặc tính lại aggregation. Support/counts vẫn là integers, N/A/null không bị đổi thành `0.00`.

Các support/class sets giống nhau giữa seeds do fixed split. Class/group không có support nhận N/A ở cả ba seeds và aggregate N/A, không thay bằng zero. Seed thiếu artifact/failed run là `incomplete`, không phải support-based N/A; không tính mean trên hai seeds hoặc bỏ seed score thấp.

Không concatenate predictions/confusion counts qua seeds để tính một Macro F1 thay cho mean per-seed Macro F1; không ensemble logits/vote hoặc coi ba models là thêm evaluation samples. Mỗi per-run matrix thuộc đúng một model/seed, không cộng ba matrices rồi gọi là official condition performance.

Std qua ba training seeds chỉ phản ánh training variation trên fixed split/locked pools, không đại diện uncertainty qua các data splits, locations hoặc generation pools khác nhau. Không suy statistical significance chỉ từ mean ± std.

## 6. Results tables và phép so sánh

Tất cả bảng chính dùng **final real test của frozen best validation checkpoints**. Nếu báo validation, đặt bảng riêng có split label rõ; không trộn validation và test values trong cùng một metric column. Cell metric là `mean ± sample std` qua đủ ba seeds, hiển thị percentage với hai decimals và header `%` theo EVAL-D02; support/count fields là số nguyên từ manifest.

Các bảng sau là schema, **chưa có kết quả thực nghiệm**. `chưa có kết quả` khác với N/A do không có support và khác với score=0 hợp lệ.

### 6.1. Evaluation support table

| Split | Real sample count | Classifier output classes | Evaluated classes | Head classes/support | Medium classes/support | Extreme-tail classes/support |
|---|---|---|---|---|---|---|
| Validation | Từ locked manifest | 124 | Từ C_eval(val) | Từ locked groups/ground truth | Tương tự | Tương tự |
| Test | Từ locked manifest | 124 | Từ C_eval(test) | Từ locked groups/ground truth | Tương tự | Tương tự |

Lưu danh sách evaluated class IDs, per-class supports và singleton flags ở machine-readable artifacts; bảng tóm tắt không thay thế danh sách này. Không hard-code số support từ repository cũ hoặc coi mọi lớp có test support.

### 6.2. Main comparison — E0 vs E1E2 vs E3-A4

| Condition | Macro F1 — primary (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Weighted F1 (%) | Balanced Accuracy (%) |
|---|---|---|---|---|---|
| E0 — Original | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E1E2 — Hybrid baseline | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E3-A4 — Final Geo-DiSC-DS | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |

Group table có một row cho mỗi cặp condition/group trong ba conditions trên:

| Condition | Group | Evaluated classes | Real test support | Group Macro F1 (%) | Group Macro Recall (%) |
|---|---|---|---|---|---|
| E0 / E1E2 / E3-A4 | Head | Locked count | Locked support | mean ± sample std | mean ± sample std |
| E0 / E1E2 / E3-A4 | Medium | Locked count | Locked support | mean ± sample std | mean ± sample std |
| E0 / E1E2 / E3-A4 | Extreme-tail | Locked count | Locked support | mean ± sample std | mean ± sample std |

Ba dòng trên là template được materialize thành **9 rows**, không gộp ba conditions vào một value. RQ1 đối chiếu E0 với E1E2; RQ2 đối chiếu E1E2 với E3-A4, đặc biệt extreme-tail F1/Recall, đồng thời vẫn báo primary overall Macro F1 và các nhóm còn lại. Không đổi primary metric thành tail score sau khi xem kết quả.

### 6.3. Ablation comparison — Base vs A1 vs A2 vs A3 vs A4

| Variant | Thành phần thêm so với bước trước | Macro F1 — primary (%) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Weighted F1 (%) | Balanced Accuracy (%) |
|---|---|---|---|---|---|---|
| Base = official E1E2 | E2 Base + same E1 | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E3-A1 | CADS | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E3-A2 | DINOv3 mining + CCFG | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E3-A3 | Geo-preservation | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |
| E3-A4 | Same-class real–synthetic Mixup | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả | chưa có kết quả |

Kèm group table cùng schema mục 6.2 cho cả năm rows/variants, tạo **15 variant/group rows**. So sánh các bước kế tiếp đúng master: Base→A1, A1→A2, A2→A3, A3→A4. A2 là joint addition DINOv3 mining + CCFG, không tách thành hai unofficial conditions.

Base trỏ tới **cùng per-seed metrics/checkpoints/predictions của official E1E2**; A4 ở main và ablation cũng là cùng artifacts. Không rerun test, thêm seed hoặc tạo một row execution độc lập chỉ vì alias report khác nhau. Tổng matrix vẫn là sáu conditions × ba seeds, không phải tổng số rows xuất hiện trong các bảng.

### 6.4. Per-seed và per-class tables

Per-seed appendix ghi condition, seed, selected epoch, official/source-run identity, checkpoint hash, overall metrics và group metrics cho từng split. Không chỉ công bố seed đại diện hoặc seed tốt nhất.

Per-class table giữ đủ 124 classes cho mỗi condition/seed/split:

| Class ID/name | Group | n_original | n_train | Singleton | Real support | Evaluable | TP/FP/FN | F1 | Recall |
|---|---|---|---|---|---|---|---|---|---|
| Từ mapping | Locked group | Locked count | Locked count | Locked flag | Từ split | support >0 | Full-split counts | Numeric hoặc N/A | Numeric hoặc N/A |

Condition-level per-class summary có cùng identity/support fields và F1/Recall `mean ± sample std`; không average raw F1 của các lớp khác nhau để gọi là per-class score. Không thêm Precision, Micro F1 hoặc metrics khác ngoài tập nguồn vào result tables.

### 6.5. Confusion matrices

Xuất raw integer matrix đủ `124 × 124` theo fixed class-ID order cho mỗi official condition/seed, ở validation-at-best và final test. Row/column labels phải đi kèm hoặc tham chiếu mapping hash; support=0 vẫn giữ zero row và mọi prediction counts trong column tương ứng.

Heatmaps nếu render phải lấy từ đúng matrices đã lưu, ghi condition/seed/split/checkpoint; hiển thị đủ bộ seeds hoặc cung cấp đường dẫn tới toàn bộ matrices, không chọn một seed có hình đẹp nhất. Không tạo một pooled/averaged matrix thay thế per-seed counts. Normalized/trimmed views chưa được chọn làm output bắt buộc; raw matrix là evidence có thẩm quyền.

## 7. Prediction artifacts, schemas và locking

### 7.1. Per-run evaluation outputs

Giữ run root theo specs đã chốt: `runs/<condition>/seed_<seed>/`. Mỗi official run có các outputs sau; `val` luôn chỉ selected best checkpoint, không phải last epoch:

| Artifact | Nội dung |
|---|---|
| `predictions_val.csv`, `predictions_test.csv` | Mỗi real record một row, true/Top-1/Top-5 IDs và liên kết logits row |
| `logits_val.npy`, `logits_test.npy` | Raw FP32 logits `[N_split,124]`, class columns theo common mapping; không thay bằng logits đã mask/rounded |
| `confusion_matrix_val.csv`, `confusion_matrix_test.csv` | Raw integer 124×124 counts, row=true/column=predicted, mapping identity |
| `per_class_metrics_val.csv`, `per_class_metrics_test.csv` | 124 rows/split, schema mục 6.4, N/A handling rõ |
| `group_metrics.json` | Validation-at-best và final-test tách riêng; group F1/Recall, support và evaluated class IDs/counts |
| `metrics.json` | Overall validation-at-best/final-test metrics, checkpoint/epoch/identity references; final-test chưa chạy là null |
| `evaluation_metadata.json` | Runtime/code/policy hashes, split/mapping/group/support/checkpoint identity, sample coverage, execution status và failure history |
| `evaluation_lock.json` | Hashes của complete evaluation outputs và mọi upstream dependencies |

Prediction CSV dùng fixed field order: `image_id, relative_path, true_class_id, predicted_class_id, top5_class_ids, logits_row`. `top5_class_ids` là JSON array năm unique class IDs theo ranking trong một quoted CSV cell; `predicted_class_id` bằng phần tử đầu. IDs thuộc `0..123` và ground truth khớp manifest. Rows sắp theo `(class_id, relative_path)` của fixed split, `logits_row` trỏ đúng row NumPy; không dựa vào filesystem order.

Logits/CSV/metadata phải đủ để tính lại toàn bộ requested metrics và đối chiếu confirmed EVAL-D01 ranking. NumPy arrays không dùng object/pickle; serialization không làm tròn logits. Không thêm calibration, temperature scaling hoặc probability threshold vào evaluation để tạo predictions. Các machine-readable numeric CSV/JSON giữ raw values; formatted percentage tables là presentation artifacts riêng.

Validation predictions tại best checkpoint có thể được giữ khi best được commit hoặc export từ chính frozen best checkpoint trên real validation cùng identity. Nếu export lại, phải khớp validation evidence/history trong numerical contract đã khóa; không dùng một score mới để chọn lại best. Không bắt buộc persist logits/predictions của mọi epoch; history validation scores và selected-epoch provenance vẫn theo common checkpoint contract.

### 7.2. Reused E1E2 artifacts

Với verified reuse, official binding tham chiếu source training/checkpoint/validation evidence; không sửa immutable tuning namespace. Có thể tham chiếu đã verified validation artifacts bằng hash thay vì nhân bản file. Final-test outputs được publish trong **official namespace**, chỉ sau binding; không thêm test outputs vào unselected tuning runs.

Base alias và main E1E2 cùng resolve một evaluation lock/seed/split; A4 main/ablation cũng vậy. Export/copy khi trình bày không làm thành một official execution mới và phải giữ source hash.

### 7.3. Coverage, recomputation và atomic publication

Trước lock, xác minh prediction IDs đúng bằng manifest membership, không thiếu/thừa/lặp; row-to-logit mapping, true labels và full output shape hợp lệ. Kiểm tra `sum(M)=N`, row supports, Top-1 count và mọi per-class/group metric được suy ra từ cùng prediction set. Không chấp nhận metrics từ một checkpoint nhưng predictions từ checkpoint khác.

Để tổng hợp lại report, ưu tiên recompute từ frozen predictions/logits thay vì chạy inference hoặc training lại. Prediction processing không được bỏ lớp support=0 ra khỏi label space, đổi group hoặc sửa GT. Cùng policy/runtime phải tái tính cùng kết quả theo numerical convention đã khóa.

Viết outputs ở staging, chỉ atomic-publish evaluation lock khi coverage/metric checks hoàn tất; incomplete inference không được tạo final aggregate. Theo Common Spec, missing file/checksum mismatch phải chặn downstream reporting, không tự sửa lock hoặc điền score mặc định. JSON không chứa NaN/Infinity; unsupported metrics là null và lỗi execution có status/reason riêng.

## 8. Reporting artifacts và diễn giải

### 8.1. Benchmark-level outputs

| Artifact | Nội dung |
|---|---|
| `reports/evaluation_support.json` | Real counts, full label space, C_eval(val/test), groups/intersections, per-class supports và split/group hashes |
| `reports/per_seed_metrics.csv` | Sáu official conditions × ba seeds, mỗi split có keys rõ; overall/group values, support, selected epoch/checkpoint và source identity |
| `reports/aggregate_metrics.json` | Mean/sample std cho từng condition/split/metric/group và per-class summaries, đủ seed IDs/raw values và N/A status |
| `reports/main_comparison.csv` | Main overall/group tables mục 6.2, tham chiếu official aggregate rows |
| `reports/ablation_comparison.csv` | Base/A1/A2/A3/A4 tables mục 6.3; alias-to-condition mapping, không double-count runs |
| `reports/per_class_comparison.csv` | Per-class F1/Recall/support và summaries cho official conditions; 124 labels giữ nguyên |
| `reports/final_report.md` | Research questions, support table, main/ablation results, per-seed appendix, confusion-matrix links, provenance và limitations |
| `reports/report_lock.json` | Upstream evaluation locks, aggregation/presentation policy/config hashes và output checksums |

Per-seed table dùng long form khi có group rows: key gồm condition, seed, split, scope (`overall`/group), metric và checkpoint; không nhầm số table rows với số classifier runs. Mỗi overall/group metric có một raw value cho mỗi required seed, không lặp Base alias thành condition thứ bảy.

### 8.2. Nội dung final report

Report phải thể hiện:

1. RQ1/RQ2, đúng main/ablation matrix, primary/secondary metric roles.
2. Fixed real split identity, 124 output classes, evaluation support sets/counts và singleton limitation.
3. Shared real/E1 pools, per-class tail budgets, selected strength, cached A2 bank/confusers và A3/A4 pool/step equality qua artifact references.
4. Runtime/preprocessing/checkpoint policy identities, selected epoch từng seed và verified reuse/fresh status của E1E2.
5. Overall, group-wise, per-class results và confusion matrices; numeric values được dẫn về frozen prediction/checkpoint artifacts.
6. Run accounting: 18 official identities; nếu verified reuse đủ ba selected runs thì 24 unique full classifier executions gồm 9 search và 15 official còn lại, 6 unselected search executions là overhead. Nếu dùng fresh path thì báo actual accounting theo Common Spec; generation/reference jobs và retries không phải thêm seeds.
7. Generation attempts/accepted/rejected/shortage, training pool/step counts và các traces đã có trong specs nhánh như provenance/exposure diagnostics, không biến chúng thành classifier metrics mới hoặc dùng để chọn lại winner.
8. Kết quả không đổi/giảm, failure/resume/version history và các verification chưa hoàn tất; không ẩn runs score thấp hoặc báo model chưa hoàn tất như đã pass.

Không gọi Macro F1 là “đánh giá trực tiếp đủ 124 lớp” khi ground-truth support không đủ. F1 của một class có rất ít real test samples có thể đổi mạnh do một prediction; luôn giữ support bên cạnh để diễn giải. Một generated reference centroid không tạo thêm evaluation support cho singleton.

Head/medium data giống nhau giữa augmented conditions không buộc scores của chúng giống nhau vì classifier được train chung. Không gán riêng effect cho DINOv3 hoặc CCFG từ A2 joint addition. Fixed epochs trên pool khác size có thể có optimizer steps/exposure khác E0; riêng A3/A4 phải có steps bằng nhau như đã khóa.

Không thêm p-values, confidence intervals, AUC, calibration scores, geographic distance hoặc metric khác vào mandatory result set. Không đưa teacher/feature-space scores hoặc semantic QC vào evaluation contract. Các Geo-specific differences với official DiSC-DS được dẫn về E3 Spec, không thay công thức/model sau khi thấy kết quả.

## 9. Evaluation acceptance criteria

Các tiêu chí sau là hợp đồng cho implementation, chưa phải kết quả đã kiểm chứng.

| ID | Tiêu chí | Bằng chứng |
|---|---|---|
| EVAL-AC01 | Validation/test membership là đúng locked real splits, đủ mỗi record một lần | Manifest/support identities và prediction coverage audit |
| EVAL-AC02 | Inference dùng common preprocessing/runtime, không train-time augmentation/Mixup hoặc mask logits | Resolved config/hash, model mode và `[N,124]` logits artifacts |
| EVAL-AC03 | Confusion matrix đủ 124×124, đúng row/column convention, row sums/supports và total N | Raw matrices, mapping hash, coverage/count consistency |
| EVAL-AC04 | Macro F1 chỉ average ground-truth-supported classes, unsupported class F1/Recall N/A | C_eval/support artifacts và per-class/count recomputation |
| EVAL-AC05 | Predictions vào singleton/unsupported classes vẫn tính sai, không bỏ sample/cột | Full label-space predictions/matrices và counts |
| EVAL-AC06 | Đúng metric set và definitions mục 3; không thêm metric hoặc adjusted variant | Metric config/schema và overall recomputation |
| EVAL-AC07 | Group counts từ full-split confusion matrix, groups theo pre-augmentation n_train | Locked class_groups và group/per-class consistency |
| EVAL-AC08 | Empty group nhận N/A, không 0; missing run nhận incomplete, không support-based N/A | Status/null/support evidence và aggregation validation |
| EVAL-AC09 | Best checkpoint chọn overall Validation Macro F1, exact-score tie lấy earliest epoch | Full-precision epoch history, best-checkpoint identity/lock |
| EVAL-AC10 | Strength selection dùng đủ ba seeds, mean validation tail score; không đổi checkpoint criterion | Nine-run selection lock/results, evaluated tail IDs và official binding |
| EVAL-AC11 | Test chỉ dùng đủ điều kiện frozen official checkpoint, không candidates/last/best-seed selection | Test gate/dependency record và inference failure history |
| EVAL-AC12 | Reused E1E2 đủ verification/binding ba seeds, source evidence bất biến, Base alias không double-count | Official bindings, source/evaluation hashes và matrix accounting |
| EVAL-AC13 | Từng metric có đủ ba raw seed values, mean ± sample std ddof=1, không pooling/ensemble | Per-seed/aggregate artifacts và recomputation |
| EVAL-AC14 | Main tables đúng E0/E1E2/A4; ablation đúng Base/A1/A2/A3/A4, cùng aliases/evidence | Comparison tables và alias/checkpoint mapping |
| EVAL-AC15 | Per-class đủ 124 labels, support/counts/N/A đầy đủ, không loại lớp khó | Per-class CSVs và report/support consistency |
| EVAL-AC16 | Prediction/logit/metric/report artifacts cùng identity, đầy đủ và hash-locked | Evaluation/report locks, row mapping, immutable source references |
| EVAL-AC17 | Report không bịa improvement hoặc che failure/incomplete/unsupported scope | Final report, raw results và provenance/status history |
| EVAL-AC18 | Ranking logit giảm dần/class ID tăng dần cho exact ties, gồm Top-5 boundary, áp từ first validation | Confirmed EVAL-D01 và evaluator config/conformance evidence |
| EVAL-AC19 | Metric tables hiển thị percentage hai decimals, mean ± sample std; raw values vẫn full precision | Confirmed EVAL-D02, raw-vs-render checks và presentation policy hash |

Không có ngưỡng Accuracy/Macro F1 để quyết định benchmark có hợp lệ hay không. Nghiệm thu yêu cầu thực thi đúng policy, đầy đủ evidence và reporting trung thực; không yêu cầu một condition nhất định phải thắng.

## 10. Confirmed decisions và verification requirements

### 10.1. EVAL-D01/D02 — CONFIRMED

| ID | Quyết định confirmed | Phạm vi |
|---|---|---|
| EVAL-D01 — Exact logit ties | Logits descending; ascending class ID cho exact ties, không tolerance/rounding | Toàn bộ classifier validation/tuning/test ranking và Top-1/Top-5 |
| EVAL-D02 — Presentation formatting | Percentages hai decimals, mean ± sample std; bảo toàn full-precision raw values | Mọi rendered metric tables; không đổi raw metric/selection/aggregation |

Người dùng đã xác nhận cả hai quyết định; không còn pending evaluation policy trong EVAL-D01/D02. Support denominator, metric set, checkpoint criterion, seed aggregation, strength protocol và research matrix giữ nguyên. Các decision/config hashes phải được khóa trước bước liên quan; xác nhận specification không đồng nghĩa verification đã chạy.

### 10.2. Verification khi triển khai

Chưa có trained models, predictions hoặc metrics để chứng minh acceptance criteria. Khi triển khai cần materialize/verify all locks, actual split supports, evaluator ranking/metric conformance, per-seed artifacts và reuse accounting. Các E3 snapshot/token/runtime requirements còn thiếu tiếp tục theo E3 Spec mục 13.3, không biến thành metrics mới hoặc yêu cầu sửa prompts trong task này.

Tài liệu này chỉ đặc tả evaluation/reporting. Chưa triển khai code, chạy inference hoặc tạo bảng số liệu thực nghiệm.
