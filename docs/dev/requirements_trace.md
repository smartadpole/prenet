# Requirements Traceability

本文档记录「用户/对话 Prompt → 需求规格 → 设计策略 → 实现」的追溯关系，便于后续迁移与冲突检查。所有条目按需求 ID 组织，并与版本、代码、文档挂钩。

## 格式说明

| 字段 | 说明 |
|------|------|
| **ID** | 唯一需求编号（REQ-xxx） |
| **Source** | 需求来源（用户 Prompt / 讨论摘要） |
| **Requirement** | 标准化需求描述 |
| **Design Strategy** | 设计策略与取舍 |
| **Implementation** | 实现位置（代码/配置/文档） |
| **Version** | 纳入的版本号 |
| **Coherence** | 与现有业务/模块/工具的一致性说明 |

---

## 需求列表

### REQ-001: 闭集推理支持类别过滤（allowed_indices）

| 项目 | 内容 |
|------|------|
| **ID** | REQ-001 |
| **Source** | 在 CSV 批量推理场景下，当提供 `--label_file` 时，模型输出头类别数可能大于标签文件定义的类别数，需要将预测限制在「标签文件中定义的类别」子集内，避免选中无关或越界类别。 |
| **Requirement** | 1) 闭集推理接口支持可选参数，将预测限制在指定类别索引子集内；2) `test_full.py` 在使用 `--label_file` 时，根据有效类别名推导允许的类别索引并传入推理，实现过滤模式；3) 过滤无效或空集合时回退为不过滤，并给出明确日志。 |
| **Design Strategy** | 在 `eval.py` 的 `classify` / `classify_batch` 中新增可选参数 `allowed_indices`；在 logits 上做掩码（非允许位置置为 -inf）再 softmax/argmax，保持接口向后兼容。`test_full.py` 从 `load_label_file` 得到类别列表后，用「非 None 类别名对应的索引」构造 `allowed_indices` 并传入；若过滤后无有效索引则置为 None（不过滤）。 |
| **Implementation** | `tools/eval.py`: `classify(..., allowed_indices=None)`, `classify_batch(..., allowed_indices=None)`；`tools/test_full.py`: 加载 label 后计算 `allowed_indices` 并传入 `classify_batch`；`docs/dev/test_full.md`: 推理模块说明、主流程、设计决策「类别过滤」；CHANGELOG 0.1.17。 |
| **Version** | 0.1.17 |
| **Coherence** | 与现有闭集评估流程一致，仅扩展可选参数；复用既有 `load_label_file` 与 `classify_batch` 调用链；不影响无 `--label_file` 或无有效类别时的行为。 |

---

### REQ-002: 开集评估输入输出与 eval 对齐并支持模板目录

| 项目 | 内容 |
|------|------|
| **ID** | REQ-002 |
| **Source** | 用户要求开集评估输入输出与 `eval.py` 保持一致，只新增 SKU 模板输入；模板支持文件列表和目录两种；输出配置与可视化与闭集 eval 对齐，未知预测计为错误并标记为 Unknown/Rejected。 |
| **Requirement** | 1) `eval_open.py` 支持 `--template_path` 输入模板，兼容 `.txt/.csv` 三列文件与目录子文件夹两种格式；2) 目录模式下按子目录名作为类别名并自动分配 label_id；3) 评估输出文件与 `eval.py` 一致（`evaluation_results.txt` + 按类可视化）；4) 预测为未知类时计入错误并在可视化中标记为 `Unknown/Rejected`；5) 保持 Top-K 投票使用相似度与类别规模归一化的权重公式。 |
| **Design Strategy** | 复用 `eval.py` 的 `load_test_file` 与 `visualize_results`，新增 `load_template_data` 支持文件/目录两种模板输入。引入 `calculate_metrics_with_unknown` 将未知预测纳入总数并计为错误，保持评估输出格式与 `eval.py` 一致。投票权重仅使用相似度与 `log(N_c+1)` 归一化，去除 rank 权重以符合公式要求。 |
| **Implementation** | `tools/eval_open.py`: 新增 `load_template_data`/`calculate_metrics_with_unknown`，支持 `--template_path`，输出对齐 `evaluation_results.txt` 与可视化；`docs/dev/eval_open_set.md`: 更新输入格式、投票公式、输出一致性说明。 |
| **Version** | 0.1.20 |
| **Coherence** | 与闭集 `eval.py` 保持同样的输出格式与可视化流程，仅增加模板输入与开集拒识逻辑；不影响现有闭集工具。 |

---

### REQ-003: 开集评估支持可控可视化与模板库缓存

| 项目 | 内容 |
|------|------|
| **ID** | REQ-003 |
| **Source** | 用户要求可视化结果通过参数控制是否保存、功能封装、模板库默认缓存。 |
| **Requirement** | 1) 增加参数控制是否保存可视化结果，默认保持可视化输出；2) 模板库默认开启缓存，支持禁用开关与指定缓存目录；3) 抽取评估结果与可视化保存逻辑为可复用函数。 |
| **Design Strategy** | 在 `eval_open.py` 中新增 `--save_vis` 参数，默认启用缓存但关闭可视化；通过封装 `_write_evaluation_results`、`_save_visualizations` 等函数减少主流程重复；模板库缓存基于模板指纹与模型路径生成 key，缓存特征与标签信息，缓存目录与模板目录同级并按模型版本区分。 |
| **Implementation** | `tools/eval_open.py`: 新增缓存与可视化开关参数、缓存读写函数、评估结果与可视化输出封装；`README.md` 与 `docs/user/eval_open_set.md`、`docs/dev/eval_open_set.md`: 补充参数说明与默认行为。 |
| **Version** | 0.1.20 |
| **Coherence** | 保持现有评估输出结构与逻辑一致，仅添加可选行为控制与缓存层；不影响无缓存或关闭可视化时的评估统计。 |

---

## 冲突与替代记录

（当某需求被后续需求替代时，在此标记并链接新需求 ID。）

- 暂无。

---

## 维护说明

- 新增需求：按上表格式追加一条，ID 递增（REQ-002, REQ-003, …）。
- 需求被替代：在「需求列表」中该条目标注 `[Superseded by REQ-xxx]`，并在「冲突与替代记录」中简要说明。
- 实现或文档变更：更新对应条目的 Implementation / Version，保证与 CHANGELOG 和代码一致。
