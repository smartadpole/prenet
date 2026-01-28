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

## 冲突与替代记录

（当某需求被后续需求替代时，在此标记并链接新需求 ID。）

- 暂无。

---

## 维护说明

- 新增需求：按上表格式追加一条，ID 递增（REQ-002, REQ-003, …）。
- 需求被替代：在「需求列表」中该条目标注 `[Superseded by REQ-xxx]`，并在「冲突与替代记录」中简要说明。
- 实现或文档变更：更新对应条目的 Implementation / Version，保证与 CHANGELOG 和代码一致。
