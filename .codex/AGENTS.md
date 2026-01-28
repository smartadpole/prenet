# AGENTS.md

> Auto-generated from .cursor/rules/*.mdc. Do not edit by hand.

## Global principles
- Follow the rules below. If rules conflict, prefer later sections.
- Keep diffs minimal; do not touch unrelated files.


## Cursor rule: auto-changelog.mdc

# ===========================
#  Changelog Auto-Writer Rule
# ===========================

# Ⅰ. Versioning Rules
- Project uses SemVer: MAJOR.MINOR.PATCH.
- PATCH increments automatically only when the workspace changes from clean → dirty.
- If the workspace is already dirty, do NOT increment PATCH again.
- MINOR and MAJOR are changed only when explicitly requested by the user.

# Ⅱ. Date/Time Rules
- All generated timestamps must use current system time, formatted as YYYY/MM/DD HH:mm.
- NEVER copy timestamps from existing content.
- If a file contains:
      @time: YYYY/MM/DD HH:mm
  → update it automatically to the current timestamp whenever the file is edited.
- When creating a new file with an author/file header, automatically insert:
      @time: YYYY/MM/DD HH:mm

# Ⅲ. Changelog Structure Rules (Flat Structure)
- ALWAYS follow the existing style seen in earlier entries of CHANGELOG.md.
- DO NOT introduce section headers such as “### 新增”, “### 优化”, “### 修复”, etc.
- Use a **flat list** of bullet items:
  - 每条以 “新增：… / 优化：… / 修复：… / 移除：… / 变更：… / 重大变更：…” 开头
  - 保留你现有的“主项 + 子项”子弹结构；子级 bullet 缩进不变。

When generating a new changelog entry, the format MUST be:

## {new_version} - {current_datetime} ({git_user_name})
{English one-line summary}

- 优化：xxxx
- 优化：AAAA
- 新增：yyyy
- 修复：zzzz
- 重大变更：aaaa
…

RULES FOR THIS SECTION:
- {git_user_name} must be retrieved from `git config user.name`.
- The English summary MUST be a single, imperative sentence, one line only, no trailing period.
- DO NOT output extra blocks like:
  “主要变更内容：”
  “Changes:”
  “Summary:”
- DO NOT output any “### xxx” headings.
- DO NOT output numbered lists (1., 2., …).
- NEVER collapse multiple items into one.
- NEVER merge categories; every item must self-identify via its prefix (“新增：/优化：/修复：…”).
- Do NOT delete, compress or rewrite existing bullets; for new versions, keep the same level of detail as before（可参考 1.0.0 的主项 + 子项结构）.

Classification Rules (MANDATORY)
--------------------------------
Every bullet must be classified into one of the following categories by using the correct prefix:
“新增：”, “优化：”, “修复：”, “移除：”, “变更：”, “重大变更：”.

Cursor MUST choose the correct prefix based on the nature of the change.
Using only “优化：” for all items is strictly forbidden.

Below are the REQUIRED definitions for each category:

1. “新增：”
   Used when the change ADDS something that did not exist before:
   - 新增功能
   - 新增命令、新增接口、新增类、新增函数
   - 新增文档、新增测试
   - 新增模块、文件、新增环境变量、新增配置项
   Examples:
   - 新增：添加新的日志文档 utils.md
   - 新增：创建 metrics/new_metric.py 文件

2. “修复：”
   Used when the change fixes an issue or bug:
   - 修复异常、修复报错、修复崩溃
   - 修复路径、拼写、导入错误
   - 修复测试失败
   - 修复错误逻辑、边界条件、异常处理
   Examples:
   - 修复：移除 tests/test_config.py 中不必要的 logging 配置代码
   - 修复：修正 CSV 解析时的编码错误

3. “优化：”
   Used for improvements WITHOUT adding or fixing:
   - 改进性能、结构、可读性
   - 重构代码
   - 简化导入逻辑、减少依赖、统一调用方式
   - 提升可维护性
   Examples:
   - 优化：重构日志系统，统一使用 print(level=...) 输出
   - 优化：简化 analyzer.py 内部的逻辑路径

4. “移除：”
   Used when something is removed:
   - 删除功能、删除文件、删除字段
   - 删除不再使用的模块或逻辑
   Examples:
   - 移除：删除旧版 logging 初始化逻辑
   - 移除：移除 deprecated 模块 config_old.py

5. “变更：”
   Used for backward-compatible behavior changes:
   - 修改默认行为、修改默认参数
   - 配置项变化但仍向后兼容
   Examples:
   - 变更：修改默认的日志级别为 info

6. “重大变更：”
   Used for breaking changes that are NOT backward compatible:
   - API 改名、删除字段
   - 模块/目录重构
   - 需要业务方升级代码
   Examples:
   - 重大变更：重构项目结构至 src/review_core/
   - 重大变更：删除旧版 ingest API

Rules:
- Cursor MUST classify correctly based on these definitions.
- Cursor MUST NOT classify everything as “优化：”.
- Cursor MUST NOT guess randomly; it must read item content carefully.
- Cursor MUST adjust bullet prefix automatically when necessary.


# Ⅳ. Editing Behavior
- Always insert the new changelog block at the very top of CHANGELOG.md.
- Never modify or rewrite historical entries unless the user explicitly asks.
- Preserve indentation, bullet style, and all formatting.
- Only add new content; do not reformat anything else.

# Ⅴ. Forbidden Behavior
- Do NOT generate future timestamps.
- Do NOT copy timestamps from git log or other files.
- Do NOT generate English section titles (Added/Changed/Fixed).
- Do NOT create summary paragraphs.
- Do NOT change bullet prefix words (“新增：优化：修复：…” are mandatory).

# End of rules

## Cursor rule: clean-code.mdc

# Role: Clean Code & Architecture Architect

## Core Philosophy
- **Single Responsibility Principle (SRP):** Each function/class must do one thing and one thing only.
- **Self-Documenting Code:** Code should be readable like a story. Favor clear naming over comments.
- **Separation of Concerns:** Distinctly separate Data, Logic, and Presentation.

## Refactoring & Implementation Rules

### 1. Function Level (Micro)
- **Granularity:** If a function exceeds 40 lines or has more than 2 levels of nesting, break it down.
- **Composition over Bloat:** Use a "Coordinator" function to call small, specialized "Worker" functions.
- **Naming:** Use `Action + Context` (e.g., `calculate_tax_rates`, `fetch_user_profile`). Avoid generic names like `handle`, `process`, `data`.
- **Logic Separation:** Separate "Pure Functions" (math/transformations) from "I/O Functions" (DB/API).

### 2. Class & Object Level (Meso)
- **Size Limit:** Classes should rarely exceed 300 lines. If larger, split into Mixins, Services, or Utils.
- **Encapsulation:** Keep internal state private. Expose behavior, not just data.
- **Dependency Injection:** Don't hardcode dependencies inside classes; pass them in via constructors.

### 3. Architecture & Reports Level (Macro)
- **The "Report" Pattern:** For report generation, strictly follow this pipeline:
    1.  **Fetcher:** Extract raw data.
    2.  **Transformer:** Business logic & calculations (Pure logic).
    3.  **Formatter:** Structure data for specific formats (PDF/CSV/JSON).
    4.  **Exporter:** Handle the file system or stream.
- **Error Handling:** Use guard clauses (early returns) to keep the "happy path" aligned to the left.

## Refactoring Instructions (When asked to "Optimize" or "Refactor")
1.  **Analyze:** Identify "God Functions" and logical clusters.
2.  **Extract:** Move logic into private methods/functions with descriptive names.
3.  **Interface:** Ensure the main entry point reads like a high-level summary.
4.  **Verify:** Ensure no side effects are introduced in pure logic segments.

## Forbidden Patterns
- No "God Classes" that manage everything.
- No "Boolean Trap" (functions with too many flag parameters).
- No deep nesting (3+ levels of `if/for` is a failure).
- No hardcoded configuration or magic numbers.

## Strict Enforcement
- BEFORE generating code, outline the proposed function breakdown.
- AFTER generating code, perform a self-check: "Does any function exceed 40 lines? Is naming purely descriptive?"
- If the complexity is high, the model MUST suggest a class-based abstraction instead of a long script.

## Cursor rule: coding-style.mdc



# Project Code Style Rules

## 1. Comment Rules (Very Strict)
- Source code files MUST NOT contain any Chinese characters in comments;
- All comments MUST use English only;
- If the user request contains Chinese explanation, Cursor must auto-translate comment content into English before inserting into code;
- If translation is ambiguous, Cursor should generate a clear English explanation instead of outputting Chinese;

## 2. Allowed Chinese Content
Chinese characters are ONLY allowed in:
- Log messages (e.g., logger.info("开始执行任务"));
- Business text returned to frontend or API;
- User-facing error messages;
- Documentation (*.md files) outside of source code;
- Test data strings;

## 3. Disallowed Chinese Content
Chinese MUST NOT appear in:
- Code comments (#, //, /* */);
- Variable names, function names, class names;
- Inline type documentation in code;
- Docstrings used for code documentation (Python triple quotes) — must be English;

## 4. Auto Correction Behavior
When Cursor generates or edits code:
1. Automatically convert comment content to English;
2. If Chinese appears in a code comment → remove or rewrite in English;
3. If Chinese appears in log text → keep it unchanged;
4. If Chinese appears in variable/function/class names → auto-convert to meaningful English identifiers;

## 5. Summary
- Comments: **English only**;
- Logs: **Chinese allowed**;
- Identifiers: **English only**;
- Docs: **Chinese allowed**;

## 6. Changelog Language Rules
changelog:
  language: chinese-preferred   # changelog 内容默认使用中文书写

  # CHANGELOG 的正文语言要求：
  rules:
    - description: "CHANGELOG 每条记录的叙述、说明、变更描述均使用中文"
    - description: "CHANGELOG 中允许出现英文文件名、目录名、模块名、函数名、API 名、参数名等技术标识符"
    - description: "允许出现英文代码关键字，例如 class, function, import，但不可出现大段英文叙述"
    - description: "若内容涉及目录结构、路径、版本号等，保持英文原样，不需要翻译"
    - description: "若 Cursor 自动生成 CHANGELOG，必须严格以中文撰写说明性内容"

  # 示例（Cursor 可参照这类写法自动生成）
  examples:
    - "- 修复：`analysis/parser.py` 中正则表达式匹配异常的问题"
    - "- 优化：重构 `prompt_builder`，减少重复代码"
    - "- 调整：目录 `evaluation/` 下的结构更清晰，拆分 metric 模块"
    - "- 新增：支持从 `pyproject.toml` 自动读取版本号"

## Cursor rule: doc.mdc

# Documentation & File Generation Rules

You are strictly bound by the following documentation and file generation standards.

## 1. Documentation Structure Standards
All documentation must be classified into exactly two categories and placed in specific locations.

### A. User & Runtime Documentation (End-User Focus)
* **Purpose:** Guides for installing, running, and using the software.
* **Locations:**
    * `README.md` (Root): High-level introduction, quick start.
    * `README_*.md` (Optional): Module-specific readmes if the project is large.
    * `docs/user/`: Detailed user manuals, configuration guides, or operational playbooks.
* **Tone:** Clear, instructive, non-technical (black box perspective).

### B. Developer Documentation (Contributor/Maintainer Focus)
* **Purpose:** Deep technical context for upgrades, migration, maintenance, and traceability.
* **Location:** Strictly inside `docs/dev/`.
* **Tone:** Technical, architectural, precise (white box perspective).

---

## 2. Developer Documentation Content Matrix
When creating or updating files in `docs/dev/`, you must address the following five pillars to ensure consistency:

### I. Business Logic & Rationale
* **Flows:** Describe core business processes using pseudocode or text flows.
* **Design Decisions:** Explicitly explain "Why this design?" (Design Rationale) to aid future refactoring.
* **Visuals:** Insert Mermaid flowcharts **ONLY** when explaining complex logic branches.

### II. Features & Modules
* **Boundaries:** Define clear module boundaries and responsibilities.
* **Details:** Document implementation details of key feature points.

### III. Tools & Dependencies
* **Internal Utils:** Explain the purpose and usage of internal utility classes.
* **External Libs:** List dependencies with usage context and strictly enforced version limitations.
* **Toolchain:** Document build scripts, testing frameworks, and CI tools.
* **Cleanup**: If a tool is no longer needed due to a requirement change, remove it from the list immediately.

### IV. Interfaces & Data
* **APIs:** Define Input/Output formats, strict Exception handling rules, and error codes.
* **Data:** Explain Database Schema (ER relationships) and key fields.

### V. Requirements & Context Traceability (Crucial for AI Coding)
* **Prompt to Requirement:** Before coding, transform the raw User Prompt into a standardized **Requirement Specification**.
* **Strategy Integration:** Combine the Requirement with a specific **Design Strategy**.
* **Coherence:** Ensure this new requirement logically connects with existing Business Logic (Pillar I), Features (Pillar II), and Tools (Pillar III).
* **Persistence:** Record this "Requirement + Strategy + Implementation" context into a dedicated file (e.g., `doc/dev/requirements_trace.md` or feature-specific doc) to ensure the intent is preserved for future migrations.
* **Conflict Check**: Before coding, check if the new requirement contradicts any existing Logic (Pillar I) or Requirements.

---

## 3. Conflict Resolution & Versioning Protocol (CRITICAL)

If a new requirement conflicts with existing logic, you must strictly follow this **"Supersede & Purge"** flow:

1. **Trace (The "What"):** In your requirements tracking file (e.g., `docs/dev/requirements_trace.md`):
* Mark the old requirement as `[Superseded]`.
* Link it to the new requirement ID.
* *Example:* `~~REQ-01: Sync Processing~~ -> [Superseded by REQ-05: Async Queue]`

2. **Rationale (The "Why"):** In `docs/dev/architecture.md` (Pillar I), explicitly record the **Design Change**:
* State clearly: "We switched from A to B because [Reason]."

3. **Rewrite (The "How"):**
* **Rule:** The documentation must reflect the **current** state.
* **Action:** Completely rewrite the affected Feature (Pillar II) and Interface (Pillar IV) sections. Do not keep "legacy descriptions" alongside new ones.

4. **Purge (The "Cleanup"):**
* Remove implementation code, configurations, and dependencies (Pillar III) associated with the old logic. **No zombie code.**

---

## 4. Anti-Clutter & Update Protocols
* **Clean Output:** NEVER generate temporary files, log files, `.bak`, or random notes outside `docs/`.
* **Atomic Updates:** Code changes trigger immediate documentation updates.
    * *Rule:* Code Change -> Update corresponding `docs/dev/` section -> Verify consistency with Requirements.
* **No Redundancy:** Do not create separate "update logs". The documentation reflects the *current* state of the system including its design history.

## 5. Enforcement Checklist
Before outputting any file, verify:
1.  **Target Audience:** Is this for End User (`docs/user`) or Developer (`docs/dev`)?
2.  **Content Depth:** Did I cover Logic, Features, Tools, Data?
3.  **Traceability:** Did I convert the *User Prompt* into a *Standardized Requirement* and document the *Design Strategy* in `doc/dev/`?
4.  **Conflict Handling**: Did I explicitly mark old conflicting requirements as superseded and explain the rationale?
5.  **Cleanup:** Is this a random note? -> DELETE IT. Did I purge the old logic/dependencies from the docs and code?

## Cursor rule: file_head.mdc

# Python File Header Rules

All newly created Python files MUST include the standard file header.
When I ask you to create a new Python file, generates the content starting with:

#!/usr/bin/env python3
# encoding: utf-8
'''
@author: <AUTHOR_NAME>
@contact: <CONTACT_EMAIL>
@file: <FILENAME>
@time: <CURRENT_DATETIME>
@desc: <DESCRIPTION>
'''

## Field Filling Instructions:
1.  **@author & @contact**:
    - Attempt to infer the author and email from the current git repository context (e.g., other files' headers or git config if visible).
    - If you cannot infer it, use:
      - @author: 孙昊
      - @contact: smartadpole@163.com
    - (You can modify the default values above to match your preference).

2.  **@file**:
    - Use the actual filename (e.g., `analyzer.py`).

3.  **@time**:
    - Use the current time in the format: `YYYY/MM/DD HH:MM` (e.g., 2025/11/18 16:17).

4.  **@desc**:
    - Generate a concise summary of what the code does based on my instructions or the code context.

5.  **Strict Adherence**:
    - Do not omit this header in new Python files.
    - Ensure strict indentation and spacing as shown in the example.

## Cursor rule: no-repeat.mdc


# 5. DRY & Reuse Rules（避免复制粘贴）

当生成或修改代码时，必须遵守以下约束：

1. 禁止复制粘贴式重复  
   - 不允许在不同文件或函数中直接复制粘贴相同或高度相似的代码块  
   - 一旦发现逻辑重复（包括测试代码），必须抽取为公共函数、类或工具模块，并通过调用复用  

2. 数据/配置只定义在单一来源  
   - 测试数据、常量、配置、枚举等**必须只在一个地方定义**  
   - 其他模块只能通过 `import` / 调用接口 获取这些数据，禁止在多个地方手写/重复定义  
   - 如果发现相同的测试数据或常量在多个文件中出现，优先抽取到统一的 `fixtures` / `config` / `constants` 模块  

3. 重复代码必须封装  
   - 任何被多处使用的逻辑（包括测试前置步骤、数据构造、日志格式化、参数校验等），都必须封装成可复用的函数或类  
   - 封装后的接口要清晰、单一职责，方便后续修改时能够**一处修改，全局生效**  
   - 在重构时，优先消除复制粘贴代码，再考虑优化实现细节  

4. 修改策略  
   - 当 Cursor 需要新增功能时，如遇到类似逻辑已存在，必须优先复用或扩展已有实现，而不是新写一套  
   - 当用户请求「简单复制某段代码到别处」时，Cursor 应优先建议封装和复用方案，而不是直接黏贴重复代码

## Cursor rule: timing.mdc


# ✅ Cursor 时间规则（最终版，结构化、可直接复制）

````yaml
## 日期与时间规则（精确到分钟）

### 1. 全局格式要求
- 所有自动生成的时间格式统一为：`YYYY/MM/DD HH:MM`
  - 示例：`2025/11/24 10:32`
- Cursor 必须使用当前系统时间生成日期，不得从其他文件复制已有日期。
- 禁止生成秒、毫秒；时间精确到“分钟”即可。

---

### 2. 文件头注释（@time）
当 Cursor **创建新文件**或**首次生成文件头**时：

- 如果文件头包含 `@time:` 字段，则自动填入当前时间；
- 若文件是从其他文件复制/重构得到，`@time` 仍必须更新为当前时间；
- 禁止继承源文件的旧时间。

示例（Cursor 自动生成）：
```python
'''
……
@file: evaluator.py
@time: 2025/11/24 10:32
@desc: Main metrics program, integrates all metrics
'''
````

当 Cursor **编辑现有文件**时：

* 不得自动修改已有 `@time` 字段，除非用户明确要求更新。

---

### 3. Changelog 时间规则

在 changelog 中：

* 每次新增一个版本条目时，Cursor 自动补充当前时间（精确到分钟）；
* 修改历史条目内容时，不得更新其时间；
* 禁止从其他条目复制时间用于新条目。

示例：

```markdown
## [1.0.12] - 2025/11/24 10:32
……
```

---

### 4. 其它可能出现的时间字段

对于 `@created:`、`@updated:` 等类似时间字段：

* 新建文件 → 自动填入当前时间；
* 修改文件 → 不主动更新已有时间；
* 仅在用户明确要求时才更新。

---

### 5. 自动遵守但无需用户编写的隐含规则

* Cursor 不应生成动态“当前时间常量”写入源代码（如 Python 中的硬编码日期）。
* 单元测试中禁止使用实时动态日期，避免随着时间推移导致测试不稳定。
* 所有文档（Markdown / LaTeX / 说明文档）中新建文件时可自动填日期，但编辑历史文档时不更改日期，除非用户要求。

```

## Cursor rule: ui-language.mdc

# 跨平台/跨框架 i18n & 双语强制规范

## 1. 核心目标
- 彻底杜绝代码中出现硬编码的中/英文字符串（面向用户部分）。
- 确保所有 UI 文本均通过翻译函数/宏进行包装。
- 保证中英文资源文件（JSON/PO/YAML）的 KV 对 100% 同步。

## 2. 框架适配规则 (Framework Agnostic Patterns)

### A. Python UI (PySide, PyQt, Streamlit, Flet, Custom)
- **禁止**: `label.text = "提交"` 或 `st.write("Welcome")`
- **强制**: 使用翻译函数，如 `label.text = _("submit")` 或 `st.write(t("welcome"))`。

### B. Web 模板 (Jinja2, Django Template, HTML)
- **禁止**: `<button>保存</button>`
- **强制**: 使用模板标签，如 `<button>{{ _('save') }}</button>` 或 `{% trans "save" %}`。

### C. 前端框架 (React, Vue, Alpine.js)
- **禁止**: `<span>Name:</span>` 或 `placeholder="Enter password"`
- **强制**: 
  - JSX: `<span>{t('name')}:</span>`
  - Vue: `<span>{{ $t('name') }}</span>`
  - Attributes: `:placeholder="t('enter_password')"`

## 3. 翻译管理流 (Translation Workflow)
1. **识别与提取**: 
   - 发现硬编码文本 -> 自动生成语义化 Key (如 `error_timeout`) -> 用 `_()` 或 `t()` 替换原文本。
2. **多语言更新**:
   - 自动检测项目中的语言库目录（如 `locales/`, `translations/`, `i18n/`）。
   - **同步写入**: 在修改代码的同时，必须在 `zh` 和 `en` 文件中同步添加对应的 Key-Value。
3. **命名规范**:
   - Key 统一使用 `snake_case` 或 `dot.notation`。
   - 禁止使用数字命名（如 `text_1`），必须反映业务含义。

## 4. 自动化指令 (For Cursor AI)
- **代码审查**: 在生成或修改 UI 代码时，如果看到引号内包含中文字符或非变量的英文短语，立即重构为 i18n 形式。
- **缺失提醒**: 如果我添加了一个 Key 但忘记定义翻译，请主动询问：“我需要为您在 zh 和 en 文件中补充这个词条吗？”
- **上下文推断**: 根据周围代码判断应使用哪种翻译函数（如在 Flask 里用 `gettext`，在 Vue 里用 `$t`）。

## 5. 排除项
- 代码内部逻辑标识符（变量名、枚举常量）。
- 后端非用户可见的 Log 日志。
- 纯技术参数（API URL、颜色代码、CSS 类名）。

## Cursor rule: version-control.mdc


# Version & Changelog Rules

## 1. 版本号基本约束（SemVer）

- 项目统一使用语义化版本号：`MAJOR.MINOR.PATCH`；
- 除非我在对话里**明确说明要发大版本 / 次版本**，否则：
  - MAJOR 不允许自动改动；
  - MINOR 不允许自动改动；
  - 任何版本更新**一律只允许自增 PATCH（修正版本号）**；
- 示例：当前为 `0.3.5` 时，允许自动改为 `0.3.6`，**不允许**改为 `0.4.0` 或 `1.0.0`，除非我另行说明；

## 2. 版本号与包内版本保持一致

当 Cursor 修改代码时，必须保证各处版本号一致，尤其是：

- 包内版本定义（示例，按项目类型自动识别）：
  - Python：`pyproject.toml` / `setup.cfg` / `setup.py` / `__init__.py` 中的 `__version__`；
  - Node：`package.json` 中的 `version`；
  - 其他语言：遵循项目里已有的版本字段；
- `CHANGELOG` / `CHANGELOG.md` / `docs/CHANGELOG.md` 中最近一条版本记录的版本号
- 规则：  
  - 当需要更新版本时，**先读取现有版本号**，再在此基础上 PATCH+1；
  - 更新后，**包内版本字段与 changelog 中最新版本号必须完全一致**（字符串完全相同）；

## 3. 何时必须自增 PATCH（修正版本号）

- 只要满足以下任一条件，就视为“需要版本更新”，必须执行 PATCH+1：
  - 修改了**业务代码 / 核心逻辑**（非纯注释、非格式化）；
  - 修改了**对外行为**（返回值、API、配置解释方式等），即便是小改动；
  - 修改了**打包配置、依赖版本**，会影响构建结果；
- 以下情况**可以不改版本号**（由我决定是否改）：
  - 只修改测试代码（`tests/`、`*_test.py` 等）；
  - 只修改文档（`*.md`、`docs/`）；
  - 只做格式化 / 排版调整（不改变语义）；
- 当你判断“是否需要 bump 版本”时，遵循：  
  - **宁可多自增 PATCH，也不要少自增**；如果不确定，就建议自增 PATCH，并在改动说明中标记原因；

## 4. 更新版本时的具体行为

当 Cursor 进行一次需要版本更新的修改时，应遵循以下步骤：

1. **读取当前版本号**：
   - 从包内版本字段中读取（如 `__version__` 或 `package.json` 的 `version`）；
2. **计算新版本号**：
   - 只修改 PATCH：`MAJOR.MINOR.(PATCH+1)`；
3. **统一写回**：
   - 在所有定义版本的文件中写入相同的新版本号；
   - 确保 changelog 中最新记录的版本号与包内版本号完全一致；
4. **CHANGELOG 记录**：
   - 如果 changelog 已存在当前版本标题：
     - 在对应版本下追加本次修改条目，而**不创建新版本号**；
   - 如果 changelog 没有当前新版本：
     - 创建形如 `## vX.Y.Z - YYYY-MM-DD` 的新条目；
     - 在其下用列表简要说明本次变更（至少 1 条），例如：
       - `- Fix: xxx`
       - `- Refactor: yyy`
5. 若由于上下文不完整无法确定版本号位置或格式：
   - 不要随意猜测写入；
   - 在生成说明中**明确指出“需要手工对齐版本号和 changelog”**，并给出推荐的新版本号字符串；

## 5. 与我交互时的说明习惯

- 当你因为一次修改而自增了 PATCH 时：
  - 在最终回答中用一行简要说明，例如：
    - `Version bump: 0.3.5 → 0.3.6 (patch only, keep MAJOR.MINOR unchanged)`
- 如果你认为应该更新 MINOR 或 MAJOR：
  - **不要直接改**，而是先在回答中说明理由，并提示我确认：
    - `当前改动影响到对外 API，建议从 0.3.6 升级为 0.4.0，请确认后再执行`


## 6. 工作区变更控制（非常重要）

# 核心原则：只有在“工作区原本干净 → 本次修改导致变脏”时，才允许自增 PATCH。
# 如果工作区本来已经有修改（dirty），Cursor 不得因为继续编辑而重复升级版本。

# 规则描述：
# 1. Cursor 必须在执行版本号 bump 之前判断工作区状态：
#    - 若 workspace 是 clean（无未提交修改），且本次操作修改了业务代码 → 允许 PATCH+1；
#    - 若 workspace 已经是 dirty → 禁止重复 bump，同一轮开发只允许 bump 一次；
# 2. 在单次编辑 Session 内，如果已经 bump 过版本号 → 禁止再次 bump，直到用户完整完成一次 commit。

versionBump:
  workspace:
    requireCleanBeforeChange: true      # 需要干净 → 修改 才能 bump
    forbidBumpIfAlreadyDirty: true      # 工作区脏时禁止 bump
    bumpOncePerSession: true            # 一个编辑 session 内最多 bump 一次

## 7. 版本号更新的判定逻辑

# Cursor 判断是否需要 bump PATCH 时必须按以下优先级执行：

# Step 1. 判断是否修改了业务代码
versionBump.changeTypesRequiringPatch:
  - source_code        # *.py, *.ts, *.js, *.go, *.java, 等业务源文件
  - configs_affecting_build  # 如 pyproject, package.json 等
  - public_api_changes

# 以下修改 **不触发版本号 bump**（由用户手工控制）
versionBump.changeTypesNotRequiringPatch:
  - documentation      # *.md
  - tests              # tests/*
  - formattingOnly     # 代码格式化、黑化、排序
  - commentsOnly       # 只改注释
  - samplesOrExamples
