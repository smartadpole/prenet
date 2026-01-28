# Developer Documentation Index

This directory contains technical documentation for developers and maintainers of the PRENet project, with a focus on the DINOv2 ArcFace training system.

## Documentation Structure

### Architecture & Design

- **[DINOv2 Training Architecture](dinov2_training_architecture.md)**
  - Complete system architecture overview
  - Model components (DINOv2 Embedder, ArcFace Head)
  - Two-stage training strategy
  - Optimization and learning rate scheduling
  - Design decisions and rationale

- **[Training Data Augmentation](training_data_augmentation.md)**
  - Fine-grained classification optimized augmentation strategy
  - Parameter specifications and design rationale
  - Comparison with previous parameters
  - Future improvement suggestions

- **[Data Loader Design](data_loader_design.md)**
  - Unified data loader architecture
  - Supported dataset formats (ImageFolder, Text List)
  - Auto-detection logic
  - API reference and usage examples

### Testing & Evaluation Tools

- **[DINOv2 Classification Testing](test_dinov2_classification.md)**
  - Classification test tool documentation
  - Batch inference implementation
  - Visualization features

- **[Full Pipeline Testing](test_full.md)**
  - CSV-based batch inference with bbox cropping
  - Multi-frame processing (take/return scenarios)
  - Category-based image saving
  - Visualization features

- **[Open-Set Recognition Evaluation](eval_open_set.md)**
  - Open-set recognition evaluator documentation
  - Offline feature gallery construction
  - Top-K nearest neighbor search
  - Weighted voting mechanism
  - Open-set rejection logic

- **[Template Library Generation](gen_sku_template_library.md)**
  - Fastdup-based clustering sampling for template library generation
  - Center sampling and hybrid sampling strategies
  - Batch processing and auto-detection modes
  - Performance optimization for complex environments

## Documentation Standards

All developer documentation follows the **Five Pillars** structure:

### I. Business Logic & Rationale
- Core business processes and flows
- Design decisions and "why" explanations
- Mermaid flowcharts for complex logic (when needed)

### II. Features & Modules
- Module boundaries and responsibilities
- Implementation details of key features

### III. Tools & Dependencies
- Internal utility classes and their purposes
- External libraries with version constraints
- Build scripts and toolchain

### IV. Interfaces & Data
- API input/output formats
- Exception handling rules
- Data schemas and key fields

### V. Requirements & Context Traceability
- **[Requirements Traceability](requirements_trace.md)**：Prompt → 需求规格 → 设计策略 → 实现的完整追溯
- Requirement specifications and design strategies
- Connection to existing business logic and conflict resolution

## Quick Links

### For Training
1. Start with [DINOv2 Training Architecture](dinov2_training_architecture.md) for system overview
2. Review [Training Data Augmentation](training_data_augmentation.md) for augmentation strategy
3. Check [Data Loader Design](data_loader_design.md) for dataset format support

### For Testing
1. [DINOv2 Classification Testing](test_dinov2_classification.md) for single image classification
2. [Full Pipeline Testing](test_full.md) for batch CSV processing
3. [Open-Set Recognition Evaluation](eval_open_set.md) for open-set recognition with offline feature gallery
4. [Template Library Generation](gen_sku_template_library.md) for generating diverse template libraries

### For Maintenance
- **[Requirements Traceability](requirements_trace.md)**：新增/变更需求时在此记录 ID、规格、策略与实现，便于迁移与冲突检查
- All documents include design rationale sections explaining "why"
- Parameter comparison tables show evolution of design decisions
- Future improvements sections outline potential enhancements

## Version Information

- **Current Version**: 0.1.16
- **Last Updated**: 2026/01/28
- **Key Changes**: Refactor gen_sku.py CLI to --input/--output, add file-list mode and leaf-dir recursive mode; output single txt template list

## Contributing

When adding or updating documentation:

1. **Follow the Five Pillars structure** for comprehensive coverage
2. **Include design rationale** explaining why decisions were made
3. **Update this index** when adding new documents
4. **Maintain consistency** with existing documentation style
5. **Link related documents** for easy navigation

## Related Documentation

- **User Documentation**: `docs/user/` - End-user guides and tutorials
- **Project README**: `README.md` - High-level project overview
- **Changelog**: `CHANGELOG.md` - Version history and changes
