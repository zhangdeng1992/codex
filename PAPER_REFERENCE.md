# 论文代码对照

目标论文：*A unified cell atlas of vascular plants reveals cell-type foundational genes and accelerates gene discovery*。

- DOI：https://doi.org/10.1016/j.cell.2025.07.036
- 公开仓库：https://github.com/xuzhougeng/CrossSpeciesPlantShootAtlas
- 核查的提交：`ca2bcdd39c2e55cf8ea6b9120fd18f096234fa5a`
- SAMap 目录：https://github.com/xuzhougeng/CrossSpeciesPlantShootAtlas/tree/main/02_CrossSpeciesIntegration/SAMap

仓库 README 明确引用目标论文。以下结论来自公开代码，不代表已核实论文全文中的所有实验设置。

| 作者代码 | 本项目的对应实现 |
| --- | --- |
| `SAMap_prepare.ipynb`：每物种 `SAM.load_data → preprocess_data → run` | 各物种独立构建 SAM 对象 |
| 多样本物种 `sam.run(batch_key='orig.ident')` | `--batch-key gb=orig.ident`，在物种内部做 Harmony |
| `pairwise_blastp.sh`：所有蛋白组两两双向 BLASTP | `build_protein_maps.py`，保留 `outfmt 6`、`max_hsps 1`、`evalue 1e-6` |
| `SAMap_integration.ipynb`：预处理 SAM 对象传给 SAMAP | 直接传内存 SAM 对象 |
| `sm.run(pairwise=True)` | 显式启用 `pairwise=True`，另外提供迭代数和线程参数 |

## 使用示例：裸子植物与被子植物

`gb`、`at` 是本项目示例物种 ID，分别表示银杏和拟南芥；文件名和批次列需要替换成你的实际数据。
示例不表示作者 notebook 使用了这两个物种。

```bash
python build_protein_maps.py \
  --protein gb=data/ginkgo.pep \
  --protein at=data/arabidopsis.pep \
  --out maps/plant --threads 4

python samap_integrate.py \
  --species gb=data/ginkgo.h5ad \
  --species at=data/arabidopsis.h5ad \
  --maps maps/plant --out results/gb_at \
  --counts-layer counts --label-key cell_type \
  --batch-key gb=orig.ident --threads 4
```

没有多个批次时省略 `--batch-key`；不要将物种作为 Harmony 批次。
每个物种的蛋白 FASTA ID 必须与该物种表达矩阵基因 ID 对应。
多个转录本对应一个基因时，先根据注释明确选择代表蛋白或制定 ID 映射规则；
不能简单按字符串裁剪所有物种的 ID，也不能把一对一同源筛选强加到 SAMap 的多对多同源图上。

## 与作者代码的差异及验证边界

作者 notebook 使用旧接口 `samap.mapping.SAMAP` 与 `samalg.SAM`。
本项目使用 `sc-samap==3.0.1` 的 `samap.SAMAP` 与内置 `samap.sam.SAM`，
属于参考其分析步骤的新版实现，不是作者软件环境或论文数值结果的精确复现。
Harmony 固定为 `0.0.10`，对应当前 SAM 使用的 `Z_corr.T` 输出布局。

作者 BLAST 脚本会同时启动所有方向的任务，且每个任务用 40 线程。
本项目顺序执行，每个任务使用 `--threads`，避免多物种时 CPU 过度订阅；
数据库放在输出目录，保留源蛋白文件，失败时停止。

公开仓库还提供 SATURN、AUCell、调控网络等代码；当前脚本仅覆盖 SAMap 分支，
不声称复现整篇论文或直接识别 foundational genes。
现有源码核查与命令行检查不能替代真实数据的同源覆盖、映射效果与批次校正验证。
