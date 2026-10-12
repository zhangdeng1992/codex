# codex
codex 调用

# SAMap 跨物种单细胞对齐

本脚本使用 SAMap 3.x，根据基因同源关系建立跨物种细胞图，支持两个或更多物种。
现已参考 Cell 论文的公开分析代码，加入物种内 Harmony 和双向 BLASTP 生成；
详见 [论文代码对照与植物示例](PAPER_REFERENCE.md)。
同源表生成默认使用 DIAMOND，也可通过 `--aligner blast` 使用 BLASTP。

### DIAMOND 同源表生成

需要独立安装 DIAMOND 2.x（不包含在 pip 依赖中）：

```bash
conda install -c conda-forge -c bioconda diamond
python build_protein_maps.py \
  --protein gb=data/ginkgo.pep --protein at=data/arabidopsis.pep \
  --out maps/plant --aligner diamond --threads 4
```

默认启用 `--ultra-sensitive`、`--evalue 1e-6`、`--max-hsps 1` 和
`--max-target-seqs 0`，输出双向标准 12 列表，后续 SAMap 命令保持一致。
结果不保证与 BLASTP 相同，应比较同源覆盖及细胞映射效果。
联合对象用于跨物种映射与可视化；其表达矩阵不是统一同源基因空间的批次校正矩阵，
不要直接用于跨物种差异表达。若需要这种表达矩阵，需要另行制定同源基因汇总规则。

## 安装

需要 Python 3.11 或更高版本。

```bash
cd /workspace/codex
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 输入准备

- 每个物种一个 `.h5ad`，细胞为行、基因为列，事先完成 QC 和双细胞处理。
- `X` 应为原始非负整数 counts；若原始 counts 在 `layers['counts']`，使用 `--counts-layer counts`。
  不要传入 log-normalized、scaled 或 integrated 表达矩阵。
- `var_names` 必须与 BLAST 使用的 FASTA 序列 ID 一致，不能一边用基因符号、一边用 Ensembl ID。
  不要直接重命名重复基因，先明确重复 ID/转录本如何汇总。
- 可选：`obs['cell_type']` 保存细胞类型标签。没有标签的物种自动使用 Leiden 聚类。
- 物种 ID 为两个小写字母，如 `hs` 和 `mm`。每对物种都需要双向 BLAST 表。

```text
maps/
└── hsmm/
    ├── hs_to_mm.txt
    └── mm_to_hs.txt
```

表是无表头、制表符分隔的标准 BLAST `outfmt 6`，共 12 列，包含 E-value 和 bit score。
不能用只有两列的 ortholog 表替代。SAMap 会筛选同源关系；基因 ID 不匹配或同源基因过少
会导致失败或不可靠的映射。

若还没有同源表，可使用官方脚本生成（需要 NCBI BLAST+，例如通过
`conda install -c bioconda blast` 安装）：

```bash
git clone https://github.com/atarashansky/SAMap.git SAMap_tools
# 在当前目录的 maps/ 中生成结果；prot 表示蛋白序列，nucl 表示核酸序列。
bash SAMap_tools/map_genes.sh \
  --tr1 human_proteins.fasta --t1 prot --n1 hs \
  --tr2 mouse_proteins.fasta --t2 prot --n2 mm
```

## 运行

```bash
python samap_integrate.py \
  --species hs=data/human.h5ad \
  --species mm=data/mouse.h5ad \
  --maps maps --out results/hs_mm \
  --counts-layer counts --label-key cell_type --threads 4
```

如果原始 counts 在 `X`，省略 `--counts-layer counts`。
添加第三个物种时再传一个 `--species`，并准备全部物种对的双向同源表。
脚本不修改源数据，不覆盖已有输出目录；失败后的重跑请使用新目录。
随机种子默认 0，但并行计算不保证逐位一致。

## 输出

- `samap_joint.h5ad`：SAMap 联合对象，含物种信息、联合图和 UMAP。
- `mapping_scores.csv`：细胞类型或聚类之间的映射分数矩阵。
- `mapping_ranking.csv`：SAMap 返回的映射排序表。
- `cells.csv`、`umap.csv`：细胞元数据和坐标。
- `hs_processed.h5ad` 等：各物种处理后的对象。
- `inputs/`：本次使用的 counts 副本；`run.json`：参数及 SAMap 版本。

映射分数不是概率，不能直接解释为确定的细胞类型对应关系；需要结合已知 marker、
基因同源覆盖和生物学信息审核。

官方项目：https://github.com/atarashansky/SAMap
