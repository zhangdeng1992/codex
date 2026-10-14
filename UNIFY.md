# Unify：非洲菊 GB 与甘菊 CL15 跨物种整合

入口 `unify_integrate.py` 调用官方 Unify 的 `integration` 模式。
默认沿用 CL15 的 rep1/rep2 子集，原始文件不变。原始计数必须在 X 或显式指定的 layer 中。
当前核查的上游提交：`40597f69dcb3b8e8952c20a5c989140f6314d183`。

## 必需输入

| 物种 | H5AD | 细胞标签列 | 蛋白嵌入 | 功能文本嵌入 |
| --- | --- | --- | --- | --- |
| GB | Gerbera_scRNA.all_samples.2026-Sep-15.h5ad | cell-type | gb_ESM2.pt | gb_text.pt（需准备） |
| CL | CL15_rep1_rep2.h5ad | celltype | cl_ESM2.pt | cl_text.pt（需准备） |

两类嵌入都必须是 `本物种基因ID -> 一维浮点Tensor` 字典。蛋白嵌入在两物种间须使用同一 ESM 模型；
文本嵌入也须使用同一个文本模型和提取方法。两种模态之间的向量维度可以不同。
不能把蛋白嵌入复制一份作为文本嵌入，也不能用随机向量代替缺失的真实功能描述。

拟南芥同源注释可以用于本物种功能描述：保留 GB/CL 原始基因 ID，按明确的对应关系关联
TAIR 功能描述或 GO term 名称/定义。不要把表达矩阵 var_names 重命名为拟南芥 ID。
仅有 `AT1G...` 或 `GO:...` 标识符时，还需要取得对应的描述。多个本物种基因可对应同一个拟南芥基因。
多对多映射应保留证据与规则，不能随意取第一条；不要把转移的同源注释当作本物种的实验验证。
原始注释表的列名和 ID 类型尚需核对，尤其是蛋白 ID 到表达基因 ID 的转换。

用户提供的 GB 表头是 `gerbera_id, ath_acc, ath_symbol, ath_description`（实际为制表符分隔），
可直接使用新脚本整理。假设保存为 `Gerbera_ath_annotations.tsv`：

```bash
python codex-main/unify_prepare_descriptions.py \
  --annotations Gerbera_ath_annotations.tsv \
  --h5ad Gerbera_scRNA.all_samples.2026-Sep-15.h5ad \
  --gene-column gerbera_id --description-column ath_description \
  --strip-suffix '\.m[0-9]+$' --out gb_unify_annotations
```

将下文生成 GB 嵌入命令中的输入换成 `--descriptions gb_unify_annotations/gene_descriptions.tsv`。
脚本跳过 `no full name available` 等空描述，按 H5AD ID 核对，保存 `annotation_evidence.csv`、
`genes_without_descriptions.csv` 和 `coverage.json`；同一基因的重复描述去重，不同描述保留并在嵌入时取平均。
CL 表的前两列为 `gene`、`tair_id`，后两列没有列名，分别是 `Symbols:` 和功能描述。
保存为制表符分隔的 `CL15_ath_annotations.tsv`，使用专用格式读取：

```bash
python codex-main/unify_prepare_descriptions.py \
  --annotations CL15_ath_annotations.tsv --annotation-format cl-tair \
  --h5ad CL15_rep1_rep2.h5ad --out cl_unify_annotations
```

该表的 ID 已是 `CL1A000080`，不需要去后缀。格式读取会处理空列名、字段空格和引号，
保留描述中的逗号，并把 `tair_id` 和 symbol 保存到来源表；空描述仍跳过。

## 安装

建议独立 Python 3.10 环境，服务器现有驱动支持 CUDA 12.4 时可使用：

```bash
micromamba create -n unify -c conda-forge python=3.10 pip -y
micromamba activate unify
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r codex-main/requirements-unify.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

git clone https://github.com/huawen-poppy/Unify.git /data/home/dingbaoqing/zhangdeng/soft/Unify
git -C /data/home/dingbaoqing/zhangdeng/soft/Unify checkout 40597f69dcb3b8e8952c20a5c989140f6314d183
python /data/home/dingbaoqing/zhangdeng/soft/Unify/Unify.py --help
```

`requirements-unify.txt` 仅列当前 integration 路径实际需要的包；不需要安装官方通用清单中的 faiss-gpu。
上游代码直接用 pandas Series 索引 torch Tensor，与 pandas 3 不兼容，所以固定 pandas 2.2.3。
若本机没有 louvain 的二进制 wheel，需要 C/C++ 编译器和 CMake；优先使用 Python 3.10 的可用 wheel。

## 从功能描述生成文本嵌入

先整理为 CSV 或 TSV，含 `gene_id` 和 `description` 两列。
同一基因可以多行，分别保存不同的功能描述。缺少描述的基因不应填造通用或虚构功能。
文本建议采用两物种统一语言的原始英文功能描述；两物种使用同一模型。

`unify_text_embeddings.py` 使用本地 Hugging Face 模型目录（不是 ESM 的单个 .pt 文件），
按官方示例对每条描述取最后一个 token 的 hidden state，再对同一基因的多条描述取平均。
示例使用与官方流程相同类别的 LLaMA-2-7B 模型，需自行准备完整的模型、config 和 tokenizer 文件。
本脚本只从本地加载，不提交注释给在线 API，不自动下载模型，也不执行自定义远程模型代码。

```bash
python -m pip install -r codex-main/requirements-unify-text.txt

# 如果尚无本地文本模型，可下载官方示例使用的模型（需预留较大的磁盘空间）。
huggingface-cli download NousResearch/Llama-2-7b-hf --local-dir ./models/Llama-2-7b-hf

bsub -q gpu -gpu "num=1" -n 4 -J gb_text -o gb_text.%J.out -e gb_text.%J.err \
  "$HOME/micromamba/envs/unify/bin/python" codex-main/unify_text_embeddings.py \
  --descriptions gb_unify_annotations/gene_descriptions.tsv \
  --model-dir ./models/Llama-2-7b-hf --device cuda:0 --out gb_text.pt

bsub -q gpu -gpu "num=1" -n 4 -J cl_text -o cl_text.%J.out -e cl_text.%J.err \
  "$HOME/micromamba/envs/unify/bin/python" codex-main/unify_text_embeddings.py \
  --descriptions cl_unify_annotations/gene_descriptions.tsv \
  --model-dir ./models/Llama-2-7b-hf --device cuda:0 --out cl_text.pt
```

## 提交整合

先确认上述两个文本嵌入已经生成。以下读取 X 中的 counts；不要使用 SAMap/SATURN 联合表达对象作输入。
GB 与 CL 的标签列不必同名，准备脚本会生成独立输入副本并保留原标签/样本元数据。

```bash
bsub -q gpu -gpu "num=1" -n 4 -J unify_gb_cl \
  -o unify.%J.out -e unify.%J.err \
  "$HOME/micromamba/envs/unify/bin/python" \
  ./codex-main/unify_integrate.py \
  --species gb=Gerbera_scRNA.all_samples.2026-Sep-15.h5ad \
  --species cl=CL15_rep1_rep2.h5ad \
  --protein gb=gb_ESM2.pt --protein cl=cl_ESM2.pt \
  --text gb=gb_text.pt --text cl=cl_text.pt \
  --label-key gb=cell-type --label-key cl=celltype \
  --unify-dir /data/home/dingbaoqing/zhangdeng/soft/Unify \
  --out unify_gb_cl_rep1_rep2 \
  --hv-genes 8000 --esm-macrogenes 2000 --text-macrogenes 2000 \
  --batch-size 256 --epochs 500 --threads 4 --seed 0 --run
```

如果只有某个物种的 counts 位于 layer，可单独加 `--counts-layer cl=counts` 或 `--counts-layer gb=counts`。
输出目录必须不存在。不加 `--run` 时仅准备、验证输入，保存 `run_unify.sh` 后可在分配 GPU 的作业内运行。
不要根据共享登录节点的 nvidia-smi 输出手动指定其他作业的 GPU；由 LSF 分配可见设备。

脚本先检查标签、计数、嵌入维度与匹配数量，并生成 `input_report.json`。
少于一个 batch 的细胞会被拒绝，避免官方 drop_last 造成零批次训练。
上游会将矩阵转为稠密矩阵，并在评估/推理时将全体 macrogene 表达放入 GPU；实际内存需求取决于细胞数与特征数。
调整 batch-size 并不能消除全量推理的显存占用。

## 批次与结果解释

核查版本的 `--batch_labels` 从命令行接收字符串，但 dataset 按物种字典取列，且 integration
训练数据返回值中没有样本 batch covariate。因此此包装脚本不传入该参数；保留 sample/orig.ident 供结果检查。
这里的跨物种整合不等于另做了 GB sample / CL orig.ident 的 Harmony 校正。
细胞标签参与训练与模型选择，因此基于这些标签的 ARI 不能当作独立验证。

最终输出：`unify_gb_cl_rep1_rep2/training/macrogene_merged_with_unify.h5ad`。
`X` 是 macrogene 特征，`obsm['X_unify']` 是联合低维嵌入；不能直接作为原基因 counts 做差异表达。

```python
import scanpy as sc
adata = sc.read_h5ad('unify_gb_cl_rep1_rep2/training/macrogene_merged_with_unify.h5ad')
sc.pp.neighbors(adata, use_rep='X_unify', random_state=0)
sc.tl.umap(adata, random_state=0)
sc.pl.umap(adata, color=['species', 'unify_cell_type'])
```

真实 GB/CL 数据和功能注释位于用户服务器，尚未在云环境运行；正式 LLaMA 文本嵌入与 GPU 训练也尚未验证。

本次已验证：Python 3.12、PyTorch 2.5.1 CPU 与 requirements-unify.txt 中的依赖可运行官方
猫/虎 toy_data，一轮训练后得到 1788 个细胞、256 维且全部有限的 X_unify。
短跑使用 500 HVG、每个模态 16 macrogenes、batch-size 128；这只验证流程，不代表整合质量。
另有 16 项自动测试通过，涵盖输入检查、物种/模态顺序、文本池化和拟南芥注释转换。
