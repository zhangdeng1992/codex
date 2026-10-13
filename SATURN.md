# 使用 SATURN 整合 Gerbera 和 CL15

SATURN 将不同物种的细胞映射到共享嵌入空间，同时学习 macrogene 表达。
联合 H5AD 的 `X` 是细胞嵌入，`obsm['macrogenes']` 是 macrogene 表达；
它不是把两种物种的原始基因 counts 简单拼接，也不是统一同源基因的批次校正 counts。

本仓库新增：

- `saturn_protein_embeddings.py`：ESM-2 蛋白嵌入，可显式把蛋白 ID 映射到表达基因 ID。
- `saturn_integrate.py`：检查输入并生成 counts 副本、CSV 和训练脚本，`--run` 时调用官方训练。
- `requirements-saturn.txt`：独立环境的起始依赖约束，避免影响原有 SAMap 环境。

## 1. 服务器环境

需要 NVIDIA GPU 和兼容的 PyTorch/CUDA。先运行 `nvidia-smi` 检查驱动。
下面以 Python 3.10、PyTorch 2.2.2 / CUDA 12.1 为例；驱动不兼容时，按
[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)选择对应版本。
这些依赖约束是适配建议，尚未在本云环境完成 GPU 训练验证。

```bash
conda create -n saturn python=3.10 -y
conda activate saturn
# 请在本仓库根目录执行。
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-saturn.txt
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# tools 和结果均放在仓库外；无需修改官方代码。
git clone https://github.com/snap-stanford/SATURN.git ../SATURN
# 本次核查的官方版本：
git -C ../SATURN checkout 6906abf23728aa8ebf400cedf4ce38e4821a0a49
python ../SATURN/train-saturn.py --help
```

FAISS 这里只用于 CPU 近邻计算，模型训练仍使用 GPU。官方入口无条件调用
`torch.cuda.set_device`，所以不能仅靠 `--device cpu` 在无 GPU 机器上训练。

## 2. 准备输入

每个物种需要：

1. QC 后的 H5AD，`X` 或指定 layer 是非负整数原始 counts。
2. 一个细胞类型标签列，如 `cell_type`。两文件须使用同一列名；标签不要求跨物种一一相同。
3. `gene_id -> 一维浮点 Tensor` 的 `.pt` 嵌入字典。两物种须使用相同蛋白语言模型，维度一致。

基因 ID 必须唯一，且与嵌入键对应。不要直接用 `.m1` / `.5` 的字符串裁剪规则猜测
蛋白与表达基因对应关系。SATURN 会按小写 ID 匹配，因此脚本也会检查小写后的重复。

如果只有蛋白 FASTA，提供显式注释映射 CSV，例如：

```csv
protein_id,gene_id
Ger_000002.m1,Ger_000002
```

CL15 使用自己的对应表。FASTA 中每条序列都须有映射；建议先根据注释选定所需蛋白。
多个蛋白对应一个基因时，本脚本对它们的嵌入取均值。
如果 FASTA ID 本来就是表达基因 ID，可省略 `--gene-map`。

```bash
export TORCH_HOME="$HOME/.cache/torch"
python saturn_protein_embeddings.py \
  --fasta /path/to/Gerbera.pep.fa \
  --gene-map /path/to/Gerbera_protein_gene.csv \
  --out /path/to/gb_ESM2.pt --device cuda:0

python saturn_protein_embeddings.py \
  --fasta /path/to/CL15.pep.fa \
  --gene-map /path/to/CL15_protein_gene.csv \
  --out /path/to/cl_ESM2.pt --device cuda:0
```

首次生成会下载 ESM-2 权重，需要访问官方模型下载服务及足够的显存/磁盘。
默认模型是 `esm2_t33_650M_UR50D`，显存不足可在两个命令中同时换成
`--model esm2_t30_150M_UR50D`。脚本不设置人为长度上限，保留全部残基；
如果模型有固定位置编码容量（如 ESM-1b），超长蛋白会自动按该容量分段推理，
排除特殊 token 后按残基数加权合并嵌入。分段失去跨片段上下文，属于近似处理。
无固定位置容量的 ESM-2 使用完整序列。`--max-length` 和 `--long-sequence` 参数已移除。
长序列的实际可运行长度取决于模型实现及显存，显存需求会随序列长度增长。
两个物种应使用相同模型。已有嵌入可直接跳过本步骤。

已有 ESM-1b 权重时，可用 `--model-file /absolute/path/esm1b_t33_650M_UR50S.pt`
直接加载本地文件，不会下载模型或 contact-regression 辅助权重；嵌入不需要接触预测头。
文件需保留官方模型名称，并来自可信来源；官方 checkpoint 加载会反序列化 Python 对象。
若提示缺少 regression weights，仅影响接触预测，本脚本不请求接触预测。
两个物种均须使用同一 ESM-1b 权重，不能一个用 ESM-1b、另一个用 ESM-2。
ESM-1b 的位置编码容量为 1024 tokens，含 BOS/EOS，因此每段最多 1022 个残基。
例如 1061 残基自动分为 1022 + 39 两段，不删除或截断序列。

## 3. 准备并运行 SATURN

替换所有 `/path/to/` 路径及标签列名。输出目录必须尚不存在。

```bash
python saturn_integrate.py \
  --species gb=/path/to/Gerbera_scRNA.all_samples.2026-Sep-15.h5ad \
  --species cl=/path/to/CL15_snRNA.h5ad \
  --embedding gb=/path/to/gb_ESM2.pt \
  --embedding cl=/path/to/cl_ESM2.pt \
  --counts-layer counts --label-key cell_type \
  --saturn-dir ../SATURN --out ../saturn_gb_cl \
  --hv-genes 8000 --num-macrogenes 2000 \
  --batch-size 256 --device-num 0 --run
```

如果 counts 在 `X`，省略 `--counts-layer counts`。没有细胞类型标签时，先在各物种内部
独立聚类并写入同名标签列，再运行；本脚本不会伪造标签。
不加 `--run` 时，只准备输入和 `run_saturn.sh`，之后执行：

```bash
bash ../saturn_gb_cl/run_saturn.sh
```

脚本复制 counts 和必要标签，不覆盖原 H5AD，并给细胞 ID 添加物种前缀。
`input_report.json` 报告每个物种的嵌入覆盖；零覆盖会失败，低覆盖则需要先核对注释。
SATURN 会过滤掉没有嵌入的基因。小数据集应降低 `--hv-genes` / `--num-macrogenes`。

首次可用 `--pretrain-epochs 1 --epochs 1` 做短跑，只检查流程；正式分析沿用默认
200 个预训练 epoch 和 50 个 metric-learning epoch。每次重跑使用新输出目录。

## 4. 输出与验证

正式结果位于 `saturn_gb_cl/training/saturn_results/`：

- 最终 `.h5ad`：联合细胞嵌入、物种与标签信息，以及 `obsm['macrogenes']`。
- `*_pretrain.h5ad`：预训练结果。
- `*_genes_to_macrogenes.pkl`：基因到 macrogene 权重。
- 其他训练日志及标签映射文件。

检查训练退出码为零，再读取最终 H5AD，核对细胞数、`obs['species']`、嵌入维度、
有限数值和 `obsm['macrogenes']`。不要把只完成输入准备或一轮短跑当作正式整合成功。
原标签仍存在源文件及准备副本中；官方输出的 `labels2` 对带下划线标签有拆分行为，
需要按带物种前缀的细胞 ID 从准备副本回填原标签，不能盲目依赖该列。

本云环境只验证输入准备和错误处理；没有 GPU，也没有真实 Gerbera / CL15 数据。
实际 ESM 权重推理、官方依赖安装及 SATURN GPU 训练须在服务器验证。
