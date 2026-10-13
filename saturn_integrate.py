#!/usr/bin/env python3
"""准备跨物种 SATURN 输入；可调用服务器上的官方 SATURN 训练入口。"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


def parse_pairs(items, name):
    result = {}
    for item in items:
        sid, sep, value = item.partition('=')
        if not sep or not sid.isalnum() or sid in result or not value:
            raise ValueError(f'{name}: 需要不重复的物种 ID=文件路径（ID 仅含字母/数字）')
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError(f'文件不存在：{path}')
        result[sid] = path
    return result


def prepare_counts(source, label_key, counts_layer, sid):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    data = ad.read_h5ad(source)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError(f'{sid}: 细胞/基因 ID 必须唯一')
    if not data.n_obs or not data.n_vars:
        raise ValueError(f'{sid}: 空数据集')
    # 官方 SATURN 按小写基因 ID 匹配嵌入。
    if not pd.Index(data.var_names.str.lower()).is_unique:
        raise ValueError(f'{sid}: 基因 ID 转为小写后有重复')
    if label_key not in data.obs:
        raise ValueError(f'{sid}: 缺少细胞类型列 {label_key!r}')
    labels = data.obs[label_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq('').any():
        raise ValueError(f'{sid}: 细胞类型标签含缺失值')
    if counts_layer and counts_layer not in data.layers:
        raise ValueError(f'{sid}: 缺少 counts layer {counts_layer!r}')
    counts = data.layers[counts_layer] if counts_layer else data.X
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
        raise ValueError(f'{sid}: 必须使用非负原始整数 counts')
    if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
        raise ValueError(f'{sid}: 请先去除零 counts 细胞')
    obs = pd.DataFrame(index=[f'{sid}:{cell}' for cell in data.obs_names])
    obs['saturn_cell_type'] = pd.Categorical(labels.astype(str).to_numpy())
    obs['source_cell_id'] = data.obs_names.to_numpy(dtype=object)
    obs['source_species'] = sid
    return ad.AnnData(X=sparse.csr_matrix(counts), obs=obs,
                      var=pd.DataFrame(index=data.var_names.to_numpy(dtype=object)))


def validate_embeddings(path, genes, sid):
    import numpy as np
    import torch
    # 只加载 tensor 字典，不执行任意 pickle 对象。
    embeddings = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError(f'{sid}: 嵌入必须为非空 gene_id -> Tensor 字典')
    lowered = set()
    dim = None
    for gene, vector in embeddings.items():
        if not isinstance(gene, str) or gene.lower() in lowered:
            raise ValueError(f'{sid}: 嵌入基因 ID 无效或小写后重复')
        lowered.add(gene.lower())
        if (not isinstance(vector, torch.Tensor) or vector.ndim != 1
                or not vector.is_floating_point() or not torch.isfinite(vector).all()):
            raise ValueError(f'{sid}/{gene}: 嵌入必须为有限的一维浮点 Tensor')
        if dim is None:
            dim = vector.numel()
        if not dim or vector.numel() != dim:
            raise ValueError(f'{sid}: 嵌入维度不一致')
    matched = sum(g.lower() in lowered for g in genes)
    if not matched:
        raise ValueError(f'{sid}: 表达基因与蛋白嵌入完全不匹配')
    print(f'{sid}: {matched}/{len(genes)} 个表达基因匹配嵌入，维度 {dim}')
    return {'matched_genes': int(matched), 'total_genes': len(genes), 'embedding_dim': dim}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--species', action='append', required=True, metavar='ID=H5AD')
    p.add_argument('--embedding', action='append', required=True, metavar='ID=PT')
    p.add_argument('--label-key', required=True, help='各输入共有的细胞类型标签列')
    p.add_argument('--counts-layer', help='不指定时使用 X 中的原始 counts')
    p.add_argument('--out', type=Path, required=True, help='新的输出目录')
    p.add_argument('--saturn-dir', type=Path, required=True, help='官方 SATURN checkout')
    p.add_argument('--run', action='store_true', help='准备成功后立即训练；默认只准备输入')
    p.add_argument('--device-num', type=int, default=0)
    p.add_argument('--hv-genes', type=int, default=8000)
    p.add_argument('--num-macrogenes', type=int, default=2000)
    p.add_argument('--pretrain-epochs', type=int, default=200)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    try:
        species = parse_pairs(a.species, 'species')
        embeddings = parse_pairs(a.embedding, 'embedding')
        if len(species) < 2 or set(species) != set(embeddings):
            raise ValueError('至少两个物种，且每个物种必须提供一个嵌入文件')
        if any(n < 1 for n in (a.hv_genes, a.num_macrogenes, a.pretrain_epochs, a.epochs, a.batch_size)):
            raise ValueError('基因数、macrogene 数、epoch 和 batch size 必须为正数')
        if a.device_num < 0:
            raise ValueError('GPU 编号必须非负')
        out = a.out.resolve()
        saturn = a.saturn_dir.resolve()
        if out.exists():
            raise ValueError('输出目录已存在，请使用新目录')
        if not (saturn / 'train-saturn.py').is_file():
            raise ValueError('saturn-dir 下没有官方 train-saturn.py')
        import torch
        if a.run and (not torch.cuda.is_available() or a.device_num >= torch.cuda.device_count()):
            raise ValueError('官方训练入口需要可用 NVIDIA GPU；请检查 PyTorch/CUDA 和 GPU 编号')
        prepared, report = {}, {}
        for sid, source in species.items():
            prepared[sid] = prepare_counts(source, a.label_key, a.counts_layer, sid)
            report[sid] = validate_embeddings(embeddings[sid], prepared[sid].var_names, sid)
        if len({r['embedding_dim'] for r in report.values()}) != 1:
            raise ValueError('不同物种的蛋白嵌入维度不一致；必须使用相同蛋白语言模型')
        out.mkdir(parents=True)
        inputs = out / 'inputs'
        inputs.mkdir()
        with (out / 'in_data.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['species', 'path', 'embedding_path'])
            writer.writeheader()
            for sid, data in prepared.items():
                destination = inputs / f'{sid}.h5ad'
                data.write_h5ad(destination)
                writer.writerow({'species': sid, 'path': str(destination),
                                 'embedding_path': str(embeddings[sid])})
        (out / 'input_report.json').write_text(json.dumps(report, indent=2))
        command = [sys.executable, str(saturn / 'train-saturn.py'),
                   '--in_data', str(out / 'in_data.csv'),
                   '--in_label_col', 'saturn_cell_type', '--ref_label_col', 'saturn_cell_type',
                   '--device', 'cuda', '--device_num', str(a.device_num),
                   '--work_dir', str(out / 'training') + '/',
                   '--hv_genes', str(a.hv_genes), '--num_macrogenes', str(a.num_macrogenes),
                   '--pretrain_epochs', str(a.pretrain_epochs), '--epochs', str(a.epochs),
                   '--pretrain_batch_size', str(a.batch_size), '--batch_size', str(a.batch_size),
                   '--seed', str(a.seed)]
        script = '#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(saturn)) + '\n' + shlex.join(command) + '\n'
        (out / 'run_saturn.sh').write_text(script)
        print(f'输入准备完成：{out}\n运行：bash {out / "run_saturn.sh"}')
        if a.run:
            subprocess.run(command, cwd=saturn, check=True)
            print(f'训练输出：{out / "training" / "saturn_results"}')
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
