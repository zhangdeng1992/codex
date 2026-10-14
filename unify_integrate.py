#!/usr/bin/env python3
"""检查 counts 和两类基因嵌入，准备并运行官方 Unify integration。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from saturn_integrate import parse_pairs, prepare_counts, validate_embeddings


def parse_columns(items, species, name):
    result = {}
    for item in items:
        sid, sep, column = item.partition('=')
        if not sep or sid not in species or not column or sid in result:
            raise ValueError(f'{name}: 需要不重复的已声明物种 ID=列名')
        result[sid] = column
    return result


def build_command(args, species, inputs, proteins, texts, out):
    return [sys.executable, str(Path(args.unify_dir).resolve() / 'Unify.py'),
            '--mode', 'integration',
            '--h5ad_files', *[str(inputs[sid]) for sid in species],
            '--species_labels', *species,
            '--celltype_labels', *['unify_cell_type' for _ in species],
            '--gene_esm_embedding_path', *[str(proteins[sid]) for sid in species],
            '--gene_llama_embedding_path', *[str(texts[sid]) for sid in species],
            '--output_path', str(out / 'training'),
            '--highly_variable_genes', str(args.hv_genes),
            '--num_esm_macrogene', str(args.esm_macrogenes),
            '--num_llama_macrogene', str(args.text_macrogenes),
            '--batch_size', str(args.batch_size), '--train_epochs', str(args.epochs),
            '--seed', str(args.seed)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--species', action='append', required=True, metavar='ID=H5AD')
    p.add_argument('--protein', action='append', required=True, metavar='ID=PT')
    p.add_argument('--text', action='append', required=True, metavar='ID=PT')
    p.add_argument('--label-key', action='append', required=True, metavar='ID=COLUMN')
    p.add_argument('--counts-layer', action='append', default=[], metavar='ID=LAYER',
                   help='按物种指定；未指定者读取 X 中的原始 counts')
    p.add_argument('--unify-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='新的输出目录')
    p.add_argument('--hv-genes', type=int, default=8000)
    p.add_argument('--esm-macrogenes', type=int, default=2000)
    p.add_argument('--text-macrogenes', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    try:
        species = parse_pairs(a.species, 'species')
        proteins = parse_pairs(a.protein, 'protein')
        texts = parse_pairs(a.text, 'text')
        labels = parse_columns(a.label_key, species, 'label-key')
        layers = parse_columns(a.counts_layer, species, 'counts-layer')
        if len(species) < 2 or any(set(x) != set(species) for x in (proteins, texts, labels)):
            raise ValueError('至少两个物种，每个物种须提供蛋白嵌入、功能文本嵌入及标签列')
        if any(n < 1 for n in (a.hv_genes, a.esm_macrogenes, a.text_macrogenes, a.epochs, a.threads)) or a.batch_size < 2:
            raise ValueError('数值参数必须为正；batch-size 至少为 2')
        for sid in species:
            if proteins[sid] == texts[sid]:
                raise ValueError(f'{sid}: 蛋白与功能文本嵌入不能使用同一个文件')
        for name in ('Unify.py', 'models.py', 'dataset.py', 'macrogene_initialize.py', 'min_norm_solvers.py'):
            if not (a.unify_dir / name).is_file():
                raise ValueError(f'Unify 仓库缺少 {name}，请下载完整仓库')
        out = a.out.resolve()
        if out.exists():
            raise ValueError('输出目录已存在，请使用新目录')
        import os
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ[name] = str(a.threads)
        import anndata as ad
        import torch
        prepared, report = {}, {}
        for sid, source in species.items():
            data = prepare_counts(source, labels[sid], layers.get(sid), sid)
            data.obs.rename(columns={'saturn_cell_type': 'unify_cell_type'}, inplace=True)
            # 保留原标签和批次元数据，避免丢失 sample/orig.ident 等信息。
            raw = ad.read_h5ad(source, backed='r')
            try:
                for column in raw.obs:
                    if column not in data.obs:
                        data.obs[column] = raw.obs[column].to_numpy()
            finally:
                raw.file.close()
            report[sid] = {
                'protein': validate_embeddings(proteins[sid], data.var_names, sid + '/protein'),
                'text': validate_embeddings(texts[sid], data.var_names, sid + '/text'),
                'n_cells': data.n_obs, 'label_key': labels[sid], 'counts_layer': layers.get(sid),
            }
            prepared[sid] = data
        for modality in ('protein', 'text'):
            if len({r[modality]['embedding_dim'] for r in report.values()}) != 1:
                raise ValueError(f'{modality}: 各物种嵌入维度不同；须使用同一模型和提取方法')
            # 官方按覆盖最少的物种统一 HVG 上限，然后合并两物种基因做 kmeans。
            available = min(a.hv_genes, min(r[modality]['matched_genes'] for r in report.values())) * len(species)
            requested = a.esm_macrogenes if modality == 'protein' else a.text_macrogenes
            if requested > available:
                raise ValueError(f'{modality}: macrogene 数 {requested} 超过可用基因数上限 {available}')
        if sum(d.n_obs for d in prepared.values()) < a.batch_size:
            raise ValueError('细胞总数少于 batch-size，官方 drop_last 会导致零批次训练')
        out.mkdir(parents=True)
        (out / 'inputs').mkdir()
        inputs = {}
        for sid, data in prepared.items():
            inputs[sid] = out / 'inputs' / f'{sid}.h5ad'
            data.write_h5ad(inputs[sid])
        command = build_command(a, list(species), inputs, proteins, texts, out)
        script = '#!/usr/bin/env bash\nset -euo pipefail\n'
        script += f'export OMP_NUM_THREADS={a.threads} OPENBLAS_NUM_THREADS={a.threads} MKL_NUM_THREADS={a.threads} NUMBA_NUM_THREADS={a.threads}\n'
        script += 'cd ' + shlex.quote(str(a.unify_dir.resolve())) + '\n' + shlex.join(command) + '\n'
        (out / 'run_unify.sh').write_text(script)
        (out / 'input_report.json').write_text(json.dumps(report, indent=2))
        (out / 'run.json').write_text(json.dumps({'command': command, 'batch_correction': 'No within-species batch correction; original metadata preserved'}, indent=2))
        print(f'准备完成：{out}；CUDA available: {torch.cuda.is_available()}')
        print(f'运行：bash {out / "run_unify.sh"}')
        if a.run:
            subprocess.run(command, cwd=a.unify_dir.resolve(), check=True)
            result = ad.read_h5ad(out / 'training' / 'macrogene_merged_with_unify.h5ad')
            import numpy as np
            if result.n_obs != sum(r['n_cells'] for r in report.values()):
                raise ValueError('输出细胞数与输入不一致')
            if 'X_unify' not in result.obsm or not np.isfinite(result.obsm['X_unify']).all():
                raise ValueError('输出缺少有效 X_unify')
            print(f'输出核验通过：{result.n_obs} cells；X_unify={result.obsm["X_unify"].shape}')
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
