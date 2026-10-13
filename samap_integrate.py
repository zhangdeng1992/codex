#!/usr/bin/env python3
"""使用 SAMap 3.x 对齐多个物种的单细胞数据。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import math


def write_h5ad_compatible(data, destination):
    """保留分类和数值类型，将 pandas 扩展字符串转换为传统 object。"""
    import pandas as pd
    for frame in (data.obs, data.var):
        frame.index = pd.Index(frame.index.to_numpy(dtype=object),
                               dtype=object, name=frame.index.name)
        frame.columns = pd.Index(frame.columns.to_numpy(dtype=object), dtype=object)
        for column in frame.columns:
            series = frame[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categories = series.cat.categories
                if isinstance(categories.dtype, pd.StringDtype):
                    frame[column] = pd.Categorical.from_codes(
                        series.cat.codes.to_numpy(),
                        categories=pd.Index(categories.to_numpy(dtype=object), dtype=object),
                        ordered=series.cat.ordered)
            elif isinstance(series.dtype, pd.StringDtype):
                frame[column] = series.fillna('').astype(object)
    data.write_h5ad(destination, convert_strings_to_categoricals=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--species', action='append', required=True, metavar='ID=FILE',
                   help='重复指定，例如 hs=human.h5ad、mm=mouse.h5ad')
    p.add_argument('--maps', type=Path, required=True, help='SAMap 双向 BLAST 表目录')
    p.add_argument('--out', type=Path, required=True, help='新的输出目录，不覆盖已有目录')
    p.add_argument('--counts-layer', help='原始 counts 所在 layer；默认使用 X')
    p.add_argument('--label-key', default='cell_type', help='所有物种的细胞类型列名')
    p.add_argument('--batch-key', action='append', default=[], metavar='ID=COLUMN',
                   help='物种内 Harmony 批次列，例如 gb=orig.ident；可重复指定')
    p.add_argument('--resolution', '--resolutions', action='append', default=[],
                   metavar='ID=VALUE', help='各物种 Leiden resolution，例如 gb=0.3；可重复指定')
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.iterations < 1 or a.threads < 1:
        p.error('iterations 和 threads 必须为正数')
    species = {}
    for item in a.species:
        sid, sep, filename = item.partition('=')
        if not sep or not re.fullmatch(r'[a-z]{2}', sid):
            p.error('species 格式必须为两个小写字母的物种 ID=文件路径')
        if sid in species:
            p.error(f'重复物种 ID: {sid}')
        path = Path(filename).resolve()
        if not path.is_file():
            p.error(f'输入文件不存在: {path}')
        species[sid] = path
    if len(species) < 2:
        p.error('至少需要两个物种')
    batch_keys = {}
    for item in a.batch_key:
        sid, sep, column = item.partition('=')
        if not sep or sid not in species or not column or sid in batch_keys:
            p.error('batch-key 必须为已声明且不重复的物种 ID=批次列名')
        batch_keys[sid] = column
    a.batch_keys = batch_keys
    a.resolutions = {}
    for item in a.resolution:
        sid, sep, value = item.partition('=')
        if not sep or sid not in species or sid in a.resolutions:
            p.error('resolution 必须为已声明且不重复的物种 ID=数值')
        try:
            number = float(value)
        except ValueError:
            p.error('resolution 必须为有限正数')
        if not math.isfinite(number) or number <= 0:
            p.error('resolution 必须为有限正数')
        a.resolutions[sid] = number
    for x, y in itertools.combinations(species, 2):
        folder = next((a.maps / pair for pair in (x+y, y+x)
                       if (a.maps / pair).is_dir()), None)
        if folder is None:
            p.error(f'缺少同源表目录: {a.maps / (x+y)}')
        for source, target in ((x, y), (y, x)):
            table = folder / f'{source}_to_{target}.txt'
            if not table.is_file() or table.stat().st_size == 0:
                p.error(f'缺少或为空的 BLAST 表: {table}')
    if a.out.exists():
        p.error(f'输出目录已存在，请使用新目录: {a.out}')
    return a, species


def main():
    args, species = parse_args()
    # 必须在导入数值库之前设置，避免线程过度订阅。
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    import anndata as ad
    import numpy as np
    from scipy import sparse
    from samap import SAMAP
    from samap.analysis import get_mapping_scores
    from samap.sam import SAM

    np.random.seed(args.seed)
    args.out.mkdir(parents=True)
    prepared = args.out / 'inputs'
    prepared.mkdir()
    inputs = {}
    keys = {}
    for sid, filename in species.items():
        source = ad.read_h5ad(filename)
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError(f'{sid}: 细胞和基因 ID 必须唯一；请先明确处理重复 ID')
        if source.n_obs < 50 or source.n_vars < 100:
            raise ValueError(f'{sid}: 数据过小，至少需要 50 个细胞和 100 个基因')
        if args.counts_layer:
            if args.counts_layer not in source.layers:
                raise ValueError(f'{sid}: 缺少 layer {args.counts_layer!r}')
            counts = source.layers[args.counts_layer]
        else:
            counts = source.X
        values = counts.data if sparse.issparse(counts) else np.asarray(counts)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
            raise ValueError(f'{sid}: 输入必须为非负原始整数 counts；'
                             '若 counts 位于 layer，请指定 --counts-layer')
        if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
            raise ValueError(f'{sid}: 存在零 counts 细胞，请先做 QC')
        if args.label_key in source.obs:
            if (source.obs[args.label_key].isna().any()
                    or source.obs[args.label_key].astype(str).str.strip().eq('').any()):
                raise ValueError(f'{sid}: {args.label_key} 存在缺失值')
            keys[sid] = args.label_key
        else:
            keys[sid] = 'leiden_clusters'
            print(f'{sid}: 未找到 {args.label_key}，改用 SAM 自动 Leiden 聚类')
        # 只复制 counts 和元数据，避免沿用旧 PCA、邻接图和归一化结果。
        data = ad.AnnData(X=sparse.csr_matrix(counts), obs=source.obs.copy(),
                          var=source.var.copy())
        destination = prepared / f'{sid}.h5ad'
        write_h5ad_compatible(data, destination)
        batch_key = args.batch_keys.get(sid)
        if batch_key:
            if (batch_key not in data.obs or data.obs[batch_key].isna().any()
                    or data.obs[batch_key].astype(str).str.strip().eq('').any()):
                raise ValueError(f'{sid}: 批次列 {batch_key!r} 不存在或有缺失值')
            if data.obs[batch_key].nunique() < 2:
                raise ValueError(f'{sid}: 批次列只有一个批次，请省略该物种的 --batch-key')
        # 参考作者 notebook：各物种独立 SAM，再进行跨物种 SAMap。
        sam = SAM()
        sam.load_data(str(destination.resolve()))
        sam.preprocess_data()
        sam.run(batch_key=batch_key, seed=args.seed)
        inputs[sid] = sam

    from samap._constants import DEFAULT_LEIDEN_RESOLUTION
    resolutions = {sid: args.resolutions.get(sid, DEFAULT_LEIDEN_RESOLUTION)
                   for sid in species}
    for sid in args.resolutions:
        if keys[sid] != 'leiden_clusters':
            print(f'{sid}: 使用已有标签 {keys[sid]!r}，resolution 不会合并这些标签')
    sm = SAMAP(sams=inputs, f_maps=str(args.maps.resolve()) + os.sep,
               keys=keys, resolutions=resolutions, save_processed=False, backend='cpu')
    for sid, sam in sm.sams.items():
        print(f'{sid}: 映射标签 {keys[sid]}，共 {sam.adata.obs[keys[sid]].nunique()} 个群')
    for sid, genes in sm.gns_dict.items():
        if len(genes) == 0:
            raise ValueError(f'{sid}: 没有基因匹配 BLAST 图，请检查基因 ID 和过滤')
        print(f'{sid}: {len(genes)} 个基因匹配同源图')
    sm.run(n_iterations=args.iterations, ncpus=args.threads, umap=True, pairwise=True)
    ranking, scores = get_mapping_scores(sm, keys=keys)
    ranking.to_csv(args.out / 'mapping_ranking.csv')
    scores.to_csv(args.out / 'mapping_scores.csv')
    merged = sm.samap.adata
    write_h5ad_compatible(merged, args.out / 'samap_joint.h5ad')
    merged.obs.to_csv(args.out / 'cells.csv')
    if 'X_umap' in merged.obsm:
        import pandas as pd
        pd.DataFrame(merged.obsm['X_umap'], index=merged.obs_names,
                     columns=['UMAP1', 'UMAP2']).to_csv(args.out / 'umap.csv')
    for sid, sam in sm.sams.items():
        write_h5ad_compatible(sam.adata, args.out / f'{sid}_processed.h5ad')
    import importlib.metadata
    (args.out / 'run.json').write_text(json.dumps({
        'species': {k: str(v) for k, v in species.items()},
        'maps': str(args.maps.resolve()), 'annotation_keys': keys,
        'counts_layer': args.counts_layer, 'iterations': args.iterations,
        'batch_keys': args.batch_keys, 'pairwise': True,
        'resolutions': resolutions,
        'threads': args.threads, 'seed': args.seed,
        'sc-samap_version': importlib.metadata.version('sc-samap'),
    }, indent=2), encoding='utf-8')
    print(f'完成：{args.out.resolve()}')


if __name__ == '__main__':
    main()
