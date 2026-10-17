#!/usr/bin/env python3
"""准备 SATURN H5AD：统一标签和基因 ID，验证原始 counts 与蛋白嵌入。"""
import argparse
import json
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from saturn_integrate import validate_embeddings


def convert_gene_ids(names, mode):
    original = pd.Index(names)
    converted = original
    if mode == 'gerbera':
        converted = (original.str.strip()
                     .str.replace(r'^Ger-', 'Ger_', regex=True)
                     .str.replace(r'^(Ger_[0-9]+)\.m[0-9]+$', r'\1', regex=True))
    elif mode != 'none':
        raise ValueError(f'未知 ID 转换模式：{mode}')
    if (converted.str.strip() == '').any():
        raise ValueError('基因 ID 不能为空')
    duplicated = converted.str.lower().duplicated(keep=False)
    if duplicated.any():
        examples = list(zip(original[duplicated][:10], converted[duplicated][:10]))
        raise ValueError(f'基因 ID 转换/小写后重复，不自动合并转录本或添加后缀：{examples}')
    return converted


def prepare_h5ad(source, embedding, output, label_key='cell-type',
                 id_mode='gerbera', counts_layer='auto'):
    source, embedding, output = map(lambda p: Path(p).resolve(), (source, embedding, output))
    mapping_path = output.with_suffix('.gene_id_map.csv')
    report_path = output.with_suffix('.report.json')
    destinations = (output, mapping_path, report_path)
    for path in destinations:
        if path.exists():
            raise ValueError(f'输出已存在，拒绝覆盖：{path}')
    for path in (source, embedding):
        if not path.is_file():
            raise ValueError(f'输入文件不存在：{path}')

    data = ad.read_h5ad(source)
    if not data.n_obs or not data.n_vars:
        raise ValueError('数据集为空')
    if not data.obs_names.is_unique:
        raise ValueError('细胞 ID 有重复')
    if label_key not in data.obs:
        raise ValueError(f'缺少注释列 {label_key!r}，现有列：{data.obs.columns.tolist()}')
    labels = data.obs[label_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq('').any():
        raise ValueError('细胞类型标签有缺失或空字符串')

    selected = ('counts' if 'counts' in data.layers else 'X') if counts_layer == 'auto' else counts_layer
    if selected != 'X' and selected not in data.layers:
        raise ValueError(f'缺少 counts layer：{selected}')
    matrix = data.X if selected == 'X' else data.layers[selected]
    if matrix is None:
        raise ValueError('选中的表达矩阵为空')
    counts = sparse.csr_matrix(matrix, copy=True)
    counts.sum_duplicates()
    values = counts.data
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
        raise ValueError('必须使用非负原始整数 counts；不会对标准化表达取整')
    if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
        raise ValueError('存在零 counts 细胞，请先检查')

    original = data.var_names.copy()
    converted = convert_gene_ids(original, id_mode)
    coverage = validate_embeddings(embedding, converted, 'input')
    provenance_key = 'saturn_original_gene_id'
    if provenance_key in data.var:
        raise ValueError(f'已有 {provenance_key} 列，请使用转换前的输入，避免覆盖来源记录')
    data.var[provenance_key] = original.to_numpy(dtype=object)
    data.var_names = converted
    data.obs['cell_type'] = labels.copy()
    data.X = counts

    mapping = pd.DataFrame({'original_id': original, 'saturn_id': converted})
    report = {
        'source': str(source), 'output': str(output), 'embedding': str(embedding),
        'n_cells': data.n_obs, 'n_genes': data.n_vars,
        'label_source': label_key, 'label_target': 'cell_type',
        'counts_source': 'X' if selected == 'X' else f'layers[{selected!r}]',
        'id_mode': id_mode, 'changed_gene_ids': int((original != converted).sum()),
        'embedding_coverage': coverage,
        'cell_type_counts': {str(k): int(v) for k, v in data.obs['cell_type'].value_counts().items()},
    }
    # 完成所有验证后写入临时目录，避免写入失败留下半个 H5AD。
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.saturn_prepare_', dir=output.parent) as tmp:
        staged = [Path(tmp) / path.name for path in destinations]
        data.write_h5ad(staged[0], compression='gzip')
        mapping.to_csv(staged[1], index=False)
        staged[2].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for path in destinations:
            if path.exists():
                raise ValueError(f'输出已存在，拒绝覆盖：{path}')
        for temporary, destination in zip(staged, destinations):
            temporary.rename(destination)
    print(f'已保存：{output}\nID 对照：{mapping_path}\n检查报告：{report_path}')
    print(f'细胞 {data.n_obs}；基因 {data.n_vars}；转换 ID {report["changed_gene_ids"]}；counts 来源 {report["counts_source"]}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='原始 H5AD')
    parser.add_argument('--embedding', type=Path, required=True, help='gene_id -> Tensor 的 PT 文件')
    parser.add_argument('--out', type=Path, required=True, help='新的输出 H5AD，不覆盖已有文件')
    parser.add_argument('--label-key', default='cell-type', help='来源注释列，默认 cell-type；输出统一为 cell_type')
    parser.add_argument('--id-mode', choices=['gerbera', 'none'], default='gerbera',
                        help='gerbera 修复 Ger- 前缀并去除 Ger_数字 的 .m数字 后缀；none 保留 ID')
    parser.add_argument('--counts-layer', default='auto',
                        help='auto 优先 counts layer，否则 X；也可显式指定 X 或 layer 名称')
    args = parser.parse_args()
    try:
        prepare_h5ad(args.input, args.embedding, args.out, args.label_key,
                     args.id_mode, args.counts_layer)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
