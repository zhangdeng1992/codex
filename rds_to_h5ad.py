#!/usr/bin/env python3
"""Seurat RDS → 原始 counts H5AD；需要 R 的 SeuratObject 与 Matrix。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def convert_export(folder, output):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.io import mmread
    from scipy import sparse

    genes = (folder / 'genes.txt').read_text(encoding='utf-8').splitlines()
    cells = (folder / 'cells.txt').read_text(encoding='utf-8').splitlines()
    if not genes or not cells or len(set(genes)) != len(genes) or len(set(cells)) != len(cells):
        raise ValueError('基因和细胞 ID 必须非空且唯一')
    counts = sparse.csr_matrix(mmread(folder / 'counts.mtx').T)
    if counts.shape != (len(cells), len(genes)):
        raise ValueError('矩阵形状与细胞、基因 ID 数量不一致')
    if (not np.isfinite(counts.data).all() or (counts.data < 0).any()
            or not np.allclose(counts.data, np.rint(counts.data), rtol=0, atol=1e-6)):
        raise ValueError('必须导出非负原始整数 counts')
    # 不自动解析数字或 NA 字符串，避免破坏条形码、标签和批次 ID。
    table = pd.read_csv(folder / 'metadata.csv', dtype=str, keep_default_na=False)
    ids = table.iloc[:, 0]
    metadata = table.iloc[:, 1:].astype(object).copy()
    metadata.index = pd.Index(ids.to_numpy(dtype=object), dtype=object)
    if not metadata.index.is_unique or set(metadata.index) != set(cells):
        raise ValueError('元数据与 counts 细胞 ID 不匹配')
    metadata = metadata.loc[cells]
    data = ad.AnnData(X=counts, obs=metadata,
                      var=pd.DataFrame(index=pd.Index(genes, dtype=object)))
    data.layers['counts'] = counts.copy()
    data.uns['conversion'] = 'Seurat raw counts; metadata columns stored as strings'
    output.parent.mkdir(parents=True, exist_ok=True)
    # pandas 3 自动推断的分类 categories 可能成为 nullable StringArray。
    # 保持已有 object 字符串列，不让 AnnData 自动转成分类。
    # 同目录临时写入，完整成功后发布；硬链接拒绝覆盖已有输出。
    with tempfile.TemporaryDirectory(prefix='.h5ad-write-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'complete.h5ad'
        data.write_h5ad(staged, convert_strings_to_categoricals=False)
        os.link(staged, output)
    print(f'完成：{output}，{data.n_obs} 个细胞，{data.n_vars} 个基因')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Seurat .rds 文件')
    p.add_argument('--out', type=Path, required=True, help='新的 .h5ad 文件')
    p.add_argument('--assay', default='RNA', help='含原始 counts 的 assay，默认 RNA')
    args = p.parse_args()
    if not args.input.is_file():
        p.error('输入文件不存在')
    if args.out.exists():
        p.error('输出文件已存在，不覆盖已有数据')
    rscript = shutil.which('Rscript')
    if not rscript:
        p.error('未找到 Rscript，请安装 R、SeuratObject 和 Matrix')
    exporter = Path(__file__).resolve().with_name('export_seurat_counts.R')
    with tempfile.TemporaryDirectory(prefix='seurat-export-') as tmp:
        subprocess.run([rscript, str(exporter), str(args.input.resolve()), tmp,
                        args.assay], check=True)
        convert_export(Path(tmp), args.out)


if __name__ == '__main__':
    main()
