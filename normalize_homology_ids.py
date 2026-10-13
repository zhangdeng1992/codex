#!/usr/bin/env python3
"""将双向蛋白同源表 ID 转换为 H5AD 基因 ID，不修改源文件。"""
import argparse
import itertools
from pathlib import Path
import re
import tempfile
import os


def convert_table(source, destination, query_genes, target_genes, query_pattern, target_pattern):
    def resolve(identifier, genes, pattern):
        if identifier in genes:
            return identifier
        candidate = pattern.sub('', identifier)
        return candidate if candidate in genes else None

    best = {}
    total = 0
    retained = 0
    with source.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 12:
                raise ValueError(f'{source}:{number}: 需要标准 12 列 outfmt 6')
            total += 1
            query = resolve(fields[0], query_genes, query_pattern)
            target = resolve(fields[1], target_genes, target_pattern)
            if query is None or target is None:
                continue
            retained += 1
            import math
            score, evalue = float(fields[11]), float(fields[10])
            if not math.isfinite(score) or not math.isfinite(evalue) or evalue < 0:
                raise ValueError(f'{source}:{number}: 无效 bit score / E-value')
            fields[0], fields[1] = query, target
            rank = (score, -evalue)
            key = (query, target)
            if key not in best or rank > best[key][0]:
                best[key] = (rank, fields)
    if not best:
        raise ValueError(f'{source}: 转换后没有匹配的基因对')
    with destination.open('x') as handle:
        for _, fields in best.values():
            handle.write('\t'.join(fields) + '\n')
    print(f'{source.name}: 输入 {total} 行，ID 可匹配 {retained} 行，'
          f'保留 {len(best)} 个基因对，query {len({q for q,t in best})} 个基因，'
          f'target {len({t for q,t in best})} 个基因')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--species', action='append', required=True, metavar='ID=H5AD')
    parser.add_argument('--strip-suffix', action='append', required=True, metavar='ID=REGEX')
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths, patterns = {}, {}
    for item in args.species:
        sid, sep, filename = item.partition('=')
        if not sep or not re.fullmatch('[a-z]{2}', sid) or sid in paths:
            parser.error('species 必须为不重复的两字母 ID=H5AD')
        paths[sid] = Path(filename)
    if len(paths) < 2:
        parser.error('至少需要两个物种')
    for item in args.strip_suffix:
        sid, sep, pattern = item.partition('=')
        if not sep or sid not in paths or sid in patterns or not pattern.endswith('$'):
            parser.error('strip-suffix 必须为不重复的 ID=末尾锚定正则表达式（以 $ 结尾）')
        patterns[sid] = re.compile(pattern)
    if set(patterns) != set(paths):
        parser.error('每个物种必须提供 strip-suffix')
    if args.out.exists():
        parser.error('输出目录已存在，请使用新目录')
    import anndata as ad
    genes = {}
    for sid, path in paths.items():
        data = ad.read_h5ad(path, backed='r')
        try:
            if not data.var_names.is_unique:
                raise ValueError(f'{sid}: H5AD 基因 ID 不唯一')
            genes[sid] = set(map(str, data.var_names))
        finally:
            data.file.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 所有方向成功后才发布输出目录；失败时清理临时文件。
    with tempfile.TemporaryDirectory(prefix='.homology-', dir=args.out.parent) as tmp:
        staged = Path(tmp) / 'maps'
        staged.mkdir()
        for x, y in itertools.combinations(paths, 2):
            folder = next((args.maps / pair for pair in (x+y, y+x)
                           if (args.maps / pair).is_dir()), None)
            if folder is None:
                raise FileNotFoundError(f'缺少 {x}/{y} 同源表目录')
            output = staged / (x+y)
            output.mkdir()
            for q, t in ((x,y), (y,x)):
                name = f'{q}_to_{t}.txt'
                convert_table(folder / name, output / name, genes[q], genes[t],
                              patterns[q], patterns[t])
        if args.out.exists():
            raise FileExistsError(args.out)
        os.rename(staged, args.out)
    print(f'完成：{args.out.resolve()}')


if __name__ == '__main__':
    main()
