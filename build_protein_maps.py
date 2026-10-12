#!/usr/bin/env python3
"""参考 CrossSpeciesPlantShootAtlas，生成所有物种对的双向 BLASTP 表。"""
import argparse
import itertools
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protein', action='append', required=True, metavar='ID=FASTA')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error('threads 必须为正数')
    proteins = {}
    for item in a.protein:
        sid, sep, filename = item.partition('=')
        if not sep or not re.fullmatch(r'[a-z]{2}', sid) or sid in proteins:
            p.error('protein 格式为不重复的两个小写字母 ID=FASTA')
        file = Path(filename).resolve()
        if not file.is_file():
            p.error(f'文件不存在: {file}')
        ids = set()
        with file.open() as handle:
            for line in handle:
                if line.startswith('>'):
                    fields = line[1:].split()
                    if not fields or fields[0] in ids:
                        p.error(f'{file}: 空的或重复的 FASTA ID')
                    ids.add(fields[0])
        if not ids:
            p.error(f'{file}: 没有 FASTA 序列 ID')
        proteins[sid] = file
    if len(proteins) < 2:
        p.error('至少需要两个物种')
    if a.out.exists():
        p.error('输出目录已存在，请选择新目录')
    for tool in ('makeblastdb', 'blastp'):
        if not shutil.which(tool):
            p.error(f'未安装 {tool}，请先安装 NCBI BLAST+')
    a.out.mkdir(parents=True)
    dbdir = a.out / '_blastdb'
    dbdir.mkdir()
    for sid, file in proteins.items():
        subprocess.run(['makeblastdb', '-in', str(file), '-dbtype', 'prot',
                        '-out', str(dbdir / sid)], check=True)
    # 顺序运行各方向，控制总 CPU 用量；任意 BLAST 失败即停止。
    for x, y in itertools.combinations(proteins, 2):
        folder = a.out / (x+y)
        folder.mkdir()
        for query, target in ((x, y), (y, x)):
            table = folder / f'{query}_to_{target}.txt'
            subprocess.run(['blastp', '-query', str(proteins[query]),
                            '-db', str(dbdir / target), '-outfmt', '6',
                            '-out', str(table), '-num_threads', str(a.threads),
                            '-max_hsps', '1', '-evalue', '1e-6'], check=True)
            if table.stat().st_size == 0:
                raise RuntimeError(f'没有 BLASTP 命中: {query} -> {target}')
    print(f'完成：{a.out.resolve()}')


if __name__ == '__main__':
    main()
