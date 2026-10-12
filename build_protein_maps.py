#!/usr/bin/env python3
"""使用 DIAMOND 或 BLASTP，生成 SAMap 所需的双向蛋白同源表。"""
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
    p.add_argument('--aligner', choices=['diamond', 'blast'], default='diamond',
                   help='默认 diamond；blast 使用论文原有的 BLASTP 路线')
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
    tools = ('diamond',) if a.aligner == 'diamond' else ('makeblastdb', 'blastp')
    for tool in tools:
        if not shutil.which(tool):
            p.error(f'未安装 {tool}，请先安装所选比对工具并加入 PATH')
    a.out.mkdir(parents=True)
    dbdir = a.out / '_databases'
    dbdir.mkdir()
    for sid, file in proteins.items():
        if a.aligner == 'diamond':
            command = ['diamond', 'makedb', '--in', str(file),
                       '--db', str(dbdir / sid), '--threads', str(a.threads)]
        else:
            command = ['makeblastdb', '-in', str(file), '-dbtype', 'prot',
                       '-out', str(dbdir / sid)]
        subprocess.run(command, check=True)
    # 顺序运行各方向，控制总 CPU 用量；任意比对失败即停止。
    for x, y in itertools.combinations(proteins, 2):
        folder = a.out / (x+y)
        folder.mkdir()
        for query, target in ((x, y), (y, x)):
            table = folder / f'{query}_to_{target}.txt'
            if a.aligner == 'diamond':
                command = ['diamond', 'blastp', '--query', str(proteins[query]),
                           '--db', str(dbdir / target), '--outfmt', '6',
                           '--out', str(table), '--threads', str(a.threads),
                           '--max-hsps', '1', '--evalue', '1e-6',
                           '--max-target-seqs', '0', '--ultra-sensitive']
            else:
                command = ['blastp', '-query', str(proteins[query]),
                           '-db', str(dbdir / target), '-outfmt', '6',
                           '-out', str(table), '-num_threads', str(a.threads),
                           '-max_hsps', '1', '-evalue', '1e-6']
            subprocess.run(command, check=True)
            if table.stat().st_size == 0:
                raise RuntimeError(f'没有蛋白同源比对命中: {query} -> {target}')
    print(f'完成：{a.out.resolve()}')


if __name__ == '__main__':
    main()
