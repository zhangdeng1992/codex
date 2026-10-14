#!/usr/bin/env python3
"""将拟南芥同源注释整理为本物种基因功能描述，并保存来源与覆盖清单。"""
import argparse
import csv
import json
from pathlib import Path
import re

EMPTY_DESCRIPTIONS = {'', 'no full name available', 'no description available', 'no description', 'unknown', 'nan', 'none', 'na', 'n/a', '-'}


def read_cl_tair(handle):
    """读取 gene/tair_id/无标题symbol/无标题description 四列 TSV。"""
    reader = csv.reader(handle, delimiter='\t', skipinitialspace=True)
    header = next(reader, [])
    if [value.strip() for value in header[:2]] != ['gene', 'tair_id']:
        raise ValueError('cl-tair 格式前两列表头必须为 gene 和 tair_id')
    for row in reader:
        if not row or not any(value.strip() for value in row):
            continue
        if not 2 <= len(row) <= 4:
            raise ValueError(f'第 {reader.line_num} 行不是预期的四列 TSV；请检查分隔符')
        row += [''] * (4 - len(row))
        gene, accession, symbol, description = [value.strip() for value in row]
        yield {'gene': gene, 'ath_acc': accession,
               'ath_symbol': symbol.removeprefix('Symbols:').strip(),
               'ath_description': description}


def convert_rows(rows, genes, gene_column, description_column, strip_suffix):
    pattern = re.compile(strip_suffix) if strip_suffix else None
    if pattern and not strip_suffix.endswith('$'):
        raise ValueError('strip-suffix 必须是以 $ 结尾的正则表达式')
    evidence, unmatched, empty, descriptions = [], [], [], set()
    total = 0
    for row in rows:
        total += 1
        original = (row.get(gene_column) or '').strip()
        text = (row.get(description_column) or '').strip()
        gene = original
        if gene not in genes and pattern:
            gene = pattern.sub('', gene)
        if gene not in genes:
            unmatched.append(original)
            continue
        if text.lower() in EMPTY_DESCRIPTIONS:
            empty.append(gene)
            continue
        descriptions.add((gene, text))
        evidence.append({'gene_id': gene, 'source_id': original,
                         'ath_acc': row.get('ath_acc', ''),
                         'ath_symbol': row.get('ath_symbol', ''),
                         'description': text})
    covered = {gene for gene, _ in descriptions}
    stats = {'input_annotation_rows': total, 'expression_genes': len(genes),
             'genes_with_descriptions': len(covered), 'unique_gene_description_pairs': len(descriptions),
             'rows_without_expression_match': len(unmatched),
             'matched_rows_without_description': len(empty),
             'genes_without_descriptions': len(genes - covered)}
    return sorted(descriptions), evidence, sorted(set(unmatched)), sorted(genes - covered), stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True, help='带表头的 TSV 注释表')
    p.add_argument('--annotation-format', choices=['named', 'cl-tair'], default='named',
                   help='named 使用列名；cl-tair 读取 gene/tair_id 后无标题的 symbol 和描述列')
    p.add_argument('--h5ad', type=Path, required=True, help='本物种的表达数据，仅用基因 ID')
    p.add_argument('--gene-column', default='gerbera_id')
    p.add_argument('--description-column', default='ath_description')
    p.add_argument('--strip-suffix', help='显式 ID 转换规则，如 \\.m[0-9]+$；优先原 ID 匹配')
    p.add_argument('--out', type=Path, required=True, help='新的输出目录')
    a = p.parse_args()
    if a.out.exists():
        p.error('输出目录已存在')
    import anndata as ad
    data = ad.read_h5ad(a.h5ad, backed='r')
    try:
        if not data.var_names.is_unique:
            p.error('表达基因 ID 不唯一')
        genes = set(map(str, data.var_names))
    finally:
        data.file.close()
    with a.annotations.open(encoding='utf-8-sig', newline='') as handle:
        if a.annotation_format == 'cl-tair':
            reader = read_cl_tair(handle)
            gene_column, description_column = 'gene', 'ath_description'
        else:
            reader = csv.DictReader(handle, delimiter='\t')
            gene_column, description_column = a.gene_column, a.description_column
            if not {gene_column, description_column} <= set(reader.fieldnames or []):
                p.error(f'注释表缺少列 {gene_column} 或 {description_column}')
        try:
            desc, evidence, unmatched, missing, stats = convert_rows(
                reader, genes, gene_column, description_column, a.strip_suffix)
        except ValueError as error:
            p.error(str(error))
    if not desc:
        p.error('没有可用且能匹配表达基因的功能描述；请检查注释或 ID 规则')
    a.out.mkdir(parents=True)
    with (a.out / 'gene_descriptions.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['gene_id', 'description'])
        writer.writerows(desc)
    with (a.out / 'annotation_evidence.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['gene_id', 'source_id', 'ath_acc', 'ath_symbol', 'description'])
        writer.writeheader()
        writer.writerows(evidence)
    for name, values, column in [('genes_without_descriptions.csv', missing, 'gene_id'),
                                 ('unmatched_annotation_ids.csv', unmatched, 'source_id')]:
        with (a.out / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([column])
            writer.writerows((value,) for value in values)
    (a.out / 'coverage.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f'已保存：{a.out / "gene_descriptions.tsv"}')


if __name__ == '__main__':
    main()
