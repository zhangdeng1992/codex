#!/usr/bin/env python3
"""用 ESM-1b / ESM-2 生成 SATURN gene_id -> Tensor 嵌入。"""
import argparse
import csv
from pathlib import Path

MODELS = ['esm1b_t33_650M_UR50S', 'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D']


def load_protein_model(esm, torch, model_name, model_file):
    if model_file is None:
        return getattr(esm.pretrained, model_name)()
    if not model_file.is_file():
        raise ValueError(f'本地模型文件不存在：{model_file}')
    if model_file.stem not in MODELS:
        raise ValueError('本地权重文件须保留官方模型文件名，如 esm1b_t33_650M_UR50S.pt')
    # 官方 ESM-1b checkpoint 含 argparse Namespace；只使用可信来源的本地权重。
    checkpoint = torch.load(str(model_file), map_location='cpu', weights_only=False)
    # 嵌入不使用 contact head，因此无需下载 contact-regression 辅助权重。
    return esm.pretrained.load_model_and_alphabet_core(model_file.stem, checkpoint, None)


def embed_sequence(model, convert, protein, sequence, device):
    """整条序列一次推理；BOS/EOS 不参与残基嵌入均值。"""
    _, _, tokens = convert([(protein, sequence)])
    result = model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)
    residues = result['representations'][model.num_layers][0, 1:len(sequence)+1]
    return residues.float().mean(0).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', type=Path, required=True)
    p.add_argument('--gene-map', type=Path, help='CSV: protein_id,gene_id；不指定时 FASTA ID 必须就是基因 ID')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model', choices=MODELS, default='esm2_t33_650M_UR50D')
    p.add_argument('--model-file', type=Path,
                   help='可信来源的本地官方 .pt 权重；优先于 --model，不触发下载')
    a = p.parse_args()
    if a.out.exists():
        p.error('输出文件已存在')
    import torch
    import esm
    from Bio import SeqIO
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA 不可用；可显式使用 --device cpu，但大型蛋白模型运行较慢')
    mapping = {}
    if a.gene_map:
        with a.gene_map.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'protein_id', 'gene_id'} <= set(reader.fieldnames or []):
                p.error('gene-map 必须含 protein_id,gene_id 列')
            for row in reader:
                protein, gene = row['protein_id'].strip(), row['gene_id'].strip()
                if not protein or not gene or protein in mapping:
                    p.error('gene-map 含空 ID 或重复 protein_id')
                mapping[protein] = gene
    records, seen = [], set()
    for record in SeqIO.parse(a.fasta, 'fasta'):
        if record.id in seen:
            p.error(f'重复 FASTA ID: {record.id}')
        seen.add(record.id)
        if a.gene_map and record.id not in mapping:
            p.error(f'gene-map 缺少 FASTA ID: {record.id}')
        gene = mapping[record.id] if a.gene_map else record.id
        sequence = str(record.seq).upper()
        if not sequence:
            p.error(f'{record.id}: 序列为空')
        invalid = sorted(set(sequence) - set('ACDEFGHIKLMNPQRSTVWYBXZUO'))
        if invalid:
            p.error(f'{record.id}: 不支持的字符 {invalid}')
        records.append((record.id, gene, sequence))
    if not records:
        p.error('FASTA 没有序列')
    print(f'共 {len(records)} 个蛋白，使用完整序列推理', flush=True)
    if a.gene_map and set(mapping) - seen:
        p.error('gene-map 包含 FASTA 中不存在的 protein_id')
    gene_cases = {}
    for _, gene, _ in records:
        if gene.lower() in gene_cases and gene_cases[gene.lower()] != gene:
            p.error('基因 ID 大小写冲突，SATURN 会按小写匹配')
        gene_cases[gene.lower()] = gene
    # 两个物种必须选同一模型；本地加载不调用下载接口。
    try:
        model, alphabet = load_protein_model(esm, torch, a.model, a.model_file)
    except ValueError as error:
        p.error(str(error))
    model.eval().to(a.device)
    convert = alphabet.get_batch_converter()
    sums, numbers = {}, {}
    with torch.inference_mode():
        for index, (protein, gene, sequence) in enumerate(records, 1):
            vector = embed_sequence(model, convert, protein, sequence, a.device)
            sums[gene] = sums.get(gene, torch.zeros_like(vector)) + vector
            numbers[gene] = numbers.get(gene, 0) + 1
            if index % 100 == 0:
                print(f'{index}/{len(records)} proteins', flush=True)
    # 多个蛋白对应一个基因时，对其嵌入取均值。
    embeddings = {gene: total / numbers[gene] for gene, total in sums.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as handle:
        torch.save(embeddings, handle)
    print(f'完成：{len(records)} 个蛋白 → {len(embeddings)} 个基因；{a.out}')


if __name__ == '__main__':
    main()
