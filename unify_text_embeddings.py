#!/usr/bin/env python3
"""从真实基因功能描述生成 Unify 文本嵌入；本地 Hugging Face LLaMA 模型。"""
import argparse
import csv
import json
from pathlib import Path


def read_descriptions(path):
    descriptions, casing = {}, {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t' if str(path).endswith('.tsv') else ',')
        if not {'gene_id', 'description'} <= set(reader.fieldnames or []):
            raise ValueError('功能描述表必须含 gene_id,description 两列（CSV 或 TSV）')
        for row in reader:
            gene, text = (row.get('gene_id') or '').strip(), (row.get('description') or '').strip()
            if not gene or not text:
                raise ValueError(f'第 {reader.line_num} 行的基因 ID 或描述为空；请先清理未注释记录')
            if gene.lower() in casing and casing[gene.lower()] != gene:
                raise ValueError(f'基因 ID 大小写冲突：{gene}')
            casing[gene.lower()] = gene
            descriptions.setdefault(gene, [])
            if text not in descriptions[gene]:
                descriptions[gene].append(text)
    if not descriptions:
        raise ValueError('功能描述表为空')
    return descriptions


def encode_descriptions(model, tokenizer, descriptions, device):
    import torch
    vectors = []
    for description in descriptions:
        tokens = tokenizer(description, return_tensors='pt', truncation=False)
        limit = getattr(model.config, 'max_position_embeddings', None)
        if limit and tokens['input_ids'].shape[1] > limit:
            raise ValueError(f'功能描述超过模型容量 {limit} tokens；请整理描述后重试')
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = model(**tokens).last_hidden_state
        # 与官方示例保持一致：每段描述最后一个 token，多个描述等权平均。
        vectors.append(hidden[0, -1].detach().float().cpu())
    return torch.stack(vectors).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--descriptions', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True, help='完整本地 Hugging Face 模型和 tokenizer 目录')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    metadata_path = Path(str(a.out) + '.json')
    if a.out.exists() or metadata_path.exists():
        p.error('输出已存在')
    if not a.model_dir.is_dir():
        p.error('model-dir 必须是已下载的完整文本模型目录；ESM 权重不能用于文本嵌入')
    try:
        descriptions = read_descriptions(a.descriptions)
    except (ValueError, OSError) as error:
        p.error(str(error))
    import torch
    from transformers import AutoModel, AutoTokenizer
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA 不可用；请在 LSF 作业内申请 GPU')
    dtype = torch.float16 if a.device.startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(a.model_dir, local_files_only=True, trust_remote_code=False,
                                      torch_dtype=dtype).eval().to(a.device)
    embeddings = {}
    with torch.inference_mode():
        for i, (gene, texts) in enumerate(descriptions.items(), 1):
            vector = encode_descriptions(model, tokenizer, texts, a.device)
            if not torch.isfinite(vector).all():
                raise ValueError(f'{gene}: 文本嵌入包含非有限数值')
            embeddings[gene] = vector
            if i % 100 == 0:
                print(f'{i}/{len(descriptions)} genes', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as handle:
        torch.save(embeddings, handle)
    metadata_path.write_text(json.dumps({'model_dir': str(a.model_dir.resolve()),
                                        'description_file': str(a.descriptions.resolve()),
                                        'pooling': 'last token per description; mean over descriptions',
                                        'genes': len(embeddings)}, indent=2))
    print(f'完成：{len(embeddings)} 个基因，{a.out}')


if __name__ == '__main__':
    main()
