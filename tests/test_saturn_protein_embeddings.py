from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saturn_protein_embeddings import main


class EmbeddingPreparationTest(unittest.TestCase):
    def test_isoform_mean_and_special_tokens_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / 'mapping.csv'
            mapping.write_text('protein_id,gene_id\np1,G1\np2,G1\n')
            output = root / 'out.pt'
            records = [SimpleNamespace(id='p1', seq='AC'), SimpleNamespace(id='p2', seq='DE')]
            class Model:
                num_layers = 33
                calls = 0
                def eval(self):
                    return self
                def to(self, device):
                    return self
                def __call__(self, tokens, **kwargs):
                    self.calls += 1
                    return {'representations': {33: torch.tensor(
                        [[[100., 100.], [float(self.calls), 2.],
                          [float(self.calls), 4.], [100., 100.]]])}}
            alphabet = SimpleNamespace(get_batch_converter=lambda: lambda rows: (
                None, None, torch.ones((1, 4), dtype=torch.int64)))
            fake_esm = SimpleNamespace(pretrained=SimpleNamespace(
                esm2_t33_650M_UR50D=lambda: (Model(), alphabet)))
            fake_bio = SimpleNamespace(SeqIO=SimpleNamespace(parse=lambda *args: iter(records)))
            args = ['embeddings', '--fasta', str(root / 'proteins.fa'),
                    '--gene-map', str(mapping), '--out', str(output), '--device', 'cpu']
            with patch.dict(sys.modules, {'esm': fake_esm, 'Bio': fake_bio}), patch.object(sys, 'argv', args):
                main()
            loaded = torch.load(output, weights_only=True)
            torch.testing.assert_close(loaded['G1'], torch.tensor([1.5, 3.]))

    def test_unmapped_protein_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / 'mapping.csv'
            mapping.write_text('protein_id,gene_id\np1,G1\n')
            fake_bio = SimpleNamespace(SeqIO=SimpleNamespace(parse=lambda *args: iter([
                SimpleNamespace(id='unknown', seq='AC')])))
            args = ['embeddings', '--fasta', str(root / 'proteins.fa'),
                    '--gene-map', str(mapping), '--out', str(root / 'out.pt'), '--device', 'cpu']
            with patch.dict(sys.modules, {'esm': SimpleNamespace(), 'Bio': fake_bio}), patch.object(sys, 'argv', args):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
            self.assertFalse((root / 'out.pt').exists())
