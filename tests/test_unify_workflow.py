from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from io import StringIO
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unify_integrate import parse_columns, build_command
from unify_text_embeddings import read_descriptions, encode_descriptions
from unify_prepare_descriptions import convert_rows, read_cl_tair


class UnifyWorkflowTest(unittest.TestCase):
    def test_cl_unnamed_columns_quotes_and_missing_descriptions(self):
        text = ('gene\ttair_id\t\t\n'
                'CL1A000080\tAT3G16590.1 \t Symbols: no symbol available \t no full name available \n'
                'CL1A000120\tAT1G74960.3 \t" Symbols: ATKAS2, FAB1, KAS2 "\t" ARABIDOPSIS BETA-KETOACYL-ACP SYNTHETASE 2, fatty acid biosynthesis 1 "\n'
                'CL1A000180\tAT1G19210.1 \t" Symbols: ERF017, ERF17 "\t  \n')
        desc, evidence, unmatched, missing, stats = convert_rows(
            read_cl_tair(StringIO(text)), {'CL1A000080', 'CL1A000120', 'CL1A000180'},
            'gene', 'ath_description', None)
        self.assertEqual(len(desc), 1)
        self.assertEqual(desc[0][0], 'CL1A000120')
        self.assertIn(', fatty acid biosynthesis 1', desc[0][1])
        self.assertEqual(evidence[0]['ath_acc'], 'AT1G74960.3')
        self.assertEqual(evidence[0]['ath_symbol'], 'ATKAS2, FAB1, KAS2')
        self.assertEqual(missing, ['CL1A000080', 'CL1A000180'])
        self.assertEqual(stats['matched_rows_without_description'], 2)

    def test_arabidopsis_transfer_keeps_native_ids_and_skips_missing_descriptions(self):
        rows = [
            {'gerbera_id': 'Ger_042786.m1', 'ath_acc': 'AT3G02960.1', 'ath_description': 'no full name available'},
            {'gerbera_id': 'Ger_042545.m1', 'ath_acc': 'AT2G26000.1', 'ath_description': 'BRAP2 RING ZnF UBP domain-containing protein 2'},
            {'gerbera_id': 'Ger_042545.m2', 'ath_acc': 'AT2G26000.1', 'ath_description': 'BRAP2 RING ZnF UBP domain-containing protein 2'},
            {'gerbera_id': 'other.m1', 'ath_acc': 'AT2G26000.1', 'ath_description': 'test'},
        ]
        desc, evidence, unmatched, missing, stats = convert_rows(
            rows, {'Ger_042786', 'Ger_042545'}, 'gerbera_id', 'ath_description', r'\.m[0-9]+$')
        self.assertEqual(len(desc), 1)
        self.assertEqual(desc[0][0], 'Ger_042545')
        self.assertEqual(len(evidence), 2)
        self.assertEqual(missing, ['Ger_042786'])
        self.assertEqual(unmatched, ['other.m1'])
        self.assertEqual(stats['matched_rows_without_description'], 1)

    def test_labels_and_layers_are_per_species(self):
        species = {'gb': 'gb.h5ad', 'cl': 'cl.h5ad'}
        self.assertEqual(parse_columns(['gb=cell-type', 'cl=celltype'], species, 'labels'),
                         {'gb': 'cell-type', 'cl': 'celltype'})
        with self.assertRaises(ValueError):
            parse_columns(['xx=counts'], species, 'layers')
        with self.assertRaises(ValueError):
            parse_columns(['gb=a', 'gb=b'], species, 'labels')

    def test_command_keeps_modality_order(self):
        args = SimpleNamespace(unify_dir='/tmp/Unify', hv_genes=8000, esm_macrogenes=2000,
                               text_macrogenes=2000, batch_size=256, epochs=500, seed=0)
        command = build_command(args, ['gb', 'cl'], {'cl': '/cl', 'gb': '/gb'},
                                {'cl': '/cl_protein', 'gb': '/gb_protein'},
                                {'gb': '/gb_text', 'cl': '/cl_text'}, Path('/out'))
        for flag, expected in [('--h5ad_files', ['/gb', '/cl']),
                               ('--gene_esm_embedding_path', ['/gb_protein', '/cl_protein']),
                               ('--gene_llama_embedding_path', ['/gb_text', '/cl_text'])]:
            index = command.index(flag)
            self.assertEqual(command[index+1:index+3], expected)
        self.assertNotIn('--batch_labels', command)

    def test_gene_description_groups_and_empty_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'desc.tsv'
            path.write_text('gene_id\tdescription\nGer_1\tDNA binding\nGer_1\tDNA binding\nGer_1\ttranscription regulator\n')
            self.assertEqual(read_descriptions(path), {'Ger_1': ['DNA binding', 'transcription regulator']})
            path.write_text('gene_id\tdescription\nGer_1\t\n')
            with self.assertRaises(ValueError):
                read_descriptions(path)

    def test_last_token_and_description_mean(self):
        def tokenizer(text, **kwargs):
            return {'input_ids': torch.tensor([[100, int(text)]])}
        class Model:
            config = SimpleNamespace(max_position_embeddings=3)
            def __call__(self, input_ids):
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))
        with torch.inference_mode():
            result = encode_descriptions(Model(), tokenizer, ['2', '4'], 'cpu')
        torch.testing.assert_close(result, torch.tensor([3.]))
