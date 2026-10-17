import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saturn_prepare_h5ad import convert_gene_ids, prepare_h5ad
from saturn_integrate import prepare_counts


class PrepareH5ADTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source.h5ad'
        self.output = self.root / 'prepared.h5ad'
        self.embedding = self.root / 'embedding.pt'
        self.data = ad.AnnData(
            X=sparse.csr_matrix([[0.5, 1.5], [2.5, 3.5]]),
            obs=pd.DataFrame({'cell-type': ['meristem', 'epidermis'],
                              'sample': ['S1', 'S2'], 'cell_type': ['old', 'old']},
                             index=['c1', 'c2']),
            var=pd.DataFrame(index=['Ger-000001.m1', 'Ger_000002']))
        self.data.layers['counts'] = sparse.csr_matrix([[1., 2.], [3., 4.]])
        self.data.obsm['X_umap'] = np.array([[0., 1.], [2., 3.]])
        self.data.write_h5ad(self.source)
        torch.save({'Ger_000001': torch.tensor([1., 2.]),
                    'Ger_000002': torch.tensor([2., 3.])}, self.embedding)

    def test_roundtrip_preserves_metadata_and_source_and_integrates(self):
        before = self.source.read_bytes()
        report = prepare_h5ad(self.source, self.embedding, self.output)
        result = ad.read_h5ad(self.output)
        self.assertEqual(result.var_names.tolist(), ['Ger_000001', 'Ger_000002'])
        self.assertEqual(result.obs['cell_type'].tolist(), ['meristem', 'epidermis'])
        self.assertEqual(result.obs['sample'].tolist(), ['S1', 'S2'])
        self.assertEqual(result.var['saturn_original_gene_id'].tolist(), self.data.var_names.tolist())
        np.testing.assert_array_equal(result.X.toarray(), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(result.obsm['X_umap'], self.data.obsm['X_umap'])
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(report['embedding_coverage']['matched_genes'], 2)
        saved = json.loads(self.output.with_suffix('.report.json').read_text())
        self.assertEqual(saved['changed_gene_ids'], 1)
        mapping = pd.read_csv(self.output.with_suffix('.gene_id_map.csv'))
        self.assertEqual(mapping['saturn_id'].tolist(), result.var_names.tolist())
        # The generated file must be accepted by the existing SATURN entry point.
        prepared = prepare_counts(self.output, 'cell_type', None, 'gb')
        self.assertEqual(prepared.obs['saturn_cell_type'].tolist(), ['meristem', 'epidermis'])

    def test_id_rules_are_specific_and_collisions_rejected(self):
        result = convert_gene_ids([' Ger-004062.m1 ', 'CL1A000010.m1', 'Ger_000002'], 'gerbera')
        self.assertEqual(result.tolist(), ['Ger_004062', 'CL1A000010.m1', 'Ger_000002'])
        self.assertEqual(convert_gene_ids(['CL1A000010.1'], 'none').tolist(), ['CL1A000010.1'])
        for names in (['Ger_1.m1', 'Ger_1.m2'], ['Ger-1', 'Ger_1'], ['Ger_1', 'ger_1']):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, '重复'):
                convert_gene_ids(names, 'gerbera')

    def test_no_outputs_on_invalid_counts_labels_or_ids(self):
        cases = ('normalized', 'missing_label', 'empty_label', 'zero_cell', 'collision')
        for case in cases:
            with self.subTest(case=case):
                data = self.data.copy()
                if case == 'normalized':
                    del data.layers['counts']
                elif case == 'missing_label':
                    del data.obs['cell-type']
                elif case == 'empty_label':
                    data.obs.loc['c1', 'cell-type'] = ' '
                elif case == 'zero_cell':
                    data.layers['counts'] = sparse.csr_matrix([[0., 0.], [1., 2.]])
                else:
                    data.var_names = ['Ger_1.m1', 'Ger_1.m2']
                data.write_h5ad(self.source)
                with self.assertRaises(ValueError):
                    prepare_h5ad(self.source, self.embedding, self.output)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.output.with_suffix('.report.json').exists())

    def test_zero_embedding_coverage_rejected(self):
        torch.save({'other': torch.tensor([1., 2.])}, self.embedding)
        with self.assertRaisesRegex(ValueError, '完全不匹配'):
            prepare_h5ad(self.source, self.embedding, self.output)
        self.assertFalse(self.output.exists())

    def test_x_counts_and_cl_label_conversion(self):
        self.data.X = self.data.layers['counts'].copy()
        del self.data.layers['counts']
        self.data.var_names = ['CL1A000010', 'CL1A000020']
        self.data.obs.rename(columns={'cell-type': 'celltype'}, inplace=True)
        self.data.write_h5ad(self.source)
        torch.save({'CL1A000010': torch.tensor([1., 2.])}, self.embedding)
        report = prepare_h5ad(self.source, self.embedding, self.output,
                              label_key='celltype', id_mode='none')
        self.assertEqual(report['counts_source'], 'X')
        self.assertEqual(report['changed_gene_ids'], 0)
        # Partial coverage is reported; no expression genes are silently dropped.
        self.assertEqual(report['embedding_coverage']['matched_genes'], 1)
        self.assertEqual(ad.read_h5ad(self.output).n_vars, 2)

    def test_explicit_counts_selection(self):
        with self.assertRaisesRegex(ValueError, '整数 counts'):
            prepare_h5ad(self.source, self.embedding, self.output, counts_layer='X')
        with self.assertRaisesRegex(ValueError, '缺少 counts layer'):
            prepare_h5ad(self.source, self.embedding, self.output, counts_layer='missing')
        self.data.layers['raw_counts'] = self.data.layers['counts'].copy()
        self.data.write_h5ad(self.source)
        report = prepare_h5ad(self.source, self.embedding, self.output, counts_layer='raw_counts')
        self.assertEqual(report['counts_source'], "layers['raw_counts']")

    def test_cli_and_no_overwrite(self):
        script = Path(__file__).resolve().parents[1] / 'saturn_prepare_h5ad.py'
        command = [sys.executable, str(script), '--input', str(self.source),
                   '--embedding', str(self.embedding), '--out', str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = self.output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('拒绝覆盖', result.stderr)
        self.assertEqual(before, self.output.read_bytes())

    def test_sidecar_no_overwrite(self):
        sidecar = self.output.with_suffix('.gene_id_map.csv')
        sidecar.write_text('existing mapping')
        with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
            prepare_h5ad(self.source, self.embedding, self.output)
        self.assertEqual(sidecar.read_text(), 'existing mapping')
        self.assertFalse(self.output.exists())
