#!/usr/bin/env python3
"""
Test suite for the inference module.
Tests graph building, SMILES parsing, tokenization, and inference functionality.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from GUI.core.inference import (
    _get_node_features,
    _get_edge_features,
    build_graph_tensors_from_smiles,
    _one_hot_encoding,
    _encode_smiles,
    _compute_confidence_from_predictions,
    _compute_prediction_intervals,
    _resolve_model_path,
    cleanup_hybrid_engine,
)
from rdkit import Chem
from rdkit.Chem import AllChem


class TestOneHotEncoding(unittest.TestCase):
    """Test one-hot encoding utility."""

    def test_valid_element(self):
        """Test valid element encoding."""
        result = _one_hot_encoding('C', ['H', 'C', 'N', 'O', 'F', 'Unknown'])
        self.assertEqual(result, [False, True, False, False, False, False])

    def test_invalid_element_maps_to_unknown(self):
        """Test that invalid elements map to 'Unknown'."""
        result = _one_hot_encoding('X', ['H', 'C', 'N', 'O', 'F', 'Unknown'])
        self.assertEqual(result, [False, False, False, False, False, True])

    def test_degree_encoding(self):
        """Test degree encoding."""
        result = _one_hot_encoding(2, [0, 1, 2, 3, 4, 5])
        self.assertEqual(result, [False, False, True, False, False, False])


class TestNodeFeatures(unittest.TestCase):
    """Test node feature extraction."""

    def test_node_features_hydrogen(self):
        """Test node features for hydrogen atom."""
        smiles = "[H]"
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            atom = mol.GetAtomWithIdx(0)
            features = _get_node_features(atom)
            # Should return a list of features
            self.assertIsInstance(features, list)
            self.assertGreater(len(features), 0)

    def test_node_features_carbon(self):
        """Test node features for carbon atom."""
        smiles = "C"
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        mol = Chem.AddHs(mol)
        atom = mol.GetAtomWithIdx(0)
        features = _get_node_features(atom)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)


class TestEdgeFeatures(unittest.TestCase):
    """Test edge feature extraction."""

    def test_edge_features_single_bond(self):
        """Test edge features for single bond."""
        smiles = "CC"
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        bond = mol.GetBondWithIdx(0)
        features = _get_edge_features(bond)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_edge_features_double_bond(self):
        """Test edge features for double bond."""
        smiles = "C=C"
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        bond = mol.GetBondWithIdx(0)
        features = _get_edge_features(bond)
        self.assertIsInstance(features, list)


class TestGraphTensorBuilding(unittest.TestCase):
    """Test graph tensor building from SMILES."""

    def test_simple_methane(self):
        """Test graph building for methane (CH4)."""
        node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles("C")
        
        # Check shapes
        self.assertGreater(node_features.shape[0], 0)  # Should have nodes
        self.assertEqual(node_features.dtype, np.float32)
        self.assertEqual(edge_indices.dtype, np.int64)
        self.assertEqual(edge_attr.dtype, np.float32)
        self.assertEqual(batch_index.dtype, np.int64)
        
        # Check batch index matches number of nodes
        self.assertEqual(len(batch_index), node_features.shape[0])

    def test_benzene(self):
        """Test graph building for benzene."""
        node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles("c1ccccc1")
        
        self.assertGreater(node_features.shape[0], 0)
        self.assertEqual(node_features.shape[1], 26)  # Node feature dimension
        self.assertEqual(edge_indices.shape[0], 2)  # Should be 2D (edge pairs)
        self.assertGreater(edge_indices.shape[1], 0)

    def test_ethane(self):
        """Test graph building for ethane."""
        node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles("CC")
        
        self.assertGreater(node_features.shape[0], 0)
        self.assertGreater(edge_indices.shape[1], 0)

    def test_water(self):
        """Test graph building for water."""
        node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles("O")
        
        self.assertGreater(node_features.shape[0], 0)
        self.assertEqual(batch_index.shape[0], node_features.shape[0])

    def test_empty_smiles_raises_error(self):
        """Test that empty SMILES raises ValueError."""
        with self.assertRaises(ValueError):
            build_graph_tensors_from_smiles("")

    def test_none_smiles_raises_error(self):
        """Test that None SMILES raises ValueError."""
        with self.assertRaises(ValueError):
            build_graph_tensors_from_smiles(None)

    def test_whitespace_only_smiles_raises_error(self):
        """Test that whitespace-only SMILES raises ValueError."""
        with self.assertRaises(ValueError):
            build_graph_tensors_from_smiles("   ")

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises ValueError."""
        with self.assertRaises(ValueError):
            build_graph_tensors_from_smiles("XYZ123")

    def test_invalid_smiles_with_brackets(self):
        """Test another invalid SMILES pattern."""
        with self.assertRaises(ValueError):
            build_graph_tensors_from_smiles("[INVALID]")

    def test_node_features_dimension(self):
        """Test that node features have correct dimension."""
        node_features, _, _, _ = build_graph_tensors_from_smiles("C")
        # Feature dimension should be consistent
        # [6 atom types] + [6 degrees] + [3 charges] + [5 H counts] + [4 hybridization] + [2 aromatic/chiral]
        # = 26 features
        self.assertEqual(node_features.shape[1], 26)


class TestSMILESTokenization(unittest.TestCase):
    """Test SMILES tokenization."""

    def test_tokenize_simple_smiles(self):
        """Test tokenization of simple SMILES."""
        input_ids, attention_mask = _encode_smiles("C")
        
        self.assertEqual(input_ids.dtype, np.int64)
        self.assertEqual(attention_mask.dtype, np.int64)
        self.assertEqual(input_ids.shape[0], 1)  # Batch size 1
        self.assertEqual(input_ids.shape[1], 64)  # Max sequence length
        self.assertEqual(attention_mask.shape[1], 64)

    def test_tokenize_benzene(self):
        """Test tokenization of benzene."""
        input_ids, attention_mask = _encode_smiles("c1ccccc1")
        
        self.assertEqual(input_ids.shape, (1, 64))
        self.assertEqual(attention_mask.shape, (1, 64))
        # Check that attention mask marks the token
        self.assertGreaterEqual(np.sum(attention_mask), 1)

    def test_tokenize_long_smiles(self):
        """Test tokenization handles long SMILES correctly."""
        long_smiles = "C" * 100  # Very long SMILES
        input_ids, attention_mask = _encode_smiles(long_smiles)
        
        # Should still have shape (1, 64) due to truncation/padding
        self.assertEqual(input_ids.shape, (1, 64))
        self.assertEqual(attention_mask.shape, (1, 64))

    def test_tokenize_empty_smiles(self):
        """Test tokenization of empty SMILES."""
        input_ids, attention_mask = _encode_smiles("")
        
        self.assertEqual(input_ids.shape, (1, 64))
        # Should be mostly padding tokens
        self.assertGreater(np.sum(attention_mask == 0), 50)


class TestModelPathResolution(unittest.TestCase):
    """Test model path resolution."""

    def test_resolve_nonexistent_explicit_path(self):
        """Test that nonexistent explicit path raises error."""
        with self.assertRaises(FileNotFoundError):
            _resolve_model_path("/nonexistent/path/model.onnx")

    def test_resolve_with_tilde_expansion(self):
        """Test that tilde expansion works."""
        # This should fail gracefully for nonexistent paths
        with self.assertRaises(FileNotFoundError):
            _resolve_model_path("~/nonexistent_model.onnx")

    def test_resolve_with_none_checks_candidates(self):
        """Test that resolve with None checks candidate paths."""
        # Should either find a model or raise error about not finding one
        try:
            path = _resolve_model_path(None)
            # If it finds one, verify it exists
            self.assertTrue(Path(path).exists())
        except FileNotFoundError as e:
            # Expected if no model exists
            self.assertIn("not found", str(e).lower())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_atom_molecule(self):
        """Test handling of single-atom molecules."""
        for smiles in ["C", "O", "N", "[H]"]:
            node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(smiles)
            self.assertGreater(node_features.shape[0], 0)
            # Single atom should handle gracefully (synthetic self-loop added)
            if edge_indices.shape[1] > 0:
                self.assertTrue(True)

    def test_large_molecule(self):
        """Test handling of larger molecules."""
        # Caffeine SMILES
        caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(caffeine)
        
        self.assertGreater(node_features.shape[0], 5)  # Should have multiple atoms
        self.assertGreater(edge_indices.shape[1], 0)

    def test_molecule_with_charges(self):
        """Test handling of charged molecules."""
        # Charged molecules
        for smiles in ["[NH4+]", "[Cl-]", "[O-2]"]:
            node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(smiles)
            self.assertGreater(node_features.shape[0], 0)

    def test_aromatic_molecules(self):
        """Test handling of aromatic molecules."""
        for smiles in ["c1ccccc1", "c1ccc2c(c1)ccc1ccccc12"]:
            node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(smiles)
            self.assertGreater(node_features.shape[0], 0)


class TestCleanup(unittest.TestCase):
    """Test cleanup functionality."""

    def test_cleanup_engine(self):
        """Test that cleanup_engine doesn't raise errors."""
        try:
            cleanup_hybrid_engine()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"cleanup_hybrid_engine raised {type(e).__name__}: {e}")


class TestConfidenceScoring(unittest.TestCase):
    """Test confidence scoring statistics."""

    def test_confidence_shapes(self):
        preds = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.2, 2.1, 2.9],
                [0.8, 1.9, 3.2],
            ],
            dtype=np.float32,
        )
        stats = _compute_confidence_from_predictions(preds)

        self.assertEqual(stats['mean'].shape, (3,))
        self.assertEqual(stats['std'].shape, (3,))
        self.assertEqual(stats['cv_percent'].shape, (3,))
        self.assertGreaterEqual(stats['confidence_score'], 0.0)
        self.assertLessEqual(stats['confidence_score'], 100.0)

    def test_confidence_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            _compute_confidence_from_predictions(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_confidence_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            _compute_confidence_from_predictions(np.zeros((0, 12), dtype=np.float32))

    def test_zero_variance_gives_high_confidence(self):
        preds = np.ones((4, 2), dtype=np.float32) * 5.0
        stats = _compute_confidence_from_predictions(preds)

        self.assertTrue(np.allclose(stats['std'], 0.0))
        self.assertTrue(np.allclose(stats['cv_percent'], 0.0))
        self.assertAlmostEqual(stats['confidence_score'], 100.0, places=4)

    def test_interval_fallback_uses_gaussian_std(self):
        means = np.array([1.0, 2.0], dtype=np.float32)
        stds = np.array([0.5, 1.0], dtype=np.float32)
        with patch("GUI.core.inference._load_confidence_calibration", return_value={}):
            intervals = _compute_prediction_intervals(means, stds)

        self.assertEqual(intervals["method"], "gaussian_std_fallback")
        self.assertAlmostEqual(intervals["alpha"], 0.05)
        self.assertTrue(np.allclose(intervals["radius"], np.array([0.98, 1.96], dtype=np.float32), atol=1e-6))
        self.assertTrue(np.allclose(intervals["lower"], means - intervals["radius"]))
        self.assertTrue(np.allclose(intervals["upper"], means + intervals["radius"]))

    def test_interval_uses_calibration_quantiles_when_available(self):
        means = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        stds = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        calibration = {
            "path": "/tmp/calibration.json",
            "method": "conformal_residual_quantile",
            "alpha": 0.1,
            "quantiles": np.array([0.2, 0.4, 0.8], dtype=np.float32),
        }
        with patch("GUI.core.inference._load_confidence_calibration", return_value=calibration):
            intervals = _compute_prediction_intervals(means, stds)

        self.assertEqual(intervals["method"], "conformal_residual_quantile")
        self.assertAlmostEqual(intervals["alpha"], 0.1)
        self.assertEqual(intervals["calibration_path"], "/tmp/calibration.json")
        self.assertTrue(np.allclose(intervals["radius"], calibration["quantiles"]))
        self.assertTrue(np.allclose(intervals["lower"], means - calibration["quantiles"]))
        self.assertTrue(np.allclose(intervals["upper"], means + calibration["quantiles"]))


def run_tests(verbose=True):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOneHotEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestNodeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphTensorBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestSMILESTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPathResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScoring))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests(verbose=True)
    exit(0 if success else 1)
