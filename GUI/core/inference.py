import ctypes
import sys
import os
import json
from pathlib import Path
import numpy as np
from typing import Optional
from PySide6.QtCore import Signal, QThread
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem

# Load platform-specific shared library produced by Makefile/Makefile.windows.
_LIB_DIR = Path(__file__).resolve().parent
if sys.platform.startswith('win'):
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'hybrid_engine.dll'))
elif sys.platform == 'darwin':
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'libhybrid_engine.dylib'))
else:
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'libhybrid_engine.so'))

engine_lib.init_engine.argtypes = [ctypes.c_char_p]
engine_lib.init_engine.restype = ctypes.c_int

engine_lib.run_hybrid_inference.argtypes = [
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # node_features
	ctypes.c_int64,                                                           # num_nodes
	ctypes.c_int64,                                                           # node_feat_dim
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),    # edge_index
	ctypes.c_int64,                                                           # num_edges
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # edge_attr
	ctypes.c_int64,                                                           # edge_feat_dim
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),    # batch_index
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),    # input_ids
	ctypes.c_int64,                                                           # seq_len
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),    # attention_mask
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # output_properties
	ctypes.c_int64                                                            # output_dim
]
engine_lib.run_hybrid_inference.restype = ctypes.c_int

engine_lib.cleanup_engine.argtypes = []
engine_lib.cleanup_engine.restype = None


_TOKENIZER = None
_TOKENIZER_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
_MAX_SEQ_LEN = 64
_DEFAULT_OUTPUT_DIM = 12
_ENGINE_READY = False
_ENGINE_MODEL_PATH = None
_CONFIDENCE_CALIBRATION_CACHE = None

def _one_hot_encoding(x, allowable_set):
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))


def _get_node_features(atom):
	return (
		_one_hot_encoding(atom.GetSymbol(), ['H', 'C', 'N', 'O', 'F', 'Unknown']) +
		_one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
		_one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1]) +
		_one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
		_one_hot_encoding(atom.GetHybridization(), [
			Chem.rdchem.HybridizationType.SP,
			Chem.rdchem.HybridizationType.SP2,
			Chem.rdchem.HybridizationType.SP3,
			'Unknown',
		]) +
		[atom.GetIsAromatic(), atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
	)


def _get_edge_features(bond):
	return (
		_one_hot_encoding(bond.GetBondType(), [
			Chem.rdchem.BondType.SINGLE,
			Chem.rdchem.BondType.DOUBLE,
			Chem.rdchem.BondType.TRIPLE,
			Chem.rdchem.BondType.AROMATIC,
		]) +
		[bond.GetIsConjugated()]
	)


def _resolve_model_path(model_path=None):
	if model_path:
		path = Path(model_path).expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(f"ONNX model not found at: {path}")
		return path

	env_model = os.environ.get('HYBRID_ONNX_MODEL_PATH', '').strip()
	if env_model:
		path = Path(env_model).expanduser().resolve()
		if path.exists():
			return path

	project_root = _LIB_DIR.parent.parent
	candidates = [
		project_root / 'GUI' / 'assets' / 'hybrid_model.onnx',
		project_root / 'GUI' / 'assets' / 'model.onnx',
		project_root / 'GUI' / 'core' / 'hybrid_model.onnx',
		project_root / 'hybrid_model.onnx',
		project_root / 'model.onnx',
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()

	raise FileNotFoundError(
		"ONNX model file was not found. Set HYBRID_ONNX_MODEL_PATH or place model at GUI/assets/hybrid_model.onnx"
	)


def init_hybrid_engine(model_path=None):
	global _ENGINE_READY
	global _ENGINE_MODEL_PATH

	resolved = _resolve_model_path(model_path)
	if _ENGINE_READY and _ENGINE_MODEL_PATH == resolved:
		return str(_ENGINE_MODEL_PATH)

	if _ENGINE_READY:
		engine_lib.cleanup_engine()
		_ENGINE_READY = False
		_ENGINE_MODEL_PATH = None

	status = engine_lib.init_engine(str(resolved).encode('utf-8'))
	if status != 0:
		raise RuntimeError(f"Failed to initialize ONNX hybrid engine (status={status})")

	_ENGINE_READY = True
	_ENGINE_MODEL_PATH = resolved
	return str(_ENGINE_MODEL_PATH)


def cleanup_hybrid_engine():
	global _ENGINE_READY
	global _ENGINE_MODEL_PATH
	if _ENGINE_READY:
		engine_lib.cleanup_engine()
		_ENGINE_READY = False
		_ENGINE_MODEL_PATH = None


def build_graph_tensors_from_smiles(smiles, random_seed: Optional[int] = None):
	smiles = (smiles or '').strip()
	if not smiles:
		raise ValueError('SMILES is empty')

	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		raise ValueError(f'Invalid SMILES: {smiles}')

	mol = Chem.AddHs(mol)
	Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

	# Build 3D geometry so distance can be included in edge features.
	params = AllChem.ETKDGv3()
	if random_seed is not None:
		params.randomSeed = int(random_seed)
	has_3d = AllChem.EmbedMolecule(mol, params) == 0
	if has_3d:
		AllChem.MMFFOptimizeMolecule(mol)
		conf = mol.GetConformer()
	else:
		conf = None

	node_features = np.asarray([
		_get_node_features(atom) for atom in mol.GetAtoms()
	], dtype=np.float32)

	edge_indices = []
	edge_attrs = []
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		topo = _get_edge_features(bond)

		if conf is not None:
			pi = conf.GetAtomPosition(i)
			pj = conf.GetAtomPosition(j)
			distance = float(((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5)
		else:
			distance = 0.0

		edge_feature = topo + [distance]
		edge_indices.extend([[i, j], [j, i]])
		edge_attrs.extend([edge_feature, edge_feature])

	if not edge_indices:
		# Keep ONNX call valid for single-atom molecules by adding a synthetic self-loop.
		edge_indices = [[0, 0]]
		edge_attrs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

	edge_index = np.asarray(edge_indices, dtype=np.int64).T
	edge_attr = np.asarray(edge_attrs, dtype=np.float32)
	batch_index = np.zeros((node_features.shape[0],), dtype=np.int64)

	return (
		np.ascontiguousarray(node_features, dtype=np.float32),
		np.ascontiguousarray(edge_index, dtype=np.int64),
		np.ascontiguousarray(edge_attr, dtype=np.float32),
		np.ascontiguousarray(batch_index, dtype=np.int64),
	)


def run_hybrid_regression(smiles, model_path=None, random_seed: Optional[int] = None):
	init_hybrid_engine(model_path=model_path)
	node_features, edge_indices, edge_attr, batch_index = build_graph_tensors_from_smiles(
		smiles,
		random_seed=random_seed,
	)
	input_ids, attention_mask = _encode_smiles(smiles)
	output_properties = np.zeros(_DEFAULT_OUTPUT_DIM, dtype=np.float32)

	num_nodes, node_feat_dim = node_features.shape
	_, num_edges = edge_indices.shape
	edge_feat_dim = edge_attr.shape[1]
	seq_len = input_ids.shape[1]

	status = engine_lib.run_hybrid_inference(
		node_features,
		num_nodes,
		node_feat_dim,
		edge_indices,
		num_edges,
		edge_attr,
		edge_feat_dim,
		batch_index,
		input_ids,
		seq_len,
		attention_mask,
		output_properties,
		_DEFAULT_OUTPUT_DIM,
	)
	if status != 0:
		raise RuntimeError(f'C Engine Inference Failed (status={status}) for SMILES: {smiles}')

	return output_properties


def _compute_confidence_from_predictions(prediction_matrix):
	predictions = np.asarray(prediction_matrix, dtype=np.float32)
	if predictions.ndim != 2:
		raise ValueError('prediction_matrix must be 2D (num_samples, num_properties)')
	if predictions.shape[0] < 1:
		raise ValueError('prediction_matrix must include at least one sample')

	means = np.mean(predictions, axis=0).astype(np.float32)
	stds = np.std(predictions, axis=0).astype(np.float32)
	abs_means = np.abs(means)
	denominator = np.maximum(abs_means, 1e-6)
	cv_percent = ((stds / denominator) * 100.0).astype(np.float32)

	# Confidence score in [0,100], penalizing larger relative variance.
	avg_cv = float(np.mean(cv_percent))
	confidence_score = float(max(0.0, min(100.0, 100.0 * np.exp(-avg_cv / 10.0))))

	return {
		'mean': means,
		'std': stds,
		'cv_percent': cv_percent,
		'confidence_score': confidence_score,
	}


def _resolve_confidence_calibration_path():
	calibration_env = os.environ.get('HYBRID_CONFIDENCE_CALIBRATION_PATH', '').strip()
	if calibration_env:
		path = Path(calibration_env).expanduser().resolve()
		if path.exists():
			return path

	project_root = _LIB_DIR.parent.parent
	default_path = project_root / 'GUI' / 'assets' / 'confidence_calibration.json'
	if default_path.exists():
		return default_path.resolve()

	return None


def _load_confidence_calibration():
	global _CONFIDENCE_CALIBRATION_CACHE
	if _CONFIDENCE_CALIBRATION_CACHE is not None:
		return _CONFIDENCE_CALIBRATION_CACHE

	path = _resolve_confidence_calibration_path()
	if path is None:
		_CONFIDENCE_CALIBRATION_CACHE = {}
		return _CONFIDENCE_CALIBRATION_CACHE

	try:
		with path.open('r', encoding='utf-8') as f:
			data = json.load(f)
		quantiles = data.get('quantiles')
		if not isinstance(quantiles, list) or len(quantiles) != _DEFAULT_OUTPUT_DIM:
			_CONFIDENCE_CALIBRATION_CACHE = {}
			return _CONFIDENCE_CALIBRATION_CACHE
		_CONFIDENCE_CALIBRATION_CACHE = {
			'path': str(path),
			'method': str(data.get('method') or 'conformal_residual_quantile'),
			'alpha': float(data.get('alpha', 0.1)),
			'quantiles': np.asarray(quantiles, dtype=np.float32),
		}
		return _CONFIDENCE_CALIBRATION_CACHE
	except Exception:
		_CONFIDENCE_CALIBRATION_CACHE = {}
		return _CONFIDENCE_CALIBRATION_CACHE


def _compute_prediction_intervals(mean_values, std_values):
	means = np.asarray(mean_values, dtype=np.float32)
	stds = np.asarray(std_values, dtype=np.float32)
	calibration = _load_confidence_calibration()

	if calibration and isinstance(calibration.get('quantiles'), np.ndarray):
		radius = np.maximum(calibration['quantiles'], 0.0).astype(np.float32)
		method = calibration.get('method', 'conformal_residual_quantile')
		alpha = float(calibration.get('alpha', 0.1))
		calibration_path = calibration.get('path')
	else:
		# Fallback interval: Gaussian approximation around conformer-run mean.
		radius = (1.96 * stds).astype(np.float32)
		method = 'gaussian_std_fallback'
		alpha = 0.05
		calibration_path = None

	lower = (means - radius).astype(np.float32)
	upper = (means + radius).astype(np.float32)
	return {
		'lower': lower,
		'upper': upper,
		'radius': radius,
		'method': method,
		'alpha': alpha,
		'calibration_path': calibration_path,
	}


def run_hybrid_regression_with_confidence(smiles, model_path=None, n_conformers=3, base_seed=17):
	if n_conformers < 1:
		raise ValueError('n_conformers must be >= 1')

	init_hybrid_engine(model_path=model_path)
	predictions = []
	warnings = []

	for i in range(n_conformers):
		seed = int(base_seed + i)
		try:
			pred = run_hybrid_regression(smiles, model_path=model_path, random_seed=seed)
			predictions.append(pred)
		except Exception as exc:
			warnings.append(f'conformer_{i}_failed: {exc}')

	if not predictions:
		raise RuntimeError(f'All confidence conformer runs failed for SMILES: {smiles}')

	stats = _compute_confidence_from_predictions(np.vstack(predictions))
	intervals = _compute_prediction_intervals(stats['mean'], stats['std'])
	return {
		'prediction': stats['mean'],
		'confidence': {
			'std': stats['std'].tolist(),
			'cv_percent': stats['cv_percent'].tolist(),
			'interval_lower': intervals['lower'].tolist(),
			'interval_upper': intervals['upper'].tolist(),
			'interval_radius': intervals['radius'].tolist(),
			'interval_method': intervals['method'],
			'interval_alpha': intervals['alpha'],
			'interval_calibration_path': intervals['calibration_path'],
			'confidence_score': stats['confidence_score'],
			'n_conformers_requested': int(n_conformers),
			'n_conformers_used': int(len(predictions)),
			'mode': 'conformer_variance',
			'warnings': warnings,
		},
	}


class BatchInferenceThread(QThread):
	finished = Signal(list, list)
	error = Signal(str)

	def __init__(self, smiles_list, model_path=None, enable_confidence=True, n_conformers=3):
		super().__init__()
		self.smiles_list = [s.strip() for s in (smiles_list or []) if s and s.strip()]
		self.model_path = model_path
		self.enable_confidence = bool(enable_confidence)
		self.n_conformers = int(n_conformers)

	def run(self):
		if not self.smiles_list:
			self.error.emit('No SMILES provided for inference')
			return

		results = []
		failures = []
		try:
			init_hybrid_engine(model_path=self.model_path)
		except Exception as exc:
			self.error.emit(str(exc))
			return

		for smiles in self.smiles_list:
			try:
				if self.enable_confidence:
					result = run_hybrid_regression_with_confidence(
						smiles,
						model_path=self.model_path,
						n_conformers=self.n_conformers,
					)
					results.append((smiles, result['prediction'].tolist(), result['confidence']))
				else:
					pred = run_hybrid_regression(smiles, model_path=self.model_path)
					results.append((smiles, pred.tolist(), None))
			except Exception as exc:
				failures.append((smiles, str(exc)))

		self.finished.emit(results, failures)


class EngineWarmupThread(QThread):
	ready = Signal(str)
	error = Signal(str)

	def __init__(self, model_path=None):
		super().__init__()
		self.model_path = model_path

	def run(self):
		try:
			resolved_path = init_hybrid_engine(model_path=self.model_path)
			self.ready.emit(resolved_path)
		except Exception as exc:
			self.error.emit(str(exc))


def _get_tokenizer():
	global _TOKENIZER
	if _TOKENIZER is None:
		_TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)
	return _TOKENIZER


def _encode_smiles(smiles):
	tokenizer = _get_tokenizer()
	encoded = tokenizer(
		smiles,
		padding='max_length',
		truncation=True,
		max_length=_MAX_SEQ_LEN,
		return_tensors='np'
	)
	input_ids = np.ascontiguousarray(encoded['input_ids'].astype(np.int64))
	attention_mask = np.ascontiguousarray(encoded['attention_mask'].astype(np.int64))
	return input_ids, attention_mask


class InferenceThread(QThread):
	finished = Signal(np.ndarray)
	error = Signal(str)

	def __init__(self, node_features, edge_indices, smiles, edge_attr=None, batch_index=None):
		super().__init__()
		self.node_features = np.ascontiguousarray(node_features, dtype=np.float32)
		self.edge_indices = np.ascontiguousarray(edge_indices, dtype=np.int64)
		self.output_dim = 12
		self.smiles = (smiles or '').strip()
		self.edge_attr = None if edge_attr is None else np.ascontiguousarray(edge_attr, dtype=np.float32)
		self.batch_index = None if batch_index is None else np.ascontiguousarray(batch_index, dtype=np.int64)

	def run(self):
		try:
			if not self.smiles:
				self.error.emit('SMILES input is required for hybrid inference')
				return
			prediction = run_hybrid_regression(self.smiles)
			self.finished.emit(prediction)
		except Exception as exc:
			self.error.emit(str(exc))