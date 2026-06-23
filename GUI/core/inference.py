import ctypes
import sys
import os
import json
from pathlib import Path
import numpy as np
from typing import Optional, List
from PySide6.QtCore import Signal, QThread
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.qm9_delta import (
	QM9_DELTA_TARGET_COLUMNS,
)

# Load platform-specific shared library
_LIB_DIR = Path(__file__).resolve().parent
if sys.platform.startswith('win'):
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'hybrid_engine.dll'))
elif sys.platform == 'darwin':
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'libhybrid_engine.dylib'))
else:
	engine_lib = ctypes.CDLL(str(_LIB_DIR / 'libhybrid_engine.so'))

# --- C Function Signatures ---
engine_lib.init_engine.argtypes = [ctypes.c_char_p]
engine_lib.init_engine.restype = ctypes.c_int

engine_lib.run_hybrid_inference.argtypes = [
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
	ctypes.c_int64,
	ctypes.c_int64,
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),
	ctypes.c_int64,
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
	ctypes.c_int64,
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),
	ctypes.c_int64,
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
	ctypes.c_int64
]
engine_lib.run_hybrid_inference.restype = ctypes.c_int

engine_lib.cleanup_engine.argtypes = []
engine_lib.cleanup_engine.restype = None


# --- Global State ---
_TOKENIZER = None
_MAX_SEQ_LEN = 64
_DEFAULT_OUTPUT_DIM = 12
_PROPERTY_NAMES = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
_ENGINE_READY = False
_ENGINE_MODEL_PATH = None
_CONFIDENCE_CALIBRATION_CACHE = None
_PROPERTY_STATS_CACHE = None
_TRANSFORMER_MODEL = None
_EXPLAINABILITY_MODEL_PATH = None
_TOKENIZER_FALLBACK_WARNED = False
_MORGAN_FP_GENERATOR = None


# --- Feature Generation ---
def _one_hot_encoding(x, allowable_set):
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def _get_node_features(atom):
	return (_one_hot_encoding(atom.GetSymbol(), ['H', 'C', 'N', 'O', 'F', 'Unknown']) +
			_one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
			_one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1]) +
			_one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
			_one_hot_encoding(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 'Unknown']) +
			[atom.GetIsAromatic(), atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED])

def _get_edge_features(bond):
	return (_one_hot_encoding(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]) +
			[bond.GetIsConjugated()])


# --- Engine Management ---
def _resolve_model_path(model_path=None):
	if model_path:
		path = Path(model_path).expanduser().resolve()
		if not path.exists(): raise FileNotFoundError(f"ONNX model not found at: {path}")
		return path
	env_model = os.environ.get('HYBRID_ONNX_MODEL_PATH', '').strip()
	if env_model:
		path = Path(env_model).expanduser().resolve()
		if path.exists(): return path
	project_root = _LIB_DIR.parent.parent
	candidates = [project_root / 'GUI' / 'assets' / 'hybrid_model.onnx', project_root / 'GUI' / 'assets' / 'model.onnx', project_root / 'GUI' / 'core' / 'hybrid_model.onnx', project_root / 'hybrid_model.onnx', project_root / 'model.onnx']
	for candidate in candidates:
		if candidate.exists(): return candidate.resolve()
	raise FileNotFoundError("ONNX model file was not found. Set HYBRID_ONNX_MODEL_PATH or place model at GUI/assets/hybrid_model.onnx")

def init_hybrid_engine(model_path=None):
	global _ENGINE_READY, _ENGINE_MODEL_PATH
	resolved = _resolve_model_path(model_path)
	if _ENGINE_READY and _ENGINE_MODEL_PATH == resolved: return str(_ENGINE_MODEL_PATH)
	if _ENGINE_READY: cleanup_hybrid_engine()
	status = engine_lib.init_engine(str(resolved).encode('utf-8'))
	if status != 0: raise RuntimeError(f"Failed to initialize ONNX hybrid engine (status={status})")
	_ENGINE_READY = True
	_ENGINE_MODEL_PATH = resolved
	return str(_ENGINE_MODEL_PATH)

def cleanup_hybrid_engine():
	global _ENGINE_READY, _ENGINE_MODEL_PATH
	if _ENGINE_READY:
		engine_lib.cleanup_engine()
		_ENGINE_READY = False
		_ENGINE_MODEL_PATH = None


# --- Data Preparation ---
def build_graph_tensors_from_smiles(smiles, random_seed: Optional[int] = None):
    smiles = (smiles or '').strip()
    if not smiles: raise ValueError('SMILES is empty')
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f'Invalid SMILES: {smiles}')
    
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    
    conf = None
    try:
        params = AllChem.ETKDGv3()
        if random_seed is not None:
            params.randomSeed = int(random_seed)
        
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("RDKit EmbedMolecule failed.")
        
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=500) != 0:
            raise RuntimeError("RDKit MMFFOptimizeMolecule failed.")
            
        conf = mol.GetConformer()
    except Exception as e:
        raise RuntimeError(f"Failed to generate 3D conformer for SMILES '{smiles}': {e}")

    node_features = np.asarray([_get_node_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        topo = _get_edge_features(bond)
        dist = (conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length()
        edge_feature = topo + [dist]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([edge_feature, edge_feature])

    if not edge_indices:
        edge_indices, edge_attrs = [[0, 0]], [[0.0] * 6]

    edge_index = np.asarray(edge_indices, dtype=np.int64).T
    edge_attr = np.asarray(edge_attrs, dtype=np.float32)
    batch_index = np.zeros((node_features.shape[0],), dtype=np.int64)
    
    return (np.ascontiguousarray(node_features, dtype=np.float32),
            np.ascontiguousarray(edge_index, dtype=np.int64),
            np.ascontiguousarray(edge_attr, dtype=np.float32),
            np.ascontiguousarray(batch_index, dtype=np.int64))


# --- Single Inference ---
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


# --- Descaling and Confidence ---
def _compute_confidence_from_predictions(prediction_matrix):
	predictions = np.asarray(prediction_matrix, dtype=np.float32)
	if predictions.ndim != 2 or predictions.shape[0] < 1:
		raise ValueError('prediction_matrix must include at least one sample')
	means = np.mean(predictions, axis=0).astype(np.float32)
	stds = np.std(predictions, axis=0).astype(np.float32)
	cv_percent = ((stds / np.maximum(np.abs(means), 1e-6)) * 100.0).astype(np.float32)
	confidence_score = float(max(0.0, min(100.0, 100.0 * np.exp(-float(np.mean(cv_percent)) / 10.0))))
	return {'mean': means, 'std': stds, 'cv_percent': cv_percent, 'confidence_score': confidence_score}

def _resolve_property_stats_path():
	stats_env = os.environ.get('HYBRID_PROPERTY_STATS_PATH', '').strip()
	if stats_env:
		path = Path(stats_env).expanduser().resolve()
		if path.exists(): return path
	project_root = _LIB_DIR.parent.parent
	candidates = [project_root / 'models' / 'qm9_property_stats.json', project_root / 'GUI' / 'assets' / 'qm9_property_stats.json']
	for candidate in candidates:
		if candidate.exists(): return candidate.resolve()
	return None

def _load_property_stats():
	global _PROPERTY_STATS_CACHE
	if _PROPERTY_STATS_CACHE is not None: return _PROPERTY_STATS_CACHE
	path = _resolve_property_stats_path()
	if path is None: _PROPERTY_STATS_CACHE = {}; return _PROPERTY_STATS_CACHE
	try:
		with path.open('r', encoding='utf-8') as f: data = json.load(f)
		properties = data.get('properties')
		if not isinstance(properties, dict): _PROPERTY_STATS_CACHE = {}; return _PROPERTY_STATS_CACHE
		stats = {}
		for prop in _PROPERTY_NAMES:
			entry = properties.get(prop)
			if isinstance(entry, dict) and entry.get('mean') is not None and entry.get('std') is not None:
				stats[prop] = {'mean': float(entry['mean']), 'std': float(entry['std'])}
		if isinstance(data.get('atom_reference_energies'), dict):
			stats['_atom_reference_energies'] = data['atom_reference_energies']
		stats['_path'] = str(path)
		_PROPERTY_STATS_CACHE = stats
		return _PROPERTY_STATS_CACHE
	except Exception: _PROPERTY_STATS_CACHE = {}; return _PROPERTY_STATS_CACHE

def _descale_prediction_values(values, smiles: str | List[str]):
    arr = np.asarray(values, dtype=np.float32).copy()
    stats = _load_property_stats()
    if not stats: return arr
    
    is_batch = isinstance(smiles, list)
    
    # Step 1: Inverse scale all values using standardization stats
    for idx, prop in enumerate(_PROPERTY_NAMES):
        entry = stats.get(prop)
        if not isinstance(entry, dict): continue
        
        if arr.ndim == 1:
            arr[idx] = (arr[idx] * float(entry['std'])) + float(entry['mean'])
        elif arr.ndim == 2:
            arr[:, idx] = (arr[:, idx] * float(entry['std'])) + float(entry['mean'])

    # Step 2: For delta-trained properties, add the atomic reference energy back.
    atom_ref_payload = stats.get('_atom_reference_energies')
    if not (smiles and isinstance(atom_ref_payload, dict)): return arr

    smiles_list = smiles if is_batch else [smiles]
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if not mol: continue
        mol = Chem.AddHs(mol)
        
        for idx, prop in enumerate(_PROPERTY_NAMES):
            if prop in QM9_DELTA_TARGET_COLUMNS:
                correction = 0.0
                prop_refs = atom_ref_payload.get(prop, {})
                for atom in mol.GetAtoms():
                            val = prop_refs.get(atom.GetSymbol(), 0.0)
                            # Enforce negative sign: free atom energies are physically negative
                            correction -= abs(val)

                        # Recover Total Energy by adding the negative reference sum
                if arr.ndim == 1:
                    arr[idx] += correction
                else:
                    arr[i, idx] += correction
    return arr

def _descale_spread_values(values):
	arr = np.asarray(values, dtype=np.float32).copy()
	stats = _load_property_stats()
	if not stats: return arr
	for idx, prop in enumerate(_PROPERTY_NAMES):
		entry = stats.get(prop)
		if not isinstance(entry, dict): continue
		arr[idx] = arr[idx] * float(entry['std'])
	return arr

def _resolve_confidence_calibration_path():
	calibration_env = os.environ.get('HYBRID_CONFIDENCE_CALIBRATION_PATH', '').strip()
	if calibration_env:
		path = Path(calibration_env).expanduser().resolve()
		if path.exists(): return path
	project_root = _LIB_DIR.parent.parent
	default_path = project_root / 'GUI' / 'assets' / 'confidence_calibration.json'
	if default_path.exists(): return default_path.resolve()
	return None

def _load_confidence_calibration():
	global _CONFIDENCE_CALIBRATION_CACHE
	if _CONFIDENCE_CALIBRATION_CACHE is not None: return _CONFIDENCE_CALIBRATION_CACHE
	path = _resolve_confidence_calibration_path()
	if path is None: _CONFIDENCE_CALIBRATION_CACHE = {}; return _CONFIDENCE_CALIBRATION_CACHE
	try:
		with path.open('r', encoding='utf-8') as f: data = json.load(f)
		quantiles = data.get('quantiles')
		if not (isinstance(quantiles, list) and len(quantiles) == _DEFAULT_OUTPUT_DIM):
			_CONFIDENCE_CALIBRATION_CACHE = {}; return _CONFIDENCE_CALIBRATION_CACHE
		_CONFIDENCE_CALIBRATION_CACHE = {'path': str(path), 'method': str(data.get('method') or 'conformal_residual_quantile'), 'alpha': float(data.get('alpha', 0.1)), 'quantiles': np.asarray(quantiles, dtype=np.float32)}
		return _CONFIDENCE_CALIBRATION_CACHE
	except Exception: _CONFIDENCE_CALIBRATION_CACHE = {}; return _CONFIDENCE_CALIBRATION_CACHE

def _compute_prediction_intervals(mean_values, std_values):
	means, stds = np.asarray(mean_values, dtype=np.float32), np.asarray(std_values, dtype=np.float32)
	calibration = _load_confidence_calibration()
	if calibration and isinstance(calibration.get('quantiles'), np.ndarray):
		radius, method, alpha, path = np.maximum(calibration['quantiles'], 0.0).astype(np.float32), calibration.get('method', 'conformal_residual_quantile'), float(calibration.get('alpha', 0.1)), calibration.get('path')
	else:
		radius, method, alpha, path = (1.96 * stds).astype(np.float32), 'gaussian_std_fallback', 0.05, None
	return {'lower': (means - radius).astype(np.float32), 'upper': (means + radius).astype(np.float32), 'radius': radius, 'method': method, 'alpha': alpha, 'calibration_path': path}

def run_hybrid_regression_with_confidence(smiles, model_path=None, n_conformers=3, base_seed=17, apply_descaling=True):
    if n_conformers < 1: raise ValueError('n_conformers must be >= 1')
    init_hybrid_engine(model_path=model_path)
    predictions, warnings = [], []
    for i in range(n_conformers):
        try: 
            predictions.append(run_hybrid_regression(smiles, model_path=model_path, random_seed=int(base_seed + i)))
        except Exception as exc: 
            warnings.append(f'conformer_{i}_failed: {exc}')
    
    if not predictions: 
        error_log = "\n".join([str(w) for w in warnings])
        raise RuntimeError(f'All confidence conformer runs failed for SMILES: {smiles}. Captured errors:\n{error_log}')
    
    stats = _compute_confidence_from_predictions(np.vstack(predictions))
    intervals = _compute_prediction_intervals(stats['mean'], stats['std'])
    prediction_values, std_values, interval_lower, interval_upper, interval_radius = stats['mean'], stats['std'], intervals['lower'], intervals['upper'], intervals['radius']
    
    if apply_descaling:
        prediction_values = _descale_prediction_values(prediction_values, smiles=smiles)
        std_values = _descale_spread_values(std_values)
        interval_lower = _descale_prediction_values(interval_lower, smiles=smiles)
        interval_upper = _descale_prediction_values(interval_upper, smiles=smiles)
        interval_radius = _descale_spread_values(interval_radius)
    
    stats_source = _load_property_stats().get('_path') if apply_descaling else None
    return {'prediction': prediction_values, 'confidence': {'std': std_values.tolist(), 'cv_percent': stats['cv_percent'].tolist(), 'interval_lower': interval_lower.tolist(), 'interval_upper': interval_upper.tolist(), 'interval_radius': interval_radius.tolist(), 'interval_method': intervals['method'], 'interval_alpha': intervals['alpha'], 'interval_calibration_path': intervals['calibration_path'], 'descaled': bool(apply_descaling and stats_source), 'descaling_stats_path': stats_source, 'confidence_score': stats['confidence_score'], 'n_conformers_requested': int(n_conformers), 'n_conformers_used': int(len(predictions)), 'mode': 'conformer_variance', 'warnings': warnings}}


# --- Other Utility Functions ---
def _looks_like_hf_model_dir(path_obj):
	if not path_obj.is_dir(): return False
	return (path_obj / 'config.json').exists() and ((path_obj / 'model.safetensors').exists() or (path_obj / 'pytorch_model.bin').exists()) and ((path_obj / 'tokenizer.json').exists() or (path_obj / 'tokenizer_config.json').exists())

def _resolve_explainability_model_path():
	global _EXPLAINABILITY_MODEL_PATH
	if _EXPLAINABILITY_MODEL_PATH is not None: return _EXPLAINABILITY_MODEL_PATH
	env_path = os.environ.get('HYBRID_ATTENTION_MODEL_PATH', '').strip()
	if env_path:
		candidate = Path(env_path).expanduser().resolve()
		if _looks_like_hf_model_dir(candidate): _EXPLAINABILITY_MODEL_PATH = candidate; return _EXPLAINABILITY_MODEL_PATH
		raise FileNotFoundError(f"HYBRID_ATTENTION_MODEL_PATH is set but not a valid Hugging Face model directory: {candidate}")
	assets_dir = _LIB_DIR.parent / 'assets'
	for candidate in [assets_dir / 'fine_tuned_transformer', assets_dir / 'transformer_finetuned', assets_dir / 'chemberta_finetuned', assets_dir / 'transformer']:
		if _looks_like_hf_model_dir(candidate): _EXPLAINABILITY_MODEL_PATH = candidate.resolve(); return _EXPLAINABILITY_MODEL_PATH
	raise FileNotFoundError("Attention model not found in GUI/assets. Set HYBRID_ATTENTION_MODEL_PATH or place model in a default location.")

def _get_tokenizer():
	global _TOKENIZER, _TOKENIZER_FALLBACK_WARNED
	if _TOKENIZER is None:
		try:
			model_path = _resolve_explainability_model_path()
			_TOKENIZER = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
		except FileNotFoundError:
			if not _TOKENIZER_FALLBACK_WARNED:
				print('[Inference] Attention model/tokenizer not found. Using fallback tokenizer.')
				_TOKENIZER_FALLBACK_WARNED = True
			_TOKENIZER = 'fallback'
	return _TOKENIZER

def _encode_smiles(smiles):
    tokenizer = _get_tokenizer()
    if tokenizer == 'fallback' or not hasattr(tokenizer, 'batch_encode_plus'):
        tokens = [101] + [(ord(ch) % 255) + 1 for ch in (smiles or '')[:max(0, _MAX_SEQ_LEN - 2)]] + [102]
        pad_len = _MAX_SEQ_LEN - len(tokens)
        input_ids = [tokens + [0] * pad_len]
        attention_mask = [[1] * len(tokens) + [0] * pad_len]
        return np.ascontiguousarray(input_ids, dtype=np.int64), np.ascontiguousarray(attention_mask, dtype=np.int64)
    
    encoded = tokenizer(smiles, padding='max_length', truncation=True, max_length=_MAX_SEQ_LEN, return_tensors='np')
    return np.ascontiguousarray(encoded['input_ids'].astype(np.int64)), np.ascontiguousarray(encoded['attention_mask'].astype(np.int64))


# --- Qt Threads (for GUI) ---
class BatchInferenceThread(QThread):
    finished = Signal(list, list)
    error = Signal(str)
    def __init__(self, smiles_list, model_path=None, enable_confidence=True, n_conformers=3):
        super().__init__()
        self.smiles_list = [s.strip() for s in (smiles_list or []) if s and s.strip()]
        self.model_path, self.enable_confidence, self.n_conformers = model_path, bool(enable_confidence), int(n_conformers)
    def run(self):
        if not self.smiles_list: self.error.emit('No SMILES provided'); return
        results, failures = [], []
        try: init_hybrid_engine(model_path=self.model_path)
        except Exception as exc: self.error.emit(str(exc)); return
        for smiles in self.smiles_list:
            try:
                if self.enable_confidence:
                    result = run_hybrid_regression_with_confidence(smiles, model_path=self.model_path, n_conformers=self.n_conformers, apply_descaling=True)
                    results.append((smiles, result['prediction'].tolist(), result['confidence']))
                else:
                    pred = run_hybrid_regression(smiles, model_path=self.model_path)
                    results.append((smiles, _descale_prediction_values(pred, smiles=smiles).tolist(), None))
            except Exception as exc: failures.append((smiles, str(exc)))
        self.finished.emit(results, failures)

class InferenceThread(QThread):
	finished = Signal(np.ndarray)
	error = Signal(str)
	def __init__(self, smiles, **kwargs):
		super().__init__()
		self.smiles = (smiles or '').strip()
	def run(self):
		try:
			if not self.smiles: self.error.emit('SMILES input is required'); return
			prediction = run_hybrid_regression(self.smiles)
			self.finished.emit(_descale_prediction_values(prediction, smiles=self.smiles))
		except Exception as exc: self.error.emit(str(exc))