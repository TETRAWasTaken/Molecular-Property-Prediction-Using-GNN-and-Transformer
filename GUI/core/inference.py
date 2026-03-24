import ctypes
import sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import Signal, QThread
from transformers import AutoTokenizer

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

engine_lib.run_inference.argtypes = [
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # node_features
	ctypes.c_int,                                                             # num_nodes
	ctypes.c_int,                                                             # node_feat_dim
	np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),    # edge_indices
	ctypes.c_int,                                                             # num_edges
	ctypes.c_char_p,                                                          # smiles
	np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # output_properties
	ctypes.c_int                                                              # output_dim
]
engine_lib.run_inference.restype = ctypes.c_int

engine_lib.cleanup_engine.argtypes = []
engine_lib.cleanup_engine.restype = None


_TOKENIZER = None
_TOKENIZER_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
_MAX_SEQ_LEN = 64


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
		if not self.smiles:
			self.error.emit('SMILES input is required for hybrid inference')
			return

		# Prepare an empty numpy array to receive the C output
		output_properties = np.zeros(self.output_dim, dtype=np.float32)

		num_nodes, node_feat_dim = self.node_features.shape
		_, num_edges = self.edge_indices.shape

		if self.edge_attr is None:
			# Fallback to a neutral edge feature tensor if caller does not provide bond features.
			self.edge_attr = np.zeros((num_edges, 1), dtype=np.float32)
		edge_feat_dim = self.edge_attr.shape[1]

		if self.batch_index is None:
			self.batch_index = np.zeros((num_nodes,), dtype=np.int64)

		input_ids, attention_mask = _encode_smiles(self.smiles)
		seq_len = input_ids.shape[1]

		status = engine_lib.run_hybrid_inference(
			self.node_features,
			num_nodes,
			node_feat_dim,
			self.edge_indices,
			num_edges,
			self.edge_attr,
			edge_feat_dim,
			self.batch_index,
			input_ids,
			seq_len,
			attention_mask,
			output_properties,
			self.output_dim
		)
		
		if status == 0:
			self.finished.emit(output_properties)
		else:
			self.error.emit(f'C Engine Inference Failed (status={status})')