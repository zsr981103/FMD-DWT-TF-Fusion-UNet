import numpy as np
from typing import Tuple, Optional, Union

try:
	import torch
	_TORCH_AVAILABLE = True
except Exception:
	_TORCH_AVAILABLE = False

from FMD import FMD

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _validate_input_shape(x: np.ndarray) -> None:
	if x.ndim != 3 or x.shape[1] != 1:
		raise ValueError(f"Expected input shape (B, 1, S), got {tuple(x.shape)}")


def fmd_decompose_batch_numpy(
		signal: np.ndarray,
		fs: int,
		filter_size: int,
		cut_num: int,
		mode_num: int,
		max_iter_num: int,
) -> np.ndarray:
	"""
	Batch FMD decomposition in NumPy.

	Parameters:
	- signal: np.ndarray of shape (B, 1, S)
	- fs: sampling rate (int)
	- filter_size: FIR filter length used in FMD
	- cut_num: initial subbands count in FMD
	- mode_num: desired number of output modes (N)
	- max_iter_num: maximum iterations for FMD

	Returns:
	- modes: np.ndarray of shape (B, N, S)
	"""
	if not isinstance(signal, np.ndarray):
		raise TypeError("signal must be a numpy.ndarray for fmd_decompose_batch_numpy")
	_validate_input_shape(signal)

	batch_size = signal.shape[0]
	S = signal.shape[2]
	output_modes: Optional[list[np.ndarray]] = []

	for b in range(batch_size):
		x = signal[b, 0, :].astype(np.float64)  # (S,)
		modes_bt = FMD(fs, x, filter_size, cut_num, mode_num, max_iter_num)  # expected (S, n) or (n, S)

		if modes_bt.ndim != 2:
			raise RuntimeError(f"Unexpected modes ndim from FMD: {modes_bt.ndim}, expected 2")

		# Normalize to per-sample shape (N, S)
		if modes_bt.shape[0] == S:
			# (S, n) -> transpose
			per_sample = modes_bt.T  # (n, S)
		elif modes_bt.shape[1] == S:
			# (n, S)
			per_sample = modes_bt
		else:
			raise RuntimeError(
				f"Unexpected modes shape from FMD: {modes_bt.shape}, neither dimension matches S={S}"
			)

		output_modes.append(per_sample)  # (N, S)

	# Align modes across batch: ensure each has same N
	n_list = [m.shape[0] for m in output_modes]
	min_n = min(n_list)
	if min_n != max(n_list):
		# If different, truncate to min to form a proper batch
		output_modes = [m[:min_n] for m in output_modes]

	stacked = np.stack(output_modes, axis=0)  # (B, N, S)
	return stacked.astype(np.float32)


def fmd_decompose_batch(
		signal: ArrayLike,
		fs: int,
		filter_size: int,
		cut_num: int,
		mode_num: int,
		max_iter_num: int,
) -> ArrayLike:
	"""
	Batch FMD decomposition with seamless NumPy/PyTorch support.

	- If `signal` is np.ndarray: returns np.ndarray (B, N, S)
	- If `signal` is torch.Tensor: returns torch.Tensor (B, N, S) on the same device/dtype
	"""
	if _TORCH_AVAILABLE and isinstance(signal, torch.Tensor):
		device = signal.device
		dtype = signal.dtype
		np_signal = signal.detach().cpu().numpy()
		modes_np = fmd_decompose_batch_numpy(np_signal, fs, filter_size, cut_num, mode_num, max_iter_num)
		return torch.from_numpy(modes_np).to(device=device, dtype=dtype)
	elif isinstance(signal, np.ndarray):
		return fmd_decompose_batch_numpy(signal, fs, filter_size, cut_num, mode_num, max_iter_num)
	else:
		raise TypeError("signal must be either np.ndarray or torch.Tensor with shape (B,1,S)")

def FMD_data(npy_data):
	fake = npy_data
	fs = 125
	filtersize = 30
	cutnum = 7
	modenum = 3
	maxiternum = 20
	modes = fmd_decompose_batch_numpy(fake, fs, filtersize, cutnum, modenum, maxiternum)
	print("Output shape (numpy):", modes.shape)  # (B, N, S)

	# if _TORCH_AVAILABLE:
	# 	fake_t = torch.from_numpy(fake)
	# 	modes = fmd_decompose_batch(fake_t, fs, filtersize, cutnum, modenum, maxiternum)
	# 	print("Output shape (torch):", tuple(modes.shape))
	return modes


if __name__ == "__main__":
	# Minimal example using random data
	B = 2
	S = 256
	fs = 125
	filtersize = 30
	cutnum = 7
	modenum = 5
	maxiternum = 20

	fake = np.random.randn(B, 1, S).astype(np.float32)
	modes = fmd_decompose_batch_numpy(fake, fs, filtersize, cutnum, modenum, maxiternum)
	print("Output shape (numpy):", modes.shape)  # (B, N, S)

	if _TORCH_AVAILABLE:
		fake_t = torch.from_numpy(fake)
		modes_t = fmd_decompose_batch(fake_t, fs, filtersize, cutnum, modenum, maxiternum)
		print("Output shape (torch):", tuple(modes_t.shape))
