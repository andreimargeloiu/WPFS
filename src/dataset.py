import os
from _config import *

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchnmf.nmf import NMF
import scipy.io as spio
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_csv_data(path, labels_column=-1):
	"""
	Load a data file
	- path (str): path to csv_file
	- labels_column (int): indice of the column with labels
	"""
	Xy = pd.read_csv(path, index_col=0)
	X = Xy[Xy.columns[:labels_column]].to_numpy()
	y = Xy[Xy.columns[labels_column]].to_numpy()

	return X, y


def load_lung(drop_class_5=True):
	"""
	Labels in initial dataset:
	1    139
	2     17
	3     21
	4     20
	5      6

	We drop the class 5 because it has too little examples.
	"""
	data = spio.loadmat(os.path.join(DATA_DIR, 'lung.mat'))
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	
	# Drop samples of class 5
	X = X.drop(index=[156, 157, 158, 159, 160, 161])
	Y = Y.drop([156, 157, 158, 159, 160, 161])

	new_labels = {1:0, 2:1, 3:2, 4:3, 5:4}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_prostate():
	""""
	Labels in initial dataset:
	1    50
	2    52
	"""
	data = spio.loadmat(os.path.join(DATA_DIR, 'Prostate_GE.mat'))
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_toxicity():
	"""
	Labels in initial dataset:
	1    45
	2    45
	3    39
	4    42
	"""
	data = spio.loadmat(os.path.join(DATA_DIR, 'TOX_171.mat'))
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2, 4:3}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_cll():
	"""
	Labels in initial dataset:
	1    11
	2    49
	3    51
	"""
	data = spio.loadmat(os.path.join(DATA_DIR, 'CLL_SUB_111.mat'))
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_smk():
	"""
	Labels in initial dataset:
	1    90
	2    97
	"""
	data = spio.loadmat(os.path.join(DATA_DIR, 'SMK_CAN_187.mat'))
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_your_custom_dataset():
	return np.random.rand(100, 100), np.random.randint(0, 2, 100)

class CustomPytorchDataset(Dataset):
	def __init__(self, X, y, transform=None) -> None:
		# X, y are numpy
		super().__init__()

		self.X = torch.tensor(X, requires_grad=False)
		self.y = torch.tensor(y, requires_grad=False)
		self.transform = transform

	def __getitem__(self, index):
		x = self.X[index]
		y = self.y[index]
		if self.transform:
			x = self.transform(x)
			y = y.repeat(x.shape[0]) # replicate y to match the size of x

		return x, y

	def __len__(self):
		return len(self.X)


def standardize_data(X_train, X_valid, X_test, preprocessing_type):
	if preprocessing_type == 'z_score':
		scaler = StandardScaler()
	elif preprocessing_type == 'minmax':
		scaler = MinMaxScaler()
	elif preprocessing_type == 'raw':
		scaler = None
	else:
		raise Exception("preprocessing_type not supported")

	if scaler:
		X_train = scaler.fit_transform(X_train).astype(np.float32)
		X_valid = scaler.transform(X_valid).astype(np.float32)
		X_test = scaler.transform(X_test).astype(np.float32)

	return X_train, X_valid, X_test


def compute_stratified_splits(X, y, cv_folds, seed_kfold, split_id):
	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed_kfold)
	
	for i, (train_ids, test_ids) in enumerate(skf.split(X, y)):
		if i == split_id:
			return X[train_ids], X[test_ids], y[train_ids], y[test_ids]


###############    EMBEDDINGS     ###############

def compute_histogram_embedding(args, X, embedding_size):
	"""
	Compute embedding_matrix (D x M) based on the histograms. The function implements two methods:

	DietNetwork
	- Normalized bincounts for each SNP

	FsNet
	0. Input matrix NxD
	1. Z-score standardize each column (mean 0, std 1)
	2. Compute the histogram for every feature (with density = False)
	3. Multiply the histogram values with the bin mean

	:param (N x D) X: dataset, each row representing one sample
	:return np.ndarray (D x M) embedding_matrix: matrix where each row represents the embedding of one feature
	"""
	X = np.rot90(X)
	
	number_features = X.shape[0]
	embedding_matrix = np.zeros(shape=(number_features, embedding_size))

	for feature_id in range(number_features):
		feature = X[feature_id]

		hist_values, bin_edges = np.histogram(feature, bins=embedding_size) # like in FsNet
		bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
		embedding_matrix[feature_id] = np.multiply(hist_values, bin_centers)

	return embedding_matrix


def compute_nmf_embeddings(Xt, rank):
	"""
	Note: torchnmf computes V = H W^T instead of the standard formula V = W H

	Input
	- V (D x N)
	- rank of NMF

	Returns
	- H (D x r) (torch.Parameter with requires_grad=True), where each row represents one gene embedding 
	"""
	print("Approximating V = H W.T")
	print(f"Input V has shape {Xt.shape}")
	assert type(Xt)==torch.Tensor
	assert Xt.shape[0] > Xt.shape[1]

	nmf = NMF(Xt.shape, rank=rank).cuda()
	nmf.fit(Xt.cuda(), beta=2, max_iter=1000, verbose=True) # beta=2 coresponds to the Frobenius norm, which is equivalent to an additive Gaussian noise model

	print(f"H has shape {nmf.H.shape}")
	print(f"W.T has shape {nmf.W.T.shape}")

	return nmf.H, nmf.W


def compute_svd_embeddings(X, rank=None):
	"""
	- X (N x D)
	- rank (int): rank of the approximation (i.e., size of the embedding)
	"""
	assert type(X)==torch.Tensor
	assert X.shape[0] < X.shape[1]

	U, S, Vh = torch.linalg.svd(X, full_matrices=False)

	V = Vh.T

	if rank:
		S = S[:rank]
		V = V[:rank]

	return V, S


###############    DATASETS     ###############

class DatasetModule(pl.LightningDataModule):
	def __init__(self, args, X_train, y_train, X_valid, y_valid, X_test, y_test):
		super().__init__()
		self.args = args

		# Standardize data
		self.X_train_raw = X_train
		self.X_valid_raw = X_valid
		self.X_test_raw = X_test

		X_train, X_valid, X_test = standardize_data(X_train, X_valid, X_test, args.patient_preprocessing)
		
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.X_test = X_test
		self.y_test = y_test

		self.train_dataset = CustomPytorchDataset(X_train, y_train)
		self.valid_dataset = CustomPytorchDataset(X_valid, y_valid)
		self.test_dataset = CustomPytorchDataset(X_test, y_test)

		self.args.train_size = X_train.shape[0]
		self.args.valid_size = X_valid.shape[0]
		self.args.test_size = X_test.shape[0]

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

	def val_dataloader(self):
		return DataLoader(self.valid_dataset, batch_size=self.args.batch_size,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)


	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.args.batch_size,
							num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

	def get_embedding_matrix(self, embedding_type, embedding_size):
		"""
		Return matrix D x M

		Use a the shared hyper-parameter self.args.embedding_preprocessing.
		"""
		if embedding_type == None:
			return None
		else:
			if embedding_size == None:
				raise Exception()

		# Preprocess the data for the embeddings
		if self.args.embedding_preprocessing == 'raw':
			X_for_embeddings = self.X_train_raw
		elif self.args.embedding_preprocessing == 'z_score':
			X_for_embeddings = StandardScaler().fit_transform(self.X_train_raw)
		elif self.args.embedding_preprocessing == 'minmax':
			X_for_embeddings = MinMaxScaler().fit_transform(self.X_train_raw)
		else:
			raise Exception("embedding_preprocessing not supported")

		if embedding_type == 'histogram':
			"""
			Embedding similar to FsNet
			"""
			embedding_matrix = compute_histogram_embedding(self.args, X_for_embeddings, embedding_size)
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='feature_values':
			"""
			A gene's embedding are its patients gene expressions.
			"""
			embedding_matrix = np.rot90(X_for_embeddings)[:, :embedding_size]
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='svd':
			# Vh.T (4160 x rank) contains the gene embeddings on each row
			U, S, Vh = torch.linalg.svd(torch.tensor(X_for_embeddings, dtype=torch.float32), full_matrices=False) 
			
			Vh.T.requires_grad = False
			return Vh.T[:, :embedding_size].type(torch.float32)
		elif embedding_type=='nmf':
			H, _ = compute_nmf_embeddings(torch.tensor(X_for_embeddings).T, rank=embedding_size)
			H_data = H.data
			H_data.requires_grad = False
			return H_data.type(torch.float32)
		else:
			raise Exception("Invalid embedding type")


def create_data_module(args):
	dataset = args.dataset

	if dataset in ['metabric-pam50', 'metabric-dr', 'tcga-2ysurvival', 'tcga-tumor-grade']:
		dataset_size = 200
		if dataset=='metabric-pam50':
			dataset_path = os.path.join(DATA_DIR, f'Metabric_samples/metabric_pam50_train_{dataset_size}.csv')
		elif dataset=='metabric-dr':
			dataset_path = os.path.join(DATA_DIR, f'Metabric_samples/metabric_DR_train_{dataset_size}.csv')
		elif dataset=='tcga-2ysurvival':
			dataset_path = os.path.join(DATA_DIR, f'TCGA_samples/tcga_2ysurvival_train_{dataset_size}.csv')
		elif dataset=='tcga-tumor-grade':
			dataset_path = os.path.join(DATA_DIR, f'TCGA_samples/tcga_tumor_grade_train_{dataset_size}.csv')

		X, y = load_csv_data(dataset_path)

	else:
		if dataset=='lung':
			X, y = load_lung()
		elif dataset=='toxicity':
			X, y = load_toxicity()
		elif dataset=='prostate':
			X, y = load_prostate()
		elif dataset=='cll':
			X, y = load_cll()
		elif dataset=='smk':
			X, y = load_smk()
		elif dataset=='your_custom_dataset':
			X, y = load_your_custom_dataset()
		else:
			raise Exception("Dataset not supported")

	args.dataset_size = X.shape[0]

	data_module = create_datamodule_with_cross_validation(args, X, y)


	#### Compute classification loss weights
	if args.class_weight=='balanced':
		args.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data_module.y_train), y=data_module.y_train)
	elif args.class_weight=='standard':
		args.class_weights = compute_class_weight(class_weight=None, classes=np.unique(data_module.y_train), y=data_module.y_train)
	args.class_weights = args.class_weights.astype(np.float32)
	print(f"Weights for the classification loss: {args.class_weights}")

	return data_module



def create_datamodule_with_cross_validation(args, X, y):
	"""
	Split X, y to be suitable for k-fold cross-validation.
	It uses args.test_split to create the train, valid and test stratified datasets.
	"""
	if type(X)==pd.DataFrame:
		X = X.to_numpy()
	if type(y)==pd.Series:
		y = y.to_numpy()

	args.num_features = X.shape[1]
	args.num_classes = len(np.unique(y))

	assert type(X)==np.ndarray
	assert type(y)==np.ndarray

	# seed_kfold defines the data shuflling
	# 	--- args.seed_kfold = args.repeat_id ---> args.repeat_id defines the data sluffling
	
	# args.test_split defines the fold to pick
	X_train_and_valid, X_test, y_train_and_valid, y_test = compute_stratified_splits(
		X, y, cv_folds=args.cv_folds, seed_kfold=args.seed_kfold, split_id=args.test_split)

	# randomly pick a validation set from the training_and_val data
	X_train, X_valid, y_train, y_valid = train_test_split(
		X_train_and_valid, y_train_and_valid,
		test_size = args.valid_percentage,
		random_state = args.seed_validation,
		stratify = y_train_and_valid
	)
	
	print(f"Train size: {X_train.shape[0]}\n")
	print(f"Valid size: {X_valid.shape[0]}\n")
	print(f"Test size: {X_test.shape[0]}\n")

	assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == X.shape[0]
	assert set(y_train).union(set(y_valid)).union(set(y_test)) == set(y)

	if args.train_on_full_data:
		# Train on the entire training set (train + validation)
		# Make validation and test sets the same
		return DatasetModule(args, X_train_and_valid, y_train_and_valid, X_test, y_test, X_test, y_test)
	else:
		return DatasetModule(args, X_train, y_train, X_valid, y_valid, X_test, y_test)