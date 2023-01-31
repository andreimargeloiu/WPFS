import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from sparsity_network import SparsityNetwork
from weight_predictor_network import WeightPredictorNetwork



def get_labels_lists(outputs):
	all_y_true, all_y_pred = [], []
	for output in outputs:
		all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
		all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

	return all_y_true, all_y_pred


def compute_all_metrics(args, y_true, y_pred):
	metrics = {}
	metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
	
	return metrics


def detach_tensors(tensors):
	"""
	Detach losses 
	"""
	if type(tensors)==list:
		detached_tensors = list()
		for tensor in tensors:
			detach_tensors.append(tensor.detach())
	elif type(tensors)==dict:
		detached_tensors = dict()
		for key, tensor in tensors.items():
			detached_tensors[key] = tensor.detach()
	else:
		raise Exception("tensors must be a list or a dict")
	
	return detached_tensors


def reshape_batch(batch):
	"""
	When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
	
	This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
	"""
	x, y = batch
	x = x.reshape(-1, x.shape[-1])
	y = y.reshape(-1)

	return x, y


def create_model(args, data_module=None):
	"""
	Function to create the model. It firstly creates the components (e.g., FeatureExtractor)
	and then assambles them
	"""
	pl.seed_everything(args.seed_model_init, workers=True)

	### create embedding matrices
	wpn_embedding_matrix = data_module.get_embedding_matrix(args.wpn_embedding_type, args.wpn_embedding_size)
	if args.wpn_embedding_type==args.sparsity_gene_embedding_type and args.wpn_embedding_size==args.sparsity_gene_embedding_size:
		spn_embedding_matrix = wpn_embedding_matrix
	else:
		spn_embedding_matrix = data_module.get_embedding_matrix(args.sparsity_gene_embedding_type, args.sparsity_gene_embedding_size)

	#### Create model instance
	if args.model == 'mlp':
		first_layer = FirstLinearLayer(args, is_diet_layer=False, sparsity=None)

		return GeneralNeuralNetwork(args, first_layer, None)
	
	if args.model == 'dietnetworks':
		first_layer = FirstLinearLayer(args, is_diet_layer=True, sparsity=None, wpn_embedding_matrix=wpn_embedding_matrix)

		decoder = Decoder(args, WeightPredictorNetwork(args, wpn_embedding_matrix))

		return GeneralNeuralNetwork(args, first_layer, decoder)
	elif args.model == 'wpfs':
		assert args.feature_extractor_dims[0] == args.wpn_layers[-1], "The output size of WPN must be the same as the first layer of the feature extractor."
		assert data_module != None, "You must specify a data_module to compute the feature embeddings"
		
		first_layer = FirstLinearLayer(args, is_diet_layer=True, sparsity=True,
						wpn_embedding_matrix=wpn_embedding_matrix, spn_embedding_matrix=spn_embedding_matrix)

		return GeneralNeuralNetwork(args, first_layer, None)
	elif args.model=='cae': # Supervised Autoencoder
		concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0])

		return GeneralNeuralNetwork(args, concrete_layer, None)
	elif args.model=='fsnet':
		concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0],
				is_diet_layer=True, wpn_embedding_matrix=wpn_embedding_matrix)

		decoder = Decoder(args, WeightPredictorNetwork(args, wpn_embedding_matrix))

		return GeneralNeuralNetwork(args, concrete_layer, decoder)
	else:
		raise Exception("Model not implemented")


def create_linear_layers(args, layer_sizes, layers_for_hidden_representation):
	"""
	Args
	- layer_sizes: list of the sizes of the sizes of the linear layers
	- layers_for_hidden_representation: number of layers of the first part of the encoder (used to output the input for the decoder)

	Returns
	Two lists of Pytorch Modules (e.g., Linear, BatchNorm1d, Dropout)
	- encoder_first_part
	- encoder_second_part
	"""
	encoder_first_part = []
	encoder_second_part = []
	for i, (dim_prev, dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
		if i < layers_for_hidden_representation:					# first part of the encoder
			encoder_first_part.append(nn.Linear(dim_prev, dim))
			encoder_first_part.append(nn.LeakyReLU())
			encoder_first_part.append(nn.BatchNorm1d(dim))
			encoder_first_part.append(nn.Dropout(args.dropout_rate))
		else:														# second part of the encoder
			encoder_second_part.append(nn.Linear(dim_prev, dim))
			encoder_second_part.append(nn.LeakyReLU())
			encoder_second_part.append(nn.BatchNorm1d(dim))
			encoder_second_part.append(nn.Dropout(args.dropout_rate))
		
	return encoder_first_part, encoder_second_part


class FirstLinearLayer(nn.Module):
	"""
	First linear layer (with activation, batchnorm and dropout), with the ability to include:
	- diet layer (i.e., there's a weight predictor network which predicts the weight matrix)
	- sparsity network (i.e., there's a sparsity network which outputs sparsity weights)
	"""

	def __init__(self, args, is_diet_layer, sparsity, wpn_embedding_matrix=None, spn_embedding_matrix=None):
		"""
		If is_diet_layer==None and sparsity==None, this layers acts as a standard linear layer
		"""
		super().__init__()

		self.args = args
		self.is_diet_layer = is_diet_layer
		self.sparsity = sparsity

		# DIET LAYER
		if is_diet_layer:
			# if diet layer, then initialize a weight predictor network
			self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
		else:
			# standard linear layer
			self.weights_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features))
			nn.init.kaiming_normal_(self.weights_first_layer, a=0.01, mode='fan_out', nonlinearity='leaky_relu')

		# auxiliary layer after the matrix multiplication
		self.bias_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0]))
		self.layers_after_matrix_multiplication = nn.Sequential(*[
			nn.LeakyReLU(),
			nn.BatchNorm1d(args.feature_extractor_dims[0]),
			nn.Dropout(args.dropout_rate)
		])

		# SPARSITY REGULARIZATION for the first layer
		if sparsity:
			print("Creating Sparsity network")
			self.sparsity_model = SparsityNetwork(args, spn_embedding_matrix)
		else:
			self.sparsity_model = None

	def forward(self, x):
		"""
		Input:
			x: (batch_size x num_features)
		"""
		# first layer
		W = self.wpn() if self.is_diet_layer else self.weights_first_layer # W has size (K x D)
		
		if self.sparsity_model==None:
			all_sparsity_weights = None

			hidden_rep = F.linear(x, W, self.bias_first_layer)
		
		else:
			all_sparsity_weights = self.sparsity_model() 	# Tensor (D, )
			assert all_sparsity_weights.shape[0]==self.args.num_features and len(all_sparsity_weights.shape)==1
			W = torch.matmul(W, torch.diag(all_sparsity_weights))

			hidden_rep = F.linear(x, W, self.bias_first_layer)
			
		return self.layers_after_matrix_multiplication(hidden_rep), all_sparsity_weights



class ConcreteLayer(nn.Module):
	"""
	Implementation of a concrete layer from paper "Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"
	"""

	def __init__(self, args, input_dim, output_dim, is_diet_layer=False, wpn_embedding_matrix=None):
		"""
		- input_dim (int): dimension of the input
		- output_dim (int): number of neurons in the layer
		"""
		super().__init__()
		self.args = args
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.temp_start = 10
		self.temp_end = 0.01
		# the iteration is used in annealing the temperature
		# 	it's increased with every call to sample during training
		self.current_iteration = 0 
		self.anneal_iterations = args.concrete_anneal_iterations # maximum number of iterations for the temperature optimization

		self.is_diet_layer = is_diet_layer
		if is_diet_layer:
			# if diet layer, then initialize a weight predictor matrix
			assert wpn_embedding_matrix is not None
			self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
		else:
			# alphas (output_dim x input_dim) - learnable parameters for each neuron
			# alphas[i] = parameters of neuron i
			self.alphas = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=True)
			torch.nn.init.xavier_normal_(self.alphas, gain=1) # Glorot normalization, following the original CAE implementation
		
	def get_temperature(self):
		# compute temperature		
		if self.current_iteration >= self.anneal_iterations:
			return self.temp_end
		else:
			return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)

	def sample(self):
		"""
		Sample from the concrete distribution.
		"""
		# Increase the iteration counter during training
		if self.training:
			self.current_iteration += 1

		temperature = self.get_temperature()

		alphas = self.wpn() if self.is_diet_layer else self.alphas # alphas is a K x D matrix

		# sample from the concrete distribution
		if self.training:
			samples = F.gumbel_softmax(alphas, tau=temperature, hard=False) # size K x D
			assert samples.shape == (self.output_dim, self.input_dim)
		else: 			# sample using argmax
			index_max_alphas = torch.argmax(alphas, dim=1) # size K
			samples = torch.zeros(self.output_dim, self.input_dim).cuda()
			samples[torch.arange(self.output_dim), index_max_alphas] = 1.

		return samples

	def forward(self, x):
		"""
		- x (batch_size x input_dim)
		"""
		mask = self.sample()   	# size (number_neurons x input_dim)
		x = torch.matmul(x, mask.T) 		# size (batch_size, number_neurons)
		return x, None # return additional None for compatibility


class Decoder(nn.Module):
	def __init__(self, args, wpn):
		super().__init__()
		assert wpn!=None, "The decoder is used only with a WPN (because it's only used within the DietNetwork)"

		self.wpn = wpn
		self.bias = nn.Parameter(torch.zeros(args.num_features,))

	def forward(self, hidden_rep):
		W = self.wpn().T # W has size D x K

		return F.linear(hidden_rep, W, self.bias)


class GeneralNeuralNetwork(pl.LightningModule):
	def __init__(self, args, first_layer, decoder):
		"""
		General neural network that can be instantiated as WPFS, DietNetwork, FsNet or CAE

		:param args: arguments from the command line
		:param first_layer: first layer of the network (it can be a Linear layer, a Concrete layer, or a DietLayer)
		:param decoder: decoder of the network (optional, used only for DietNetworks and FsNet)
		"""
		super().__init__()

		self.args = args
		self.log_test_key = None
		self.learning_rate = args.lr

		self.first_layer = first_layer
		encoder_first_layers, encoder_second_layers = create_linear_layers(
			args, args.feature_extractor_dims, args.layers_for_hidden_representation-1) # the -1 in (args.layers_for_hidden_representation - 1) is because we don't consider the first layer

		self.encoder_first_layers = nn.Sequential(*encoder_first_layers)
		self.encoder_second_layers = nn.Sequential(*encoder_second_layers)

		self.classification_layer = nn.Linear(args.feature_extractor_dims[-1], args.num_classes)
		self.decoder = decoder

	def forward(self, x):
		x, sparsity_weights = self.first_layer(x)			   # pass through first layer

		x = self.encoder_first_layers(x)					   # pass throught the first part of the following layers
		x_hat = self.decoder(x) if self.decoder else None      # reconstruction

		x = self.encoder_second_layers(x)
		y_hat = self.classification_layer(x)           		   # classification, returns logits
		
		return y_hat, x_hat, sparsity_weights

	def compute_loss(self, y_true, y_hat, x, x_hat, sparsity_weights):
		losses = {}
		losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))
		losses['reconstruction'] = self.args.gamma * F.mse_loss(x_hat, x, reduction='mean') if self.decoder else torch.zeros(1, device=self.device)

		### sparsity loss
		if sparsity_weights is None:
			losses['sparsity'] = torch.tensor(0., device=self.device)
		else:
			losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * torch.sum(sparsity_weights)

		losses['total'] = losses['cross_entropy'] + losses['reconstruction'] + losses['sparsity']
		
		return losses

	def log_losses(self, losses, key, dataloader_name=""):
		self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
		self.log(f"{key}/reconstruction_loss{dataloader_name}", losses['reconstruction'].item())
		self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item())
		self.log(f"{key}/sparsity_loss{dataloader_name}", losses['sparsity'].item())

	def log_epoch_metrics(self, outputs, key, dataloader_name=""):
		y_true, y_pred = get_labels_lists(outputs)
		self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred))



	def training_step(self, batch, batch_idx):
		x, y_true = batch

		y_hat, x_hat, sparsity_weights = self.forward(x)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		self.log_losses(losses, key='train')

		return {
			'loss': losses['total'],
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}

	def training_epoch_end(self, outputs):
		self.log_epoch_metrics(outputs, 'train')

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		"""
		- dataloader_idx (int) tells which dataloader is the `batch` coming from
		"""
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)
	
		return {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}

	def validation_epoch_end(self, outputs):
		losses = {
			'total': np.mean([output['losses']['total'].item() for output in outputs]),
			'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
			'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs]),
			'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs])
		}

		self.log_losses(losses, key='valid')
		self.log_epoch_metrics(outputs, key='valid')


	def test_step(self, batch, batch_idx, dataloader_idx=0):
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x)
		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		return {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1),
			'y_hat': y_hat.detach().cpu().numpy(),
		}

	def test_epoch_end(self, outputs):
		### Save losses
		losses = {
			'total': np.mean([output['losses']['total'].item() for output in outputs]),
			'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
			'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs]),
			'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs])
		}
		self.log_losses(losses, key=self.log_test_key)
		self.log_epoch_metrics(outputs, self.log_test_key)


	def configure_optimizers(self):
		params = self.parameters()

		optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])

		if self.args.lr_scheduler == None:
			return optimizer
		else:
			if self.args.lr_scheduler == 'cosine_warm_restart':
				lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
					T_0 = self.args.cosine_warm_restart_t_0,
					eta_min = self.args.cosine_warm_restart_eta_min,
					verbose=True)
			elif self.args.lr_scheduler == 'lambda':
				def scheduler(epoch):
					if epoch < 500:
						return 0.995 ** epoch
					else:
						return 0.1

				lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
					optimizer,
					scheduler)
			else:
				raise Exception()

			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': lr_scheduler,
					'monitor': 'valid/cross_entropy_loss',
					'interval': 'step',
					'frequency': self.args.val_check_interval,
					'name': 'lr_scheduler'
				}
			}