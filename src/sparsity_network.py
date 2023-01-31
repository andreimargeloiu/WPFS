import torch
import torch.nn as nn


class SparsityNetwork(nn.Module):
	"""
	Sparsity network
	- same architecture as WPN
	- input: gene embedding matrix (D x M)
	- output: 1 neuron, sigmoid activation function (which will get multiplied by the weights associated with the gene)
	"""

	def __init__(self, args, embedding_matrix):
		"""
		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()

		self.args = args
		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix

		layers = []
		dim_prev = args.sparsity_gene_embedding_size # input for global sparsity: gene embedding
		for _, dim in enumerate(args.wpn_layers):
			layers.append(nn.Linear(dim_prev, dim))
			layers.append(nn.LeakyReLU())
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.Dropout(args.dropout_rate))

			dim_prev = dim
		
		layers.append(nn.Linear(dim, 1))
		self.network = nn.Sequential(*layers)

	def forward(self):
		"""
		Input:
		- input: None

		Returns:
		- Tensor of sigmoid values (D)
		"""
		out = self.network(self.embedding_matrix) # (D, 1)
		out = torch.sigmoid(out)
		return torch.squeeze(out, dim=1) 		  # (D)