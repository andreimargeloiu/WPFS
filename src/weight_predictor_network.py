from torch import nn

class WeightPredictorNetwork(nn.Module):
	def __init__(self, args, embedding_matrix):
		"""
		WPN outputs a "virtual" weight matrix W

		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		print(f"Initializing WPN with embedding_matrix of size {embedding_matrix.size()}")
		
		self.args = args
		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix


		layers = []
		prev_dimension = args.wpn_embedding_size
		for i, dim in enumerate(args.wpn_layers):
			if i == len(args.wpn_layers)-1: # last layer
				layer = nn.Linear(prev_dimension, dim)
				nn.init.uniform_(layer.weight, -0.01, 0.01) 	# same initialization as in the Diet Network paper official implementation
				layers.append(layer)
				layers.append(nn.Tanh())
			else:
				layer = nn.Linear(prev_dimension, dim)
				nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
				layers.append(layer)
				layers.append(nn.LeakyReLU())
				layers.append(nn.BatchNorm1d(dim))
				layers.append(nn.Dropout(args.dropout_rate))
				
			prev_dimension = dim

		self.wpn = nn.Sequential(*layers)

	def forward(self):
		W = self.wpn(self.embedding_matrix) 	# W has size (D x K)
		
		return W.T # size K x D