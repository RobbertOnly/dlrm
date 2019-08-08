
import luigi
import torch

class data_config(luigi.Config):
	"""
	Configuation for dataset

	"""
	data_size = 1
	num_batches = 0				
	data_generation = 'random'
	
	data_set = 'kaggle'
	raw_data_file = ''
	processed_data_file = ''
	data_randomize = 'total'

	num_indices_per_lookup = 10
	num_indices_per_lookup_fixed = False
	round_targets = False



class training_config(luigi.Config):
	"""
	Configuration for training network

	"""

	#Activation and loss
	activation_function = 'relu'
	loss_function = 'mse'
	loss_threshold = 0.0


	#Data type and size
	mini_batch_size = 1
	num_epochs = 1
	learning_rate = 0.01
	
	
	sync_dense_params = True


class prep_config(luigi.Config):
	"""
	Configuration for preprocessing

	"""

class net_config(luigi.Config):
	"""
	Configuration for neural network

	"""
	arch_sparse_feature_size = 2
	arch_embedding_size = '4-3-2'
	arch_mlp_bot = '4-3-2'
	arch_mlp_top = '4-2-1'
	arch_interaction_op = 'dot'
	arch_interaction_itself = False
	inference_only = False


class process_config(luigi.Config):
	"""
	Configuration for software process

	"""
	use_gpu = False
	rand_seed = 123
	print_precision = 5
	experiment_name = 'dlrm_test_experiment'

class log_config(luigi.Config):
	"""
	Parameters for debugging, profiling, and logging

	"""
	print_freq = 1
	test_freq = -1
	print_time = False 
	debug_mode = False 
	enable_profiling = False 
	plot_compute_graph = False

	#Data trace informations
	data_trace_file = './input/dict_emb_j.log'
	data_trace_enable_padding = False



class io_config(luigi.Config):
	"""
	Configuration for input and output features

	"""
	save_onnx = False
	save_model = ""
	load_model = ""



# num_batches = 0
# mini_batch_size = 1
# round_targets = 
# num_indices_per_lookup = 
# num_indices_per_lookup_fixed = 
# m_den = 
# ln_emb = 
# data_trace_file = 
# data_trace_enable_padding = 
