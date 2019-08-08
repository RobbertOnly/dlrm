import d6tflow
import luigi
from luigi.util import inherits

import numpy as np
import torch
from torchviz import make_dot
import torch.nn.functional as Functional
from torch.nn.parameter import Parameter

from dlrm_utils_pytorch import time_wrap, dlrm_wrap, loss_fn_wrap
import dlrm_data_pytorch as dp
from dlrm_d6t_dlrm_pytorch import DLRM_Net

from flow_cfg import data_config, training_config, prep_config,\
                     net_config, process_config, log_config, io_config



class TaskSetupProcessor(d6tflow.tasks.TaskCache):

    rand_seed = luigi.IntParameter(default = process_config().rand_seed)
    print_precision = luigi.IntParameter(default = process_config().rand_seed)
    use_gpu = luigi.BoolParameter(default = process_config().use_gpu)

    def run(self):
        ### some basic setup ###
        processor = None
        np.random.seed(self.rand_seed)
        np.set_printoptions(precision=self.print_precision)
        torch.set_printoptions(precision=self.print_precision)
        torch.manual_seed(self.rand_seed)

        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            processor = "GPU"
            torch.cuda.manual_seed_all(self.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True
            print("Using {} GPU(s)...".format(torch.cuda.device_count()))

        else:
            processor = 'CPU'
            device = torch.device("cpu")
            print("Using CPU...")

        self.save(processor)


@inherits(TaskSetupProcessor)
class TaskGetTrainDataset(d6tflow.tasks.TaskPickle):


    #Actual dataset
    data_set = luigi.Parameter(default = data_config().data_set)
    data_randomize = luigi.Parameter(default = data_config().data_randomize)
    raw_data_file = luigi.Parameter(default = data_config().raw_data_file)
    processed_data_file = luigi.Parameter(default = data_config().processed_data_file)
    inference_only = luigi.Parameter(default = net_config().inference_only)

    #Data generation method
    data_generation = luigi.Parameter(default=data_config().data_generation)

    #Network and embedding details
    arch_mlp_bot = luigi.Parameter(default=net_config().arch_mlp_bot)
    arch_embedding_size = luigi.Parameter(default=net_config().arch_embedding_size)

    #Data information
    data_size = luigi.IntParameter(default=data_config().data_size)
    num_batches = luigi.IntParameter(default = data_config().num_batches)
    mini_batch_size = luigi.IntParameter(default = training_config().mini_batch_size)
    round_targets = luigi.BoolParameter(default = data_config().round_targets)
    num_indices_per_lookup = luigi.IntParameter(default = data_config().num_indices_per_lookup)
    num_indices_per_lookup_fixed = luigi.BoolParameter(default = data_config().num_indices_per_lookup_fixed)

    #Log information 
    data_trace_file = luigi.Parameter(default = log_config().data_trace_file)
    data_trace_enable_padding = luigi.Parameter(default = log_config().data_trace_enable_padding)

    def requires(self):
        return self.clone(TaskSetupProcessor)

    def run(self):

        #initialize dataset dict
        dataset_dict = {}

        # input data
        dataset_dict['ln_bot'] = np.fromstring(self.arch_mlp_bot, dtype=int, sep="-")
        dataset_dict['ln_emb'] = np.fromstring(self.arch_embedding_size, dtype=int, sep="-")
        dataset_dict['m_den'] = dataset_dict['ln_bot'][0]

        if self.data_generation == "dataset":
            (ndataset_dict['nbatches'], dataset_dict['lX'], 
             dataset_dict['lS_o'], dataset_dict['lS_i'],
             dataset_dict['lT'], dataset_dict['nbatches_test'],
             dataset_dict['lX_test'], dataset_dict['lS_o_test'], 
             dataset_dict['lS_i_test'], dataset_dict['lT_test'],
             dataset_dict['ln_emb'], dataset_dict['m_den']) = \
            dp.read_dataset(
                            self.data_set,
                            self.mini_batch_size,
                            self.data_randomize,
                            self.num_batches,
                            True,
                            self.raw_data_file,
                            self.processed_data_file,
                            self.inference_only,
                            )
            dataset_dict['ln_bot'][0] = dataset_dict['m_den']


        #If the data generation is random
        elif self.data_generation == "random":
            (dataset_dict['nbatches'], dataset_dict['lX'], 
             dataset_dict['lS_o'], dataset_dict['lS_i']) =\
              dp.generate_random_input_data(
                self.data_size,
                self.num_batches,
                self.mini_batch_size,
                self.round_targets,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                dataset_dict['m_den'],
                dataset_dict['ln_emb'],
            )

        #If the data genreation is synthetic
        elif self.data_generation == "synthetic":
            (dataset_dict['nbatches'], dataset_dict['lX'], 
             dataset_dict['lS_o'], dataset_dict['lS_i']) = \
            dp.generate_synthetic_input_data(
                self.data_size,
                self.num_batches,
                self.mini_batch_size,
                self.round_targets,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                dataset_dict['m_den'],
                dataset_dict['ln_emb'],
                self.data_trace_file,
                self.data_trace_enable_padding,
            )

        #Generate an error if the generation method is not supported
        else:
            raise ValueError(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        self.save(dataset_dict)



@inherits(TaskGetTrainDataset)
class TaskGetTestDataset(d6tflow.tasks.TaskPickle):

    def requires(self):
        return self.clone(TaskGetTrainDataset)

    def run(self):

        dataset_dict = self.input().load()

        if self.data_generation != 'dataset':
            (dataset_dict['nbatches'], dataset_dict['lT']) =\
             dp.generate_random_output_data(
                    self.data_size,
                    self.num_batches,
                    self.mini_batch_size,
                    round_targets=self.round_targets)

        self.save(dataset_dict)


### START HERE!!! 
@inherits(TaskGetTestDataset)
class TaskLintParameters(d6tflow.tasks.TaskPickle):

    arch_interaction_op = luigi.Parameter(default = net_config().arch_interaction_op)
    arch_interaction_itself = luigi.Parameter(default = net_config().arch_interaction_itself)
    arch_mlp_top = luigi.Parameter(default = net_config().arch_mlp_top)
    arch_sparse_feature_size = luigi.IntParameter(default = net_config().arch_sparse_feature_size)
    debug_mode = luigi.BoolParameter(default = log_config().debug_mode)


    def requires(self):
        return self.clone(TaskGetTestDataset)

    def run(self):

        #Input features
        dataset_dict = self.input().load()


        print("\n\n\n\n\n\n\n\n\n")
        #Temp variables for linting
        num_fea = dataset_dict['ln_emb'].size + 1  # num sparse + num dense features
        m_den_out = dataset_dict['ln_bot'][dataset_dict['ln_bot'].size - 1]


        if self.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if self.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif self.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )
        arch_mlp_top_adjusted = str(num_int) + "-" + self.arch_mlp_top
        dataset_dict['ln_top'] = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
        # sanity check: feature sizes and mlp dimensions must match
        if dataset_dict['m_den'] != dataset_dict['ln_bot'][0]:
            sys.exit(
                "ERROR: arch-dense-feature-size "
                + str(dataset_dict['m_den'])
                + " does not match first dim of bottom mlp "
                + str(dataset_dict['ln_bot'][0])
            )
        if self.arch_sparse_feature_size != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(self.arch_sparse_feature_size)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
        if num_int != dataset_dict['ln_top'][0]:
            sys.exit(
                "ERROR: # of feature interactions "
                + str(num_int)
                + " does not match first dimension of top mlp "
                + str(dataset_dict['ln_top'][0])
            )

        # test prints (model arch)
        if self.debug_mode:
            print("model arch:")
            print(
                "mlp top arch "
                + str(dataset_dict['ln_top'].size - 1)
                + " layers, with input to output dimensions:"
            )
            print(dataset_dict['ln_top'])
            print("# of interactions")
            print(num_int)
            print(
                "mlp bot arch "
                + str(dataset_dict['ln_top'].size - 1)
                + " layers, with input to output dimensions:"
            )
            print(dataset_dict['ln_bot'])
            print("# of features (sparse and dense)")
            print(num_fea)
            print("dense feature size")
            print(dataset_dict['m_den'])
            print("sparse feature size")
            print(self.arch_sparse_feature_size)
            print(
                "# of embeddings (= # of sparse features) "
                + str(dataset_dict['ln_emb'].size)
                + ", with dimensions "
                + str(self.arch_sparse_feature_size)
                + "x:"
            )
            print(dataset_dict['ln_emb'])

            print("data (inputs and targets):")
            for j in range(0, dataset_dict['nbatches']):
                print("mini-batch: %d" % j)
                print(dataset_dict['lX'][j].detach().cpu().numpy())
                # transform offsets to lengths when printing
                print(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(dataset_dict['lS_i'][j][i].shape)
                        ).tolist()
                        for i, S_o in enumerate(dataset_dict['lS_o'][j])
                    ]
                )
                print([S_i.detach().cpu().tolist() for S_i in dataset_dict['lS_i'][j]])
                print(dataset_dict['lT'][j].detach().cpu().numpy())


        print("\n\n\n\n\n\n\n\n\n")

        self.save(dataset_dict)

@inherits(TaskLintParameters)
class TaskBuildNetwork(d6tflow.tasks.TaskPickle):

    sync_dense_params = luigi.BoolParameter(default = training_config().sync_dense_params)
    loss_threshold = luigi.FloatParameter(default = training_config().loss_threshold)

    def requires(self):
        return self.clone(TaskLintParameters)

    def run(self):

        dataset_dict = self.input().load()

        dlrm = DLRM_Net(
            self.arch_sparse_feature_size,
            dataset_dict['ln_emb'],
            dataset_dict['ln_bot'],
            dataset_dict['ln_top'],
            arch_interaction_op=self.arch_interaction_op,
            arch_interaction_itself=self.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=dataset_dict['ln_top'].size - 2,
            sync_dense_params=self.sync_dense_params,
            loss_threshold=self.loss_threshold,
        )

        # test prints: Model setup
        if self.debug_mode:
            
            print("\n\n\n<===========================================================>")
            print("Initial parameters (weights and bias):")
            for param in dlrm.parameters():
                print(param.detach().cpu().numpy())
            print("<===========================================================>\n\n\n")
            # print(dlrm)

        #Potential setup for using the GPU
        if self.use_gpu and torch.cuda.device_count() > 1:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm.ndevices = min(torch.cuda.device_count(), 
                                self.mini_batch_size, 
                                dataset_dict['num_fea'] - 1)
            dlrm = dlrm.to(torch.device("cuda", 0))  # .cuda()

        self.save({'dlrm': dlrm,
                  'dataset_dict': dataset_dict})




@inherits(TaskBuildNetwork)
# todo: possible to write a decorator? @clone_parent
class TaskModelTrain(d6tflow.tasks.TaskPickle):

    load_model = luigi.Parameter(default = io_config().load_model)
    loss_function = luigi.Parameter(default = training_config().loss_function)
    learning_rate = luigi.FloatParameter(default = training_config().learning_rate)
    num_epochs = luigi.IntParameter(default = training_config().num_epochs)
    print_time = luigi.BoolParameter(default = log_config().print_time)
    save_model = luigi.Parameter(default = io_config().save_model)
    enable_profiling = luigi.BoolParameter(default = log_config().enable_profiling)
    loss_function = luigi.Parameter(default = training_config().loss_function)

    #Frequencies
    print_freq = luigi.IntParameter(default = log_config().print_freq)
    test_freq = luigi.IntParameter(default = log_config().test_freq)


    def requires(self):
        return self.clone(TaskBuildNetwork)

    def run(self):

        # todo: how to load/save intermediate models?
        # normally if exists, task considered complete so wouldn't continue training
        # condition is k < args.num_epochs, anything before don't save final model
        # todo: override complete function where if super_complete() output.load() and saved_epoch>=nepoch
        # this is where it loads intermediate epochs and results, rest don't need
        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']

        # specify the loss function
        if self.loss_function == "mse":
            self.loss_function = torch.nn.MSELoss(reduction="mean")
        elif self.loss_function == "bce":
            self.loss_function = torch.nn.BCELoss(reduction="mean")
        else:
            sys.exit("ERROR: --loss-function=" + self.loss_function + " is not supported")

        if not self.inference_only:
            # specify the optimizer algorithm
            optimizer = torch.optim.SGD(dlrm.parameters(), lr=self.learning_rate)        


        #Initialize vars for iteration
        best_gA_test = 0
        total_time = 0
        total_loss = 0
        total_accu = 0
        total_iter = 0
        k = 0  # epochs

        if not (self.load_model == ""):

            print("Loading saved mode {}".format(self.load_model))
            ld_model = torch.load(self.load_model)
            dlrm.load_state_dict(ld_model["state_dict"])
            ld_model = copy.deepcopy(ld_model)

            #Not inference only
            if not self.inference_only:
                optimizer.load_state_dict(ld_model["opt_state_dict"])
                best_gA_test = ld_model['test_acc']
                total_loss = ld_model['total_loss']
                total_accu = ld_model['total_accu']
                k = ld_model["epoch"]  # epochs
                j = ld_model["iter"]  # batches
            else:
                self.print_freq = ld_model['nbatches']
                self.test_freq = 0
            print(
                "Saved model Training state: epoch = {:d}/{:d}, batch = {:d}/{:d}, train loss = {:.6f}, train accuracy = {:3.3f} %".format(
                    ld_model["epoch"], ld_model["nepochs"], ld_model["iter"], ld_model['nbatches'], ld_model['train_loss'], ld_model['train_acc'] * 100
                )
            )
            print(
                "Saved model Testing state: nbatches = {:d}, test loss = {:.6f}, test accuracy = {:3.3f} %".format(
                    ld_model['nbatches_test'], ld_model['test_loss'], ld_model['test_acc'] * 100
                )
            )


        print("time/loss/accuracy (if enabled):")
        with torch.autograd.profiler.profile(self.enable_profiling, self.use_gpu) as prof:
            print("\n\n\n<==============================================================>")
            while k < self.num_epochs:
                j = 0
                while j < dataset_dict['nbatches']:
                    t1 = time_wrap(self.use_gpu)

                    # forward pass
                    Z = dlrm_wrap(dlrm, dataset_dict['lX'][j], dataset_dict['lS_o'][j], 
                                  dataset_dict['lS_i'][j], self.use_gpu, torch.device("cuda", 0))

                    # loss
                    E = loss_fn_wrap(self.loss_function, Z, dataset_dict['lT'][j], self.use_gpu, torch.device("cuda", 0))

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    S = Z.detach().cpu().numpy()  # numpy array
                    T = dataset_dict['lT'][j].detach().cpu().numpy()  # numpy array
                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    A = np.sum((np.round(S, 0) == T).astype(np.uint8)) / mbs

                    if not self.inference_only:
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        optimizer.zero_grad()
                        # backward pass
                        E.backward()
                        # debug prints (check gradient norm)
                        # for l in mlp.layers:
                        #     if hasattr(l, 'weight'):
                        #          print(l.weight.grad.norm().item())

                        # optimizer
                        optimizer.step()

                    t2 = time_wrap(self.use_gpu)
                    total_time += t2 - t1
                    total_accu += A
                    total_loss += L
                    total_iter += 1

                    print_tl = ((j + 1) % self.print_freq == 0) or (j + 1 == dataset_dict['nbatches'])
                    print_ts = (
                        (self.test_freq > 0)
                        and (self.data_generation == "dataset")
                        and (((j + 1) % self.test_freq == 0) or (j + 1 == dataset_dict['nbatches']))
                    )

                    # print time, loss and accuracy
                    if print_tl or print_ts:
                        gT = 1000.0 * total_time / total_iter if self.print_time else -1
                        total_time = 0

                        gL = total_loss / total_iter
                        total_loss = 0

                        gA = total_accu / total_iter
                        total_accu = 0

                        str_run_type = "inference" if self.inference_only else "training"
                        print(
                            "Finished {} it {}/{} of epoch {}, ".format(
                                str_run_type, j + 1, dataset_dict['nbatches'], k
                            )
                            + "{:.2f} ms/it, loss {:.6f}, accuracy {:3.3f} %".format(
                                gT, gL, gA * 100
                            )
                        )
                        total_iter = 0

                    # testing
                    if print_ts and not self.inference_only:
                        test_accu = 0
                        test_loss = 0

                        for jt in range(0, dataset_dict['nbatches_test']):
                            t1_test = time_wrap(self.use_gpu)

                            # forward pass
                            Z_test = dlrm_wrap(dlrm,
                                lX_test[jt], lS_o_test[jt], lS_i_test[jt], self.use_gpu, device)
                            # loss
                            E_test = loss_fn_wrap(Z_test, lT_test[jt], self.use_gpu, device)

                            # compute loss and accuracy
                            L_test = E_test.detach().cpu().numpy()  # numpy array
                            S_test = Z_test.detach().cpu().numpy()  # numpy array
                            T_test = lT_test[jt].detach().cpu().numpy()  # numpy array
                            mbs_test = T_test.shape[
                                0
                            ]  # = args.mini_batch_size except maybe for last
                            A_test = (
                                np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                                / mbs_test
                            )

                            t2_test = time_wrap(self.use_gpu)

                            test_accu += A_test
                            test_loss += L_test

                        gL_test = test_loss / nbatches_test
                        gA_test = test_accu / nbatches_test

                        is_best = gA_test > best_gA_test
                        if is_best:
                            best_gA_test = gA_test
                            if not (args.save_model == ""):
                                print("Saving model to {}".format(args.save_model))
                                torch.save(
                                    {
                                        "epoch": k,
                                        "nepochs": self.num_epochs,
                                        "nbatches": dataset_dict['nbatches'],
                                        "nbatches_test": dataset_dict['nbatches_test'],
                                        "iter": j + 1,
                                        "state_dict": dlrm.state_dict(),
                                        "train_acc": gA,
                                        "train_loss": gL,
                                        "test_acc": gA_test,
                                        "test_loss": gL_test,
                                        "total_loss": total_loss,
                                        "total_accu": total_accu,
                                        "opt_state_dict": optimizer.state_dict(),
                                    },
                                    self.save_model,
                                )

                        print(
                            "Testing at - {}/{} of epoch {}, ".format(j + 1, dataset_dict['nbatches'], 0)
                            + "loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                gL_test, gA_test * 100, best_gA_test * 100
                            )
                        )

                    j += 1  # nbatches
                k += 1  # nepochs
        print("<==============================================================>\n\n\n")
        dataset_dict['Z'] = Z 
        dataset_dict['L'] = L
        self.save({'dlrm' : dlrm,
                    'dataset_dict': dataset_dict})


@inherits(TaskModelTrain)
class TaskEnableProfiling(d6tflow.tasks.TaskPickle):


    def requires(self):

        return self.clone(TaskModelTrain)
        
    def run(self):

        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']
        
        if self.enable_profiling:

            with open("dlrm_s_pytorch.prof", "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                prof.export_chrome_trace("./dlrm_s_pytorch.json")

        self.save({'dlrm': dlrm,
                  'dataset_dict': dataset_dict}) 


@inherits(TaskModelTrain)
class TaskPlotComputeGraph(d6tflow.tasks.TaskPickle):

    plot_compute_graph = luigi.BoolParameter(default = log_config().plot_compute_graph)

    def requires(self):
        return self.clone(TaskEnableProfiling)

    def run(self):

        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']
        
        if self.plot_compute_graph:
            # sys.exit(
            #     "ERROR: Please install pytorchviz package in order to use the"
            #     + " visualization. Then, uncomment its import above as well as"
            #     + " three lines below and run the code again."
            # )
            V = dataset_dict['Z'].mean() if self.inference_only else dataset_dict['L']
            make_dot(V, params=dict(dlrm.named_parameters()))
            dot.render('dlrm_s_pytorch_graph') # write .pdf file

        self.save({'dlrm': dlrm,
                   'dataset_dict': dataset_dict})


@inherits(TaskPlotComputeGraph)
class TaskShowTestPrints(d6tflow.tasks.TaskPickle):

    def requires(self):
        return self.clone(TaskPlotComputeGraph)

    def run(self):

        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']

        if not self.inference_only and self.debug_mode:
            print("\n\n\n<===========================================================>")
            print("updated parameters (weights and bias):")
            for param in dlrm.parameters():
                print(param.detach().cpu().numpy())
            print("<===========================================================>\n\n\n")

        self.save({'dlrm': dlrm,
                       'dataset_dict': dataset_dict})


@inherits(TaskShowTestPrints)
class TaskSaveOnnx(d6tflow.tasks.TaskPickle):

    save_onnx = luigi.BoolParameter(default = io_config().save_onnx)

    def requires(self):
        return self.clone(TaskShowTestPrints)

    def run(self):

        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']

        if self.save_onnx:
            # export the model in onnx
            with open("dlrm_s_pytorch.onnx", "w+b") as dlrm_pytorch_onnx_file:
                torch.onnx._export(
                    dlrm, (dataset_dict['lX'][0], dataset_dict['lS_o'][0], 
                           dataset_dict['lS_i'][0]), dlrm_pytorch_onnx_file, verbose=True
                )
            # recover the model back
            dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
            # check the onnx model
            onnx.checker.check_model(dlrm_pytorch_onnx)

        self.save({'dlrm': dlrm,
                       'dataset_dict': dataset_dict})

@inherits(TaskSaveOnnx)
class TaskRunDLRMExperiment(d6tflow.tasks.TaskPickle):

    experiment_name = luigi.Parameter(default = process_config().experiment_name)

    def requires(self):
        return self.clone(TaskSaveOnnx)

    def run(self):

        input_dict = self.input().load()
        dataset_dict = input_dict['dataset_dict']
        dlrm = input_dict['dlrm']
        
        print("\n\nFINISHED EXPERIMENT {}\n\n".format(self.experiment_name))

        self.save({'dlrm': dlrm,
                   'dataset_dict': dataset_dict})
