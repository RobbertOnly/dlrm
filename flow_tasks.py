import d6tflow
import luigi
from luigi.util import inherits

import numpy as np

import dlrm_data_pytorch_EDITED as dp

from flow_cfg import data_config, training_config, prep_config,\
                     net_config, process_config, log_config, io_config

class TaskSetupProcessor(d6tflow.tasks.TaskCache):

    rand_seed = luigi.IntParameter(default = process_config().rand_seed)
    print_precision = luigi.IntParameter(default = process_config().rand_seed)
    use_gpu = luigi.BoolParameter(default = luigi.process_config().use_gpu)

    def run(self):
        ### some basic setup ###
        processor = None
        np.random.seed(self.rand_seed)
        np.set_printoptions(precision=self.print_precision)
        torch.set_printoptions(precision=self.print_precision)
        torch.manual_seed(self.rand_seed)

        use_gpu = self.use_gpu and torch.cuda.is_available()
        if use_gpu:
            processor = "GPU"
            torch.cuda.manual_seed_all(self.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda", 0)
            ngpus = torch.cuda.device_count()  # 1
            print("Using {} GPU(s)...".format(ngpus))
        else:
            processor = 'CPU'
            device = torch.device("cpu")
            print("Using CPU...")

        self.save(processor)



class TaskGetTrainDataset(d6tflow.tasks.TaskPickle):

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
    num_indices_per_lookup = luigi.BoolParameter(default = data_config().num_indices_per_lookup_fixed)

    #Log information 
    data_trace_file = luigi.Parameter(default = log_config().data_trace_file)
    data_trace_enable_padding = luigi.Parameter(default = log_config().data_trace_enable_padding)


    def run(self):

        #initialize dataset dict
        dataset_dict = {}

        # input data
        dataset_dict['ln_bot'] = np.fromstring(self.arch_mlp_bot, dtype=int, sep="-")
        dataset_dict['ln_emb'] = np.fromstring(self.arch_embedding_size, dtype=int, sep="-")
        dataset_dict['m_den'] = ln_bot[0]

        if self.data_generation == "dataset":
            (ndataset_dict['nbatches'], dataset_dict['lX'], 
             dataset_dict['lS_o'], dataset_dict['lS_i'],
             dataset_dict['lT'], dataset_dict['nbatches_test'],
             dataset_dict['lX_test'], dataset_dict['lS_o_test'], 
             dataset_dict['lS_i_test'], dataset_dict['lT_test'],
             dataset_dict['ln_emb'], dataset_dict['m_den']) = \
            dp.read_dataset(
                            args.data_set,
                            args.mini_batch_size,
                            args.data_randomize,
                            args.num_batches,
                            True,
                            args.raw_data_file,
                            args.processed_data_file,
                            args.inference_only,
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
                m_den,
                ln_emb,
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
                m_den,
                ln_emb,
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
class TaskGetTestData(d6tflow.tasks.TaskCache):

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


class TaskLintParameters(d6tflow.tasks.TaskCache):

    def requires(self):
        self.clone(TaskGetTestData)

    def run(self):
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        m_den_out = ln_bot[ln_bot.size - 1]
        if args.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif args.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + args.arch_interaction_op
                + " is not supported"
            )
        arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
        # sanity check: feature sizes and mlp dimensions must match
        if m_den != ln_bot[0]:
            sys.exit(
                "ERROR: arch-dense-feature-size "
                + str(m_den)
                + " does not match first dim of bottom mlp "
                + str(ln_bot[0])
            )
        if args.arch_sparse_feature_size != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(args.arch_sparse_feature_size)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
        if num_int != ln_top[0]:
            sys.exit(
                "ERROR: # of feature interactions "
                + str(num_int)
                + " does not match first dimension of top mlp "
                + str(ln_top[0])
            )

        # test prints (model arch)
        if args.debug_mode:
            print("model arch:")
            print(
                "mlp top arch "
                + str(ln_top.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_top)
            print("# of interactions")
            print(num_int)
            print(
                "mlp bot arch "
                + str(ln_bot.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_bot)
            print("# of features (sparse and dense)")
            print(num_fea)
            print("dense feature size")
            print(m_den)
            print("sparse feature size")
            print(args.arch_sparse_feature_size)
            print(
                "# of embeddings (= # of sparse features) "
                + str(ln_emb.size)
                + ", with dimensions "
                + str(args.arch_sparse_feature_size)
                + "x:"
            )
            print(ln_emb)

            print("data (inputs and targets):")
            for j in range(0, nbatches):
                print("mini-batch: %d" % j)
                print(lX[j].detach().cpu().numpy())
                # transform offsets to lengths when printing
                print(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[j][i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o[j])
                    ]
                )
                print([S_i.detach().cpu().tolist() for S_i in lS_i[j]])
                print(lT[j].detach().cpu().numpy())

class TaskBuildNetwork(d6tflow.tasks.TaskPickle):

    def requires(self):
        return

    def run(self):
        return

@inherits(DataInput)
# todo: possible to write a decorator? @clone_parent
class TaskModelTrain(d6tflow.tasks.TaskCache):

    def requires(self):
        return {'in':self.clone(DataInput),'out':self.clone(DataOutput)}

    def run(self):
        (nbatches, lX, lS_o, lS_i) = self.input()['in'].load()
        (nbatches, lT) = self.input()['out'].load()
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=args.sync_dense_params,
            loss_threshold=args.loss_threshold,
        )

        # todo: how to load/save intermediate models?
        # normally if exists, task considered complete so wouldn't continue training
        # condition is k < args.nepochs, anything before don't save final model
        # todo: override complete function where if super_complete() output.load() and saved_epoch>=nepoch
        # this is where it loads intermediate epochs and results, rest don't need
        if self.output().exists():
            ld_model = torch.load(args.load_model)
            dlrm.load_state_dict(ld_model["state_dict"])
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            k = ld_k  # epochs
            j = ld_j  # batches

        print("time/loss/accuracy (if enabled):")
        with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
            while k < args.nepochs:
                j = 0
                while j < nbatches:
                    t1 = time_wrap(use_gpu)

                    # forward pass
                    Z = dlrm_wrap(lX[j], lS_o[j], lS_i[j], use_gpu, device)

                    # loss
                    E = loss_fn_wrap(Z, lT[j], use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    S = Z.detach().cpu().numpy()  # numpy array
                    T = lT[j].detach().cpu().numpy()  # numpy array
                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    A = np.sum((np.round(S, 0) == T).astype(np.uint8)) / mbs

                    if not args.inference_only:
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

                    t2 = time_wrap(use_gpu)
                    total_time += t2 - t1
                    total_accu += A
                    total_loss += L
                    total_iter += 1

                    print_tl = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                    print_ts = (
                        (args.test_freq > 0)
                        and (args.data_generation == "dataset")
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if print_tl or print_ts:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        gL = total_loss / total_iter
                        total_loss = 0

                        gA = total_accu / total_iter
                        total_accu = 0

                        str_run_type = "inference" if args.inference_only else "training"
                        print(
                            "Finished {} it {}/{} of epoch {}, ".format(
                                str_run_type, j + 1, nbatches, k
                            )
                            + "{:.2f} ms/it, loss {:.6f}, accuracy {:3.3f} %".format(
                                gT, gL, gA * 100
                            )
                        )
                        total_iter = 0

                    # testing
                    if print_ts and not args.inference_only:
                        test_accu = 0
                        test_loss = 0

                        for jt in range(0, nbatches_test):
                            t1_test = time_wrap(use_gpu)

                            # forward pass
                            Z_test = dlrm_wrap(
                                lX_test[jt], lS_o_test[jt], lS_i_test[jt], use_gpu, device
                            )
                            # loss
                            E_test = loss_fn_wrap(Z_test, lT_test[jt], use_gpu, device)

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

                            t2_test = time_wrap(use_gpu)

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
                                        "nepochs": args.nepochs,
                                        "nbatches": nbatches,
                                        "nbatches_test": nbatches_test,
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
                                    args.save_model,
                                )

                        print(
                            "Testing at - {}/{} of epoch {}, ".format(j + 1, nbatches, 0)
                            + "loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                gL_test, gA_test * 100, best_gA_test * 100
                            )
                        )

                    j += 1  # nbatches
                k += 1  # nepochs

@inherits()
class TaskPlotComputeGraph(d6tflow.tasks.TaskCache):

    def requires(self):
        return 

    def run(self):
        return

@inherits()
class TaskEnableProfiling(d6tflow.tasks.TaskCache):

    def requires(self):
        return 

    def run(self):
        return

@inherits()
class TaskSaveOnnx(d6tflow.tasks.TaskCache):

    def requires(self):
        return 

    def run(self):
        return

