import d6tflow
import luigi
from luigi.util import inherits

import numpy as np

import dlrm_data_pytorch as dp

import flow_cfg as cfg

class DataInput(d6tflow.tasks.TaskCache):
    data_generation = luigi.Parameter(default='random')
    arch_mlp_bot = luigi.Parameter(default="4-3-2")
    arch_embedding_size = luigi.Parameter(default="4-3-2")
    data_size = luigi.IntParameter(default=1)

    def run(self):
        # input data
        ln_bot = np.fromstring(self.arch_mlp_bot, dtype=int, sep="-")
        ln_emb = np.fromstring(self.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        if self.data_generation == "random":
            (nbatches, lX, lS_o, lS_i) = dp.generate_random_input_data(
                self.data_size,
                self.num_batches,
                cfg.mini_batch_size,
                cfg.round_targets,
                cfg.num_indices_per_lookup,
                cfg.num_indices_per_lookup_fixed,
                m_den,
                ln_emb,
            )
        elif self.data_generation == "synthetic":
            (nbatches, lX, lS_o, lS_i) = dp.generate_synthetic_input_data(
                self.data_size,
                cfg.num_batches,
                cfg.mini_batch_size,
                cfg.round_targets,
                cfg.num_indices_per_lookup,
                cfg.num_indices_per_lookup_fixed,
                m_den,
                ln_emb,
                cfg.data_trace_file,
                cfg.data_trace_enable_padding,
            )
        else:
            raise ValueError(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        self.save((nbatches, lX, lS_o, lS_i))


@inherits(DataInput)
# todo: possible to write a decorator? @clone_parent
class DataOutput(d6tflow.tasks.TaskCache):

    def requires(self):
        return self.clone_parent()

    def run(self):
        (nbatches, lT) = dp.generate_random_output_data(
                self.data_size,
                cfg.num_batches,
                cfg.mini_batch_size,
                round_targets=cfg.round_targets,
            )

        self.save((nbatches, lT))

@inherits(DataInput)
# todo: possible to write a decorator? @clone_parent
class ModelTrain(d6tflow.tasks.TaskCache):

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


