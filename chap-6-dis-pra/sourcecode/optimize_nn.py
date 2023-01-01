import argparse
import numpy as np
import logging
import random
import torch

from core.writer import Writer
from learner.nn_learner import NNLearner
from h5py import File
########## for the slurm bug
from core.nd_data import NDData
##########

###############################################################################

if __name__ == "__main__":

    ###########################################################################

    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "writer_path", metavar="writer_path", type=str,
        help="writer_path")

    arg_parser.add_argument(
        "--data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "--model", metavar="model", type=str,
        help="model")

    arg_parser.add_argument(
        "--step", metavar="step", default="prior", type=str,
        help="step")

    arg_parser.add_argument(
        "--var", metavar="var", default="", type=str,
        help="var")

    arg_parser.add_argument(
        "--prior", metavar="prior", default=0.5, type=float,
        help="prior")
    arg_parser.add_argument(
        "--delta", metavar="delta", default=0.05, type=float,
        help="delta")
    arg_parser.add_argument(
        "--bound", metavar="bound", default="renyi", type=str,
        help="bound")

    arg_parser.add_argument(
        "--prior_lr", metavar="prior_lr", default=0.001, type=float,
        help="prior_lr")
    arg_parser.add_argument(
        "--post_lr", metavar="post_lr", default=0.001, type=float,
        help="post_lr")
    arg_parser.add_argument(
        "--prior_epoch", metavar="prior_epoch", default=1, type=int,
        help="prior_epoch")
    arg_parser.add_argument(
        "--post_epoch", metavar="post_epoch", default=1, type=int,
        help="post_epoch")

    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=64, type=int,
        help="batch_size")
    arg_parser.add_argument(
        "--sample", metavar="sample", default=5, type=int,
        help="sample")

    arg_list, _ = arg_parser.parse_known_args()

    writer_path = arg_list.writer_path

    model = arg_list.model

    var = arg_list.var

    step = arg_list.step

    prior = arg_list.prior
    delta = arg_list.delta
    bound = arg_list.bound

    prior_lr = arg_list.prior_lr
    post_lr = arg_list.post_lr
    prior_epoch = arg_list.prior_epoch
    post_epoch = arg_list.post_epoch

    batch_size = arg_list.batch_size
    sample = arg_list.sample

    dump_prior = {
        "data": arg_list.data,
        "model": model,

        "var": var,

        "step": step,

        "prior": prior,
        "delta": delta,

        "prior_lr": prior_lr,
        "prior_epoch": prior_epoch,
        "post_epoch": post_epoch,

        "batch_size": batch_size,
        "sample": sample,
    }
    if(step == "post"):
        dump_post = dict(dump_prior)
        dump_prior["step"] = "prior"
        dump_post.update({
            "post_lr": post_lr,
            "bound": bound,
        })

    ########## for the slurm bug
    save_data = NDData("exp.csv")
    if(step == "prior"):
        dump = dump_prior
    else:
        dump = dump_post
    n = len(save_data.get("div-kl", **dump))
    del save_data
    if(n == 1):
        print("OK")
        exit()
    print("GO")
    ##########

    # ----------------------------------------------------------------------- #

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #

    writer = None
    if(writer_path is not None):
        writer = Writer(writer_path)
        if(step == "prior"):
            writer.open(**dump_prior)
        else:
            writer.open(**dump_post)

    data = File("data/"+arg_list.data+".h5", "r")

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    input_size = list(x_train.shape[1:])

    # PRIOR ----------------------------------------------------------------- #

    if(step == "prior"):
        learner = NNLearner(
            model, var, batch_size, prior, "cuda", step=step,
            prior_lr=prior_lr, prior_epoch=prior_epoch, writer=writer)
        learner.fit(x_train, y_train)

    # POSTERIOR ------------------------------------------------------------- #

    elif(step == "post"):

        if(writer_path is not None):
            writer_load = Writer(writer_path)
            writer_load.open(**dump_prior)
        else:
            raise RuntimeError("Writer cannot be opened")

        learner = NNLearner(
            model, var, batch_size, prior, "cuda",
            step=step, load=writer_load["state_dict"],
            post_lr=post_lr, post_epoch=post_epoch, prior_epoch=prior_epoch,
            delta=delta, bound=bound, writer=writer)
        learner.fit(x_train, y_train)

    if(writer is not None):
        writer.save()

    ###########################################################################
