import argparse
import numpy as np
import logging

import random
import torch
from core.modules import Modules

from learner.nn_learner import NNLearner
from h5py import File
from core.nd_data import NDData
from core.writer import Writer

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
        "path", metavar="path", type=str,
        help="path csv")

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
    path = arg_list.path

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

    dump = {
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
        dump.update({
            "post_lr": post_lr,
            "bound": bound,
        })

    ########## for the slurm bug
    save_data = NDData("exp.csv")
    n = len(save_data.get("div-kl", **dump))
    del save_data
    if(n == 1):
        print("OK")
        exit()
    print("GO")
    ##########

    # ----------------------------------------------------------------------- #

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #

    writer = None
    if(writer_path is not None):
        writer = Writer(writer_path)
        writer.open(**dump)

    data = File("data/"+arg_list.data+".h5", "r")

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    input_size = list(x_train.shape[1:])

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    size_prior = int(len(x_train)*prior)
    size_post = len(x_train)-int(size_prior)
    x_post = x_train[size_prior:, :]
    y_post = y_train[size_prior:]
    if(step == "prior"):
        x_prior = x_train[:size_prior, :]
        y_prior = y_train[:size_prior]

    if(step == "prior"):
        learner = NNLearner(
            model, var, batch_size, prior, "cuda", step=step,
            prior_lr=prior_lr, prior_epoch=prior_epoch)
    else:
        learner = NNLearner(
            model, var, batch_size, prior, "cuda",
            prior_lr=prior_lr, post_lr=post_lr,
            prior_epoch=prior_epoch, post_epoch=post_epoch,
            delta=delta, bound=bound)

    learner.load(writer["state_dict"])

    zero_one_loss = Modules("ZeroOneLoss", learner.model).fit

    bound_ours = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta, T=prior_epoch)
    bound_rivasplata = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta, T=prior_epoch, bound="rivasplata")
    bound_catoni = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta, T=prior_epoch, bound="catoni")
    bound_blanchard = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta, T=prior_epoch, bound="blanchard")
    bound_renyi_sample = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta,
        T=prior_epoch, sample=sample, bound="renyi")
    bound_kl_sample = Modules(
        "Bound", learner.model,
        zero_one_loss, size_post, delta,
        T=prior_epoch, sample=sample, bound="kl")

    if(step == "prior"):
        x_prior_post_test = np.concatenate((x_prior, x_post, x_test), axis=0)
    else:
        x_post_test = np.concatenate((x_post, x_test), axis=0)

    bound_ours_list = []
    bound_rivasplata_list = []
    bound_catoni_list = []
    div_rivasplata_list = []
    bound_blanchard_list = []

    test_risk_list = []
    post_risk_list = []
    if(step == "prior"):
        prior_risk_list = []

    concat_y_pred = None
    concat_y = None

    for i in range(sample):
        logging.info("Evaluating NN ... [{}/{}]\r".format(
            i+1, sample))

        if(step == "prior"):
            y_prior_post_test_pred = learner.predict(
                x_prior_post_test, init_keep=False)
            y_prior_pred = y_prior_post_test_pred[:len(x_prior)]
            y_post_pred = y_prior_post_test_pred[
                len(x_prior):len(x_prior)+len(x_post)]
            y_test_pred = y_prior_post_test_pred[len(x_prior)+len(x_post):]
        else:
            y_post_test_pred = learner.predict(x_post_test, init_keep=False)
            y_post_pred = y_post_test_pred[:len(x_post)]
            y_test_pred = y_post_test_pred[len(x_post):]

        bound_ours_list.append(
            bound_ours.fit(y_post_pred, y_post).item())
        bound_rivasplata_list.append(
            bound_rivasplata.fit(y_post_pred, y_post).item())
        bound_catoni_list.append(
            bound_catoni.fit(y_post_pred, y_post).item())
        bound_blanchard_list.append(
            bound_blanchard.fit(y_post_pred, y_post).item())

        div_rivasplata_list.append(learner.model.div_rivasplata.item())

        test_risk_list.append(zero_one_loss(y_test_pred, y_test))
        post_risk_list.append(zero_one_loss(y_post_pred, y_post))
        if(step == "prior"):
            prior_risk_list.append(zero_one_loss(y_prior_pred, y_prior))

        if(concat_y_pred is None):
            concat_y_pred = y_post_pred
            concat_y = y_post
        else:
            concat_y_pred = np.concatenate((y_post_pred, concat_y_pred))
            concat_y = np.concatenate((y_post, concat_y))

    logging.info("\n")

    bound_ours_list = np.array(bound_ours_list)
    bound_rivasplata_list = np.array(bound_rivasplata_list)
    bound_catoni_list = np.array(bound_catoni_list)
    bound_blanchard_list = np.array(bound_blanchard_list)

    test_risk_list = np.array(test_risk_list)
    post_risk_list = np.array(post_risk_list)
    if(step == "prior"):
        prior_risk_list = np.array(prior_risk_list)

    bound_renyi_sample = bound_renyi_sample.fit(concat_y_pred, concat_y).item()
    bound_kl_sample = bound_kl_sample.fit(concat_y_pred, concat_y).item()

    save_dict = {
        "bound-ours-mean": np.mean(bound_ours_list),
        "bound-ours-std": np.std(bound_ours_list),
        "bound-rivasplata-mean": np.mean(bound_rivasplata_list),
        "bound-rivasplata-std": np.std(bound_rivasplata_list),
        "bound-catoni-mean": np.mean(bound_catoni_list),
        "bound-catoni-std": np.std(bound_catoni_list),
        "bound-blanchard-mean": np.mean(bound_blanchard_list),
        "bound-blanchard-std": np.std(bound_blanchard_list),
        "bound-renyi-sample": bound_renyi_sample,
        "bound-kl-sample": bound_kl_sample,
        "test-risk-mean": np.mean(test_risk_list),
        "test-risk-std": np.std(test_risk_list),
        "post-risk-mean": np.mean(post_risk_list),
        "post-risk-std": np.std(post_risk_list),
        "div-renyi": learner.model.div_renyi.item(),
        "div-kl": learner.model.div_kl.item(),
        "div-rivasplata-mean": np.mean(div_rivasplata_list),
        "div-rivasplata-std": np.std(div_rivasplata_list),
    }

    if(step == "prior"):
        save_dict.update({
            "prior-risk-mean": np.mean(prior_risk_list),
            "prior-risk-std": np.std(prior_risk_list),
        })

    save_data = NDData(path)
    save_data.set(save_dict, dump)
    save_data.save()
    del save_data

    ###########################################################################
