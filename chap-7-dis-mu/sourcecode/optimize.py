import argparse
from h5py import File
import logging
import numpy as np
import os
import random
import torch

from core.module import Module
from core.nd_data import NDData

from learner.nn_learner import NNLearner


###############################################################################

def main():
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path csv")

    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="seed")
    arg_parser.add_argument(
        "--lr_sgd", metavar="lr_sgd", default=0.0, type=float,
        help="lr-sgd")
    arg_parser.add_argument(
        "--lr_mh", metavar="lr_mh", default=10**(-3), type=float,
        help="lr_mh")
    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=32, type=int,
        help="batch_size")

    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=1, type=int,
        help="epoch")
    arg_parser.add_argument(
        "--epoch_mh", metavar="epoch_mh", default=1, type=int,
        help="epoch_mh")
    arg_parser.add_argument(
        "--depth", metavar="depth", default=2, type=int,
        help="")
    arg_parser.add_argument(
        "--width", metavar="width", default=8*25, type=int,
        help="")
    arg_parser.add_argument(
        "--measure", metavar="measure", default="zero", type=str,
        help="")
    arg_parser.add_argument(
        "--alpha", metavar="alpha", default=1.0, type=float,
        help="")
    arg_parser.add_argument(
        "--delta", metavar="delta", default=0.05, type=float,
        help="")

    arg_list = arg_parser.parse_known_args()[0]

    data = arg_list.data
    path = arg_list.path

    seed = arg_list.seed
    lr_sgd = arg_list.lr_sgd
    lr_mh = arg_list.lr_mh

    batch_size = arg_list.batch_size

    epoch = arg_list.epoch
    epoch_mh = arg_list.epoch_mh

    depth = arg_list.depth
    width = arg_list.width
    measure = arg_list.measure
    alpha = arg_list.alpha
    delta = arg_list.delta

    # ----------------------------------------------------------------------- #

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #

    data = File(os.path.join("data", data+".h5"), "r")

    # ----------------------------------------------------------------------- #
    # Optimization

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    y_train = np.expand_dims(y_train, 1)
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    y_test = np.expand_dims(y_test, 1)

    class_size = len(np.unique(y_train))
    input_size = x_train.shape

    model_kwargs = {
        "input_size": input_size,
        "class_size": class_size,
        "width": width,
        "depth": depth,
        "measure": measure,
    }

    learner = NNLearner(
        "NiN_NN", model_kwargs, batch_size, epoch, epoch_mh, lr_sgd, lr_mh,
        alpha, "cuda")
    learner.fit(x_train, y_train)

    # test
    save_data = NDData(path)
    save_data.save()
    del save_data

    # ----------------------------------------------------------------------- #
    # Evaluation

    out_train = learner.output(x_train)
    pred_test = learner.predict(x_test)
    pred_train = learner.predict(x_train)

    batch = {
        "out": out_train,
        "pred": pred_train,
        "x": x_train,
        "y": y_train,
        "size": len(x_train),
        "step": "measure",
        "alpha": alpha,
    }
    learner.model.get_measures(batch)
    zero_one = Module("ZeroOne", learner.model).fit

    bound = Module(
        "Bound", learner.model, learner, len(x_train),
        delta, alpha).fit

    emp_risk_bound = Module("EmpRiskBound", learner.model).fit

    risk_test = zero_one(pred_test, y_test)
    risk_train = zero_one(pred_train, y_train)
    bound = bound(x_train, pred_train, out_train, y_train)
    mcallester_bound = emp_risk_bound(pred_train, y_train, bound, "mcallester")
    seeger_bound = emp_risk_bound(pred_train, y_train, bound, "seeger")

    logging.info(f"Train Risk: {risk_train}\n")
    logging.info(f"Test Risk: {risk_test}\n")
    logging.info(f"Bound: {bound}\n")
    logging.info(f"McAllester's Bound: {risk_test} <= {mcallester_bound}\n")
    logging.info(f"Seeger's Bound: {risk_test} <= {seeger_bound}\n")

    save_dict = {
        "risk_test": risk_test.item(),
        "risk_train": risk_train.item(),
        "bound": bound.item(),
        "mcallester_bound": mcallester_bound.item(),
        "seeger_bound": seeger_bound,
        "measure": learner.model.measures[measure].item(),
    }

    dump = {
        "data": arg_list.data,
        "seed": seed,
        "lr_sgd": lr_sgd,
        "lr_mh": lr_mh,
        "batch_size": batch_size,
        "epoch": epoch,
        "epoch_mh": epoch_mh,
        "depth": depth,
        "width": width,
        "measure": measure,
        "alpha": alpha,
        "delta": delta,
        }
    save_data = NDData(path)
    save_data.set(save_dict, dump)
    save_data.save()
    del save_data

###############################################################################


if __name__ == "__main__":
    main()
