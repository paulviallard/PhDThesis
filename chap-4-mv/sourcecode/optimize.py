from time import time
import logging
import numpy as np
import argparse
import random
import torch
from h5py import File

from voter.stump import DecisionStumpMV
from voter.tree import TreeMV

import warnings
from learner.bound_joint_learner import BoundJointLearner
from learner.bound_risk_learner import BoundRiskLearner
from learner.bound_rand_learner import BoundRandLearner
from learner.c_bound_joint_learner import CBoundJointLearner
from learner.c_bound_mcallester_learner import CBoundMcAllesterLearner
from learner.c_bound_seeger_learner import CBoundSeegerLearner
from learner.stochastic_majority_vote_learner import (
    StochasticMajorityVoteLearner)
from learner.naive_bayes_learner import NaiveBayesLearner
from learner.mincq_learner import MinCqLearner
from learner.cb_boost_learner import CBBoostLearner
from learner.nothing_learner import NothingLearner
from core.modules import Modules
from core.nd_data import NDData
from core.writer import Writer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def main():

    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    # ----------------------------------------------------------------------- #

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="Path of the csv file containing the results")
    arg_parser.add_argument(
        "--writer_path", metavar="writer_path", default=None, type=str,
        help="Path of the writer file containing the results")

    arg_parser.add_argument(
        "--seed", metavar="seed", default=-1, type=int,
        help="seed")
    arg_parser.add_argument(
        "--data", metavar="data", default="moons", type=str,
        help="data")
    arg_parser.add_argument(
        "--learner", metavar="learner", default="learner-sto", type=str,
        help="learner")

    arg_parser.add_argument(
        "--nb_per_attribute", metavar="nb_per_attribute", default=10, type=int,
        help="Number of stumps per attribute (used when voter=stump)")
    arg_parser.add_argument(
        "--nb_tree", metavar="nb_tree", default=100, type=int,
        help="Number of trees (used when voter=tree)")
    arg_parser.add_argument(
        "--max_depth", metavar="max_depth", default=None, type=int,
        help="max_depth")
    arg_parser.add_argument(
        "--voter", metavar="voter", default="tree", type=str,
        help="voter")

    arg_parser.add_argument(
        "--prior", metavar="prior", default=0.5, type=float,
        help="Proportion of the prior set")
    arg_parser.add_argument(
        "--multi", metavar="multi", default=0, type=int,
        help="multi")

    arg_parser.add_argument(
        "--risk", metavar="risk", default="exact", type=str,
        help="risk")
    arg_parser.add_argument(
        "--delta", metavar="delta", default=0.05, type=float,
        help="delta")

    arg_parser.add_argument(
        "--sto_prior", metavar="sto_prior", default=1.0, type=float,
        help="sto_prior")
    arg_parser.add_argument(
        "--sigmoid_c", metavar="sigmoid_c", default=100, type=int,
        help="sigmoid_c")
    arg_parser.add_argument(
        "--rand_n", metavar="rand_n", default=100, type=int,
        help="rand_n")
    arg_parser.add_argument(
        "--mc_draws", metavar="mc_draws", default=10, type=int,
        help="mc_draws")

    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=1000, type=int,
        help="epoch")
    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=None, type=int,
        help="batch_size")

    arg_list = arg_parser.parse_known_args()[0]

    path = arg_list.path
    writer_path = arg_list.writer_path

    seed = arg_list.seed

    learner = arg_list.learner

    nb_per_attribute = arg_list.nb_per_attribute
    nb_tree = arg_list.nb_tree
    max_depth = arg_list.max_depth
    voter = arg_list.voter

    sto_prior = arg_list.sto_prior
    prior = arg_list.prior
    multi = bool(arg_list.multi)
    risk_name = arg_list.risk
    delta = arg_list.delta

    sigmoid_c = arg_list.sigmoid_c
    rand_n = arg_list.rand_n
    mc_draws = arg_list.mc_draws

    epoch = arg_list.epoch
    batch_size = arg_list.batch_size

    NB_PARAMS = 20

    dump = {
        "seed": seed,
        "data": arg_list.data,
        "learner": learner,

        "nb_per_attribute": nb_per_attribute,
        "nb_tree": nb_tree,
        "max_depth": max_depth,
        "voter": voter,

        "sto_prior": sto_prior,
        "prior": prior,
        "multi": multi,

        "risk": risk_name,
        "delta": delta,

        "sigmoid_c": sigmoid_c,
        "rand_n": rand_n,
        "mc_draws": mc_draws,

        "epoch": epoch,
        }

    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #

    data = File("data/"+arg_list.data+".h5", "r")

    writer = None
    if(writer_path is not None):
        writer = Writer(writer_path)
        writer.open(**dump)

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    y_train = np.expand_dims(y_train, 1)
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    y_test = np.expand_dims(y_test, 1)

    if(seed != 0):
        train_size = len(x_train)
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        permutation = np.arange(x.shape[0])
        np.random.shuffle(permutation)
        x = x[permutation]
        y = y[permutation]

        x_train = x[:train_size]
        y_train = y[:train_size]
        x_test = x[train_size:]
        y_test = y[train_size:]

    # ----------------------------------------------------------------------- #

    # If the voters are decision stumps
    if(voter == "stump"):
        m = len(x_train)
        # We create the majority vote based on the decision stumps ...
        majority_vote = DecisionStumpMV(
            x_train, y_train, nb_per_attribute=nb_per_attribute,
            complemented=True)

    # If the voters are trees
    elif(voter == "tree"):
        m = int(prior*len(x_train))

        # We take the two sets
        x_train_1 = x_train[:m]
        y_train_1 = y_train[:m]
        x_train_2 = x_train[m:]
        y_train_2 = y_train[m:]

        complemented = False
        if(learner == "mincq"):
            complemented = True

        if(not(multi)):
            # Majority Vote for Chapter 4
            majority_vote = TreeMV(
                x_train_1, y_train_1, nb_tree=nb_tree, max_depth=max_depth,
                complemented=complemented)
            x_train = x_train_2
            y_train = y_train_2
        else:
            # Majority Vote for Chapter 5
            majority_vote = [
                TreeMV(x_train_1, y_train_1,
                       nb_tree=nb_tree, max_depth=max_depth),
                TreeMV(x_train_2, y_train_2,
                       nb_tree=nb_tree, max_depth=max_depth)
            ]
            assert len(majority_vote) == 2

    else:
        raise NotImplementedError("voter must be either stump or tree")

    # ----------------------------------------------------------------------- #

    assert (learner == "c-bound-mcallester"
            or learner == "c-bound-seeger"
            or learner == "c-bound-joint"
            or learner == "bound-risk"
            or learner == "bound-joint"
            or learner == "bound-sto"
            or learner == "bound-rand"
            or learner == "naive-bayes"
            or learner == "mincq"
            or learner == "cb-boost"
            or learner == "nothing")

    # We define the bounds for Chapter 5
    def StochasticMajorityVoteLearner_(
        majority_vote, epoch, m, batch_size, delta, writer
    ):
        return StochasticMajorityVoteLearner(
            majority_vote, epoch, risk=risk_name, m=m,
            batch_size=batch_size, mc_draws=mc_draws,
            sigmoid_c=sigmoid_c, prior=sto_prior, writer=writer)

    def BoundRandLearner_(
        majority_vote, epoch, m, batch_size, delta, writer
    ):
        return BoundRandLearner(
            majority_vote, epoch, m=m, batch_size=batch_size, rand_n=rand_n,
            writer=writer)

    # We learn the weights of the MV with a PAC-Bayesian bound
    if(learner == "c-bound-mcallester"):
        Learner = CBoundMcAllesterLearner
    if(learner == "c-bound-joint"):
        Learner = CBoundJointLearner
    if(learner == "c-bound-seeger"):
        Learner = CBoundSeegerLearner
    if(learner == "bound-risk"):
        Learner = BoundRiskLearner
    if(learner == "bound-joint"):
        Learner = BoundJointLearner
    if(learner == "bound-sto"):
        Learner = StochasticMajorityVoteLearner_
    if(learner == "bound-rand"):
        Learner = BoundRandLearner_

    if(learner in ["c-bound-mcallester", "c-bound-joint", "c-bound-seeger",
                   "bound-risk", "bound-joint", "bound-sto", "bound-rand"]):
        learner_ = Learner(majority_vote, epoch=epoch,
                           m=m, batch_size=batch_size, delta=delta,
                           writer=writer)

    # ----------------------------------------------------------------------- #

    if(learner == "naive-bayes"):
        Learner = NaiveBayesLearner
    if(learner == "mincq"):
        Learner = MinCqLearner(majority_vote, 0.1)
        learner_params = {"mu": np.linspace(10**(-4), 0.5, NB_PARAMS)}
        zero_one = Modules("ZeroOne").fit

        # We learn the weights of the MV with MinCq with different value of mu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_classifier = GridSearchCV(
                Learner, learner_params, cv=3,
                scoring=make_scorer(zero_one, greater_is_better=False),
                refit=False)
            cv_classifier = cv_classifier.fit(x_train, y_train)

        # We learn the weights of the MV with the best mu
        mu_best = cv_classifier.best_params_["mu"]
        def MinCqLearner_(majority_vote):
            return MinCqLearner(majority_vote, mu_best)
        Learner = MinCqLearner_

    if(learner == "cb-boost"):
        Learner = CBBoostLearner

    if(learner == "nothing"):
        Learner = NothingLearner

    if(learner in ["naive-bayes", "mincq", "cb-boost",
                   "nothing"]):
        learner_ = Learner(majority_vote)

    # ----------------------------------------------------------------------- #

    t1 = time()

    if(not(multi)):
        learner_.fit(x=x_train, y=y_train)
    else:
        learner_.fit(x_1=x_train_1, x_2=x_train_2,
                     y_1=y_train_1, y_2=y_train_2)
    model = learner_.mv_diff

    t2 = time()
    logging.info(f"Time: {t2-t1}s\n")

    if(learner == "bound-sto"):
        risk = Modules("Risk", model).fit
        save_dict = {}
    else:
        gibbs = Modules("Risk", model).fit
        disa = Modules("Disagreement", model).fit
        joint = Modules("Joint", model).fit
        risk = Modules("ZeroOne", model).fit

        rS = gibbs(x=x_train, y=y_train)
        rT = gibbs(x=x_test, y=y_test)
        dS = disa(x=x_train, y=y_train)
        dT = disa(x=x_test, y=y_test)
        eS = joint(x=x_train, y=y_train)
        eT = joint(x=x_test, y=y_test)

        logging.info(f"rS: {rS.item()}, dS: {dS.item()}, eS: {eS.item()}\n")
        logging.info(f"rT: {rT.item()}, dT: {dT.item()}, eT: {eT.item()}\n")

        save_dict = {
            "rS": rS,
            "rT": rT,
            "dS": dS,
            "dT": dT,
            "eS": eS,
            "eT": eT,
        }

    if(not(multi)):
        tS = risk(x=x_train, y=y_train)
        tT = risk(x=x_test, y=y_test)
    else:
        tS = risk(x_1=x_train_1, x_2=x_train_2,
                  y_1=y_train_1, y_2=y_train_2)
        tT = risk(x_1=x_test, x_2=x_test,
                  y_1=y_test, y_2=y_test)

    logging.info(f"train risk: {tS.item()}, test risk: {tT.item()}\n")
    save_dict["tS"] = tS
    save_dict["tT"] = tT
    save_dict["time"] = t2-t1
    model.KL()
    save_dict["kl"] = model.kl.item()

    if(learner == "c-bound-mcallester"):
        bound_ = Modules("CBoundMcAllester", model, m=m, delta=delta).fit
    if(learner == "c-bound-joint"):
        bound_ = Modules("CBoundLacasse", model, m=m, delta=delta).fit
    if(learner == "c-bound-seeger"):
        bound_ = Modules("CBoundSeeger", model, m=m, delta=delta).fit
    if(learner == "bound-risk"):
        bound_ = Modules("BoundRisk", model, m=m, delta=delta).fit
    if(learner == "bound-joint"):
        bound_ = Modules("BoundJoint", model, m=m, delta=delta).fit
    if(learner == "bound-sto"):
        bound_ = Modules("BoundSto", model, m=m, delta=delta).fit
    if(learner == "bound-rand"):
        bound_ = Modules(
            "BoundRand", model, m=m, delta=delta, rand_n=rand_n).fit
    if(learner == "naive-bayes"):
        bound_ = Modules("CBoundLacasse", model, m=m, delta=delta).fit
    if(learner == "mincq"):
        bound_ = Modules(
            "CBoundLacasse", model, m=m, delta=delta/NB_PARAMS).fit
    if(learner == "cb-boost"):
        bound_ = Modules("CBoundLacasse", model, m=m, delta=delta).fit
    if(learner == "nothing"):
        bound_ = Modules("CBoundLacasse", model, m=m, delta=delta).fit

    if(not(multi)):
        b = float(bound_(x=x_train, y=y_train))
    else:
        b = float(bound_(x_1=x_train_1, x_2=x_train_2,
                         y_1=y_train_1, y_2=y_train_2))

    logging.info(f"Bound: {b}\n")
    save_dict["bound"] = b

    save_data = NDData(path)
    save_data.set(save_dict, dump)
    save_data.save()
    del save_data

    if(writer is not None):
        writer.save()


###############################################################################

if __name__ == "__main__":
    main()
