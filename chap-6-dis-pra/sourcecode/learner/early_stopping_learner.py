import math
import logging


###############################################################################

class MetaEarlyStoppingLearner(type):

    def __call__(cls, *args, **kwargs):
        bases = (cls, args[0].__class__, )
        new_cls = type(cls.__name__, bases, {})
        return super(MetaEarlyStoppingLearner, new_cls
                     ).__call__(*args, **kwargs)


class EarlyStoppingLearner(metaclass=MetaEarlyStoppingLearner):

    def __init__(self, learner, criteria, val_epoch=10):
        self.__dict__ = learner.__dict__
        self.val_epoch = val_epoch
        self.criteria = criteria

    def fit(self, x_train, y_train, x_val, y_val):

        crit_val_best = math.inf

        for epoch in range(self.val_epoch):

            logging.info(("Running validation epoch [{}/{}] ...\n").format(
                epoch+1, self.val_epoch))

            super().fit(x_train, y_train)
            predict_val = super().predict(x_val)
            crit_val = self.criteria(y_val, predict_val)

            if(crit_val_best > crit_val):
                logging.info(("{:.4f} > {:.4f} -> saving ...\n").format(
                    crit_val_best, crit_val))
                crit_val_best = crit_val
                b = self.save()

        self.load(b)

        return self

###############################################################################
