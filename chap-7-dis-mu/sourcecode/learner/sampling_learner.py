from core.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner

###############################################################################


class SamplingLearner(OptimizeGDLearner):

    def __init__(
        self, model, loss, optim, device, batch_size=None,
        epoch_size=1, epoch_mh=1, alpha=1.0, lr_sgd=0.0, lr_mh=0.0,
        writer=None
    ):
        super().__init__(
            model, loss, device, batch_size=batch_size)
        self._epoch_size = epoch_size
        self._epoch_mh = epoch_mh
        self.__alpha = alpha
        self.__objective = Module(
            "Objective", self.model, alpha=self.__alpha).fit
        self.optim = optim
        self.do_sgd = False
        self.lr_sgd = lr_sgd
        self.lr_mh = lr_mh

    def _optimize(self, batch):

        batch["alpha"] = self.__alpha

        def closure():
            self.model(batch)
            objective = self.__objective(self.model.out, batch["y"])
            loss = objective[0]
            self.optim.zero_grad()
            loss.backward()
            return objective

        self._loss, self._risk, loss, risk, measure = closure()
        self.optim.step(self._loss, self._risk, closure=closure)
        self._log["measure"] = measure.item()
        self._log["risk"] = risk.item()

        del batch["alpha"]

    def _meet_condition(self):
        if(self._epoch <= self._epoch_size-self._epoch_mh):
            self.optim.set_do_sgd(True)
            self.optim.set_lr(self.lr_sgd)
        else:
            self.optim.set_do_sgd(False)
            self.optim.set_lr(self.lr_mh)

        if(self._epoch > self._epoch_size):
            return True
        return False


###############################################################################
