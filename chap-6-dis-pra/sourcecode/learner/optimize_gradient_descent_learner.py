from learner.gradient_descent_learner import GradientDescentLearner


###############################################################################

class OptimizeGradientDescentLearner(GradientDescentLearner):

    def __init__(
        self, model, loss, criteria, optim, device,
        epoch=10, batch_size=None, writer=None
    ):
        super().__init__(
            model, device,
            epoch=epoch, batch_size=batch_size, writer=writer)

        self.optim = optim
        self.loss = loss
        self.criteria = criteria

    def _optimize(self, batch):
        self.model(batch)

        self._loss = self.loss(self.model.out, batch["y"])
        div_renyi = self.model.div_renyi
        div_rivasplata = self.model.div_rivasplata

        self.optim.zero_grad()
        self._loss.backward()
        self.optim.step()

        crit = self.criteria(
            self.model.pred.cpu().detach().numpy(), self._label.cpu().numpy())

        self._log = {
            "crit": crit,
            "div_renyi": div_renyi,
            "div_rivasplata": div_rivasplata
        }
        self._writer = {
            "crit": crit,
            "div_renyi": div_renyi.cpu().detach().numpy(),
            "div_rivasplata": div_rivasplata.cpu().detach().numpy(),
            "loss": self._loss.cpu().detach().numpy()
        }

###############################################################################
