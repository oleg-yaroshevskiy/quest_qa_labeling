from poutyne.framework.callbacks import Callback, CSVLogger


class LosswiseSessionHandler:
    def __init__(self, api_key, tag="", params=None):
        import losswise

        losswise.set_api_key(api_key)
        self._session = losswise.Session(
            tag=tag, params=params if params is not None else {}, track_git=False
        )
        self._graphs = dict()

    def create_graph(
        self, graph_name, xlabel="", ylabel="", kind=None, display_interval=1
    ):
        assert isinstance(graph_name, str)
        if graph_name not in self._graphs:
            self._graphs[graph_name] = self._session.graph(
                title=graph_name,
                xlabel=xlabel,
                ylabel=ylabel,
                kind=kind,
                display_interval=display_interval,
            )
        return self._graphs[graph_name]

    def __getitem__(self, graph_name):
        if graph_name not in self._graphs:
            self.create_graph(graph_name)
        return self._graphs[graph_name]

    def done(self):
        self._session.done()


class LosswiseCallback(Callback):
    def __init__(
        self,
        api_key: str = None,
        losswise_session: LosswiseSessionHandler = None,
        prefix="",
        tag="my awesome DL",
        period=1,
        keep_open=False,
        training_params=None,
        param_groups=dict(),
    ):
        super().__init__()
        assert (api_key is None) ^ (losswise_session is None)

        if losswise_session is None:
            self._session = LosswiseSessionHandler(
                api_key, tag=tag, params=training_params
            )
        else:
            self._session = losswise_session
        self._keep_open = keep_open
        self.period = period
        self.prefix = prefix
        self.param_groups = param_groups
        self.steps_elapsed = 0

    def on_train_begin(self, logs):
        self.metrics = ["loss"] + self.model.metrics_names

        self._session.create_graph("loss", xlabel="epoch", kind="min")
        self._session.create_graph("learning rate", xlabel="batch")
        for name in self.model.metrics_names:
            self._session.create_graph(name, xlabel="epoch")
            self._session.create_graph(
                name + "_iter", xlabel="batch", display_interval=self.params["steps"]
            )
        for name in self.param_groups:
            self._session.create_graph(name, xlabel="epoch")

    def on_train_end(self, logs):
        if not self._keep_open:
            self._session.done()

    def on_epoch_end(self, epoch, logs):
        for name in self.metrics:
            graph_args = dict()
            if name in logs:
                graph_args[self.prefix + name] = logs[name]
            if "val_" + name in logs:
                graph_args[self.prefix + "val_" + name] = logs["val_" + name]
            self._session[name].append(epoch, graph_args)

        for group, names in self.param_groups.items():
            for name in names:
                graph_args = dict()
                if name in logs:
                    graph_args[self.prefix + name] = logs[name]
                if "val_" + name in logs:
                    graph_args[self.prefix + "val_" + name] = logs["val_" + name]
                self._session[group].append(epoch, graph_args)

    def on_batch_end(self, batch, logs):
        self.steps_elapsed += 1
        if self.steps_elapsed % self.period == 0:
            for name in self.metrics:
                if name in logs:
                    self._session[name + "_iter"].append(
                        self.steps_elapsed, {self.prefix + name: logs[name]}
                    )

            if hasattr(self.model.optimizer, "get_lr"):
                learning_rates = [self.model.optimizer.get_lr()[0]]
            else:
                learning_rates = (
                    param_group["lr"]
                    for param_group in self.model.optimizer.param_groups
                )
            for group_idx, lr in enumerate(learning_rates):
                self._session["learning rate"].append(
                    self.steps_elapsed, {"lr_param_group_" + str(group_idx): lr}
                )


class CSVParamLogger(CSVLogger):
    def __init__(
        self,
        filename,
        *,
        batch_granularity=False,
        separator=",",
        append=False,
        extra_metrics=[]
    ):
        super(CSVParamLogger, self).__init__(
            filename,
            batch_granularity=batch_granularity,
            separator=separator,
            append=append,
        )
        self.extra_metrics = extra_metrics

    def on_train_begin(self, logs):
        metrics = ["loss"] + self.model.metrics_names

        if self.batch_granularity:
            self.fieldnames = ["epoch", "batch", "size", "time", "lr"]
        else:
            self.fieldnames = ["epoch", "time", "lr"]
        self.fieldnames += metrics
        self.fieldnames += ["val_" + metric for metric in metrics]
        self.fieldnames += self.extra_metrics
        self._on_train_begin_write(logs)
