from dataclasses import dataclass

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class WandbMinMaxLogger(WandbLogger):
    """Extension of the WandbLogger that tracks minimum and maximum of all metrics over time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bounds_dict = {}

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Extends log_metrics with updating the wandb summary with maximum and minimum of each metric over time.

        :param metrics: mapping from metric names to values.
        :param step: step at which the metric was measured.
        """
        super().log_metrics(metrics, step)
        for metric, value in metrics.items():
            self._update_bounds(metric, value)
        self._log_bounds()

    def _update_bounds(self, name, value):
        """Given a metric value and the name of a metric, update the metric´s new min and max values.

        :param name: name/identifier of the metric.
        :param value: newly recorded value.
        """
        bounds = self._bounds_dict.get(name, self.__MetricBounds(name, value, value))
        bounds.update(value)
        self._bounds_dict[name] = bounds

    def _log_bounds(self):
        """Log the current min and max bounds for each metric in the wandb summary.
        """
        summary = self.experiment.summary
        for bounds in self._bounds_dict.values():
            summary[bounds.min_name] = bounds.min_value
            summary[bounds.max_name] = bounds.max_value

    @dataclass
    class __MetricBounds:
        """Holds minimum and maximum value of a metric over time.

        :param name: name of the metric.
        :param max_value: current maximum of the metric.
        :param min_value: current minimum of the metric.
        """
        name: str
        max_value: int
        min_value: int

        def update(self, value):
            """update the current min and max values based on a new value.

            :param value: value to compare min and max to.
            """
            self.max_value = max(self.max_value, value)
            self.min_value = min(self.min_value, value)

        def __str__(self):
            return f'{self.name}: (min: {self.min_value}, max: {self.max_value})'

        @property
        def min_name(self):
            """Name for the minimum value of this metric.
            """
            return 'min_' + self.name

        @property
        def max_name(self):
            """Name for the maximum value of this metric.
            """
            return 'max_' + self.name
