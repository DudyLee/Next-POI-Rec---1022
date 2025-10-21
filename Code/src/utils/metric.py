import datasets
from .evaluate import evaluate_predictions_batch, get_metrics_results, evaluate_predictions_batch_filter
from .utils import extract_history


_CITATION = """
"""

_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RecMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def set_metrics(self, metrics):
        self.metrics = metrics

    def _compute(self, predictions, references, inputs):
        metrics = self.metrics.split(',')
        generate_num = max([int(m.split('@')[1]) for m in metrics])
        historys = [extract_history(i, ', ') for i in inputs]
        results = evaluate_predictions_batch_filter(historys, predictions, references, generate_num)
        metrics_res = get_metrics_results(results, metrics)
        metrics_res = metrics_res/len(references)
        output = {m: metrics_res[i] for i, m in enumerate(metrics)}
        return output