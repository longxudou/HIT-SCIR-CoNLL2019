import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'toolkit', 'mtool'))

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from graph import Graph
from metrics.mces import MCES


@Metric.register("mces")
class MCESScore(Metric):
    """
    output_type: chose which type of info to output,
    including g,s,c,p,r,f
    """

    def __init__(self, output_type: str = 'f',
                 cores=0, trace=0) -> None:
        self.output_type = output_type
        self.mces = MCES(cores=cores, trace=trace)

    @overrides
    def __call__(self,
                 predictions: List[Dict[str, Any]],
                 golds: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        predictions : the predicted graph (of mrp type)
        golds : the gold graph (just use the input mrp file loaded with json.loads)
        """

        pred_graphs = self.read_graphs(predictions)
        gold_graphs = self.read_graphs(golds)
        self.mces.evaluate(gold_graphs, pred_graphs)

    def get_metric(self, reset: bool = False):
        result_ = self.mces.get_metric()
        result = {}
        for key in result_:
            if isinstance(result_[key], float):
                result[key] = result_[key]
            else:
                if isinstance(result_[key], dict):
                    for subkey in result_[key].keys():
                        if subkey in self.output_type:
                            result[key + '-' + subkey] = result_[key][subkey]
        if reset:
            self.mces.reset()
        return result

    def read_graphs(self, sents: List[Dict[str, Any]]):
        generator = self.read(sents)
        graphs = []
        while True:
            try:
                graph = next(generator);
                graphs.append(graph);
            except StopIteration:
                break;
        return graphs

    def read(self, sents: List[Dict[str, Any]]):
        for i, sent in enumerate(sents):
            try:
                graph = Graph.decode(sent)
                yield graph
            except Exception as error:
                print("mrp_score.read(): ignoring line {}: {}"
                      "".format(i, error), file=sys.stderr);
