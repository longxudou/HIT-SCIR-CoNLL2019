from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from utils.extract_mrp_dict import extract_mrp_dict


@Predictor.register('transition_predictor_amr')
class AMRParserPredictor(Predictor):

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.
        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        return self.predict_batch_instance([instance])[0]

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = []
        for outputs_idx in range(len(outputs_batch)):
            outputs = outputs_batch[outputs_idx]
            instance = instances[outputs_idx]

            existing_edges = outputs["existing_edges"]
            node_labels = outputs["node_labels"]
            id_cnt = outputs["id_cnt"]
            tokens = outputs["tokens"]
            metadata = outputs["metadata"]
            node_types = outputs["node_types"]
            sent_len = outputs["sent_len"]

            ret_dict = extract_mrp_dict(existing_edges=existing_edges,
                                        sent_len=sent_len,
                                        id_cnt=id_cnt,
                                        node_labels=node_labels,
                                        node_types=node_types,
                                        metadata=metadata)

            ret_dict_batch.append(ret_dict)

        return sanitize(ret_dict_batch)
