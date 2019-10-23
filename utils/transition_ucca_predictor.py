import json
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from utils.transition_ucca_reader import parse_sentence


@Predictor.register('transition_predictor_ucca')
class UCCAParserPredictor(Predictor):
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

    # def predict_json(self, inputs: JsonDict) -> JsonDict:
    #     instance = self._json_to_instance(inputs)
    #     return self.predict_instance(instance)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """

        ret = parse_sentence(json.dumps(json_dict))

        tokens = ret["tokens"]
        meta_info = ret["meta_info"]
        tokens_range = ret["tokens_range"]

        return self._dataset_reader.text_to_instance(tokens=tokens, meta_info=[meta_info], tokens_range=tokens_range)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        ret_dict = ucca_trans_outputs_into_mrp(outputs)
        return sanitize(ret_dict)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = [[] for i in range(len(outputs_batch))]
        for outputs_idx in range(len(outputs_batch)):
            try:
                ret_dict_batch[outputs_idx] = ucca_trans_outputs_into_mrp(outputs_batch[outputs_idx])
            except:
                print('graph_id:' + json.loads(outputs_batch[outputs_idx]["meta_info"])['id'])

        return sanitize(ret_dict_batch)


def ucca_trans_outputs_into_mrp(outputs):
    edge_list = outputs["edge_list"]
    tokens = outputs["tokens"]
    meta_info = outputs["meta_info"]
    top_node = outputs["top_node"]
    concept_node = outputs["concept_node"]
    tokens_range = outputs["tokens_range"]

    meta_info_dict = json.loads(meta_info)
    ret_dict = {}
    for key in ["id", "flavor", "version", "framework", "time", "input"]:
        ret_dict[key] = meta_info_dict[key]
    ret_dict['flavor'] = 1

    ###Alignment between layer-0-token and layer-1-node

    # Based on 'Terminal' edge label
    terminal_dict = {}
    # value: token_id, key: node_id
    for concept_node_id in concept_node:
        terminal_dict[concept_node_id] = []
    for edge_info in edge_list:
        if edge_info[-1] == "Terminal" and edge_info[0] < len(tokens) and edge_info[1] >= len(tokens):
            terminal_dict[edge_info[1]].append(edge_info[0])

    ###Node Labels /Properties /Anchoring
    nodes_info = []

    projection_dict = {}
    # project original node index in concept into mrp indx

    # step1: extract node which links to surface token
    layer_0_nodes = []
    for concept_node_id in concept_node:
        if len(terminal_dict[concept_node_id]) == 0:  # skip non-terminal node
            continue
        node_anchors = []
        for lay_0_token_idx in terminal_dict[concept_node_id]:
            node_anchors.append({"from": tokens_range[lay_0_token_idx][0], "to": tokens_range[lay_0_token_idx][1]})
        nodes_info.append({"anchors": node_anchors, "id": len(layer_0_nodes)})
        projection_dict[concept_node_id] = len(layer_0_nodes)
        layer_0_nodes.append(concept_node_id)

    layer_1_nodes = []
    for concept_node_id in concept_node:
        if len(terminal_dict[concept_node_id]) != 0:  # skip terminal node
            continue
        nodes_info.append({"id": len(layer_0_nodes) + len(layer_1_nodes)})
        projection_dict[concept_node_id] = len(layer_0_nodes) + len(layer_1_nodes)
        layer_1_nodes.append(concept_node_id)

    assert sorted(layer_0_nodes + layer_1_nodes) == sorted(concept_node)

    ret_dict["nodes"] = nodes_info

    ###Directed Edges /Edge Labels /Edge Properties
    # 1. need post-processing the mutli-label edge
    # 2. edge prorperty, i.e remote-edges
    edges_info = []
    for edge_info in edge_list:
        for edge_label in edge_info[-1].split('+'):
            if edge_label == "Terminal" or edge_info[0] < len(tokens) or edge_info[1] < len(tokens):
                continue
            # remote edge
            if '*' in edge_info[-1]:
                edges_info.append({"label": edge_label[0],
                                   "source": projection_dict[edge_info[1]],
                                   "target": projection_dict[edge_info[0]],
                                   "properties": ["remote"],
                                   "values": [True]
                                   })
            # primary edge
            else:
                edges_info.append({"label": edge_label[0],
                                   "source": projection_dict[edge_info[1]],
                                   "target": projection_dict[edge_info[0]]
                                   })
    ret_dict["edges"] = edges_info

    ###Top Nodes
    ret_dict["tops"] = [projection_dict[top_node[0]]]

    return ret_dict
