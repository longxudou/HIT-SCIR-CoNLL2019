import json
import re
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from utils.transition_eds_reader import parse_sentence


@Predictor.register('transition_predictor_eds')
class EDSParserPredictor(Predictor):
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
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """

        json_dict["prediction"] = True
        ret = parse_sentence(json.dumps(json_dict))

        tokens = ret["tokens"]
        meta_info = ret["meta_info"]
        tokens_range = ret["tokens_range"]

        return self._dataset_reader.text_to_instance(tokens=tokens, meta_info=[meta_info], tokens_range=tokens_range)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        ret_dict = eds_trans_outputs_into_mrp(outputs)
        return sanitize(ret_dict)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = [[] for i in range(len(outputs_batch))]
        for outputs_idx in range(len(outputs_batch)):
            try:
                ret_dict_batch[outputs_idx] = eds_trans_outputs_into_mrp(outputs_batch[outputs_idx])
            except:
                print('graph_id:' + json.loads(outputs_batch[outputs_idx]["meta_info"])['id'])

        return sanitize(ret_dict_batch)


def eds_trans_outputs_into_mrp(outputs):
    edge_list = outputs["edge_list"]
    tokens = outputs["tokens"]
    meta_info = outputs["meta_info"]
    tokens_range = outputs["tokens_range"]
    top_node_list = outputs["top_node"]
    concept_node_list = outputs["concept_node"]

    meta_info_dict = json.loads(meta_info)
    ret_dict = {}
    for key in ["id", "flavor", "version", "framework", "time", "input"]:
        ret_dict[key] = meta_info_dict[key]
    ret_dict['flavor'] = 1

    ###Node Labels /Properties /Anchoring
    nodes_info_list = []

    node_cnt = 0
    processing_node_dict = {}
    for node_idx, node_info in concept_node_list.items():
        if "end" in node_info:
            crag_ret = get_carg_value(node_info["label"],
                                      ' '.join(tokens[node_info["start"]:node_info["end"] + 1]))
            if crag_ret != False:
                nodes_info_list.append(
                    {"anchors": [
                        {"from": tokens_range[node_info["start"]][0], "to": tokens_range[node_info["end"]][1]}], \
                        "id": node_cnt,
                        "label": node_info["label"],
                        "properties": ["carg"],
                        "values": [crag_ret]})
            else:
                nodes_info_list.append(
                    {"anchors": [
                        {"from": tokens_range[node_info["start"]][0], "to": tokens_range[node_info["end"]][1]}], \
                        "id": node_cnt,
                        "label": node_info["label"]})
            processing_node_dict[node_idx] = node_cnt
            node_cnt += 1

    ret_dict["nodes"] = nodes_info_list
    ###Directed Edges /Edge Labels /Edge Properties
    # 1. need post-processing the mutil-label edge
    # 2. edge prorperty, i.e remote-edges
    edges_info = []
    for edge_info in edge_list:
        if edge_info[0] in processing_node_dict and edge_info[1] in processing_node_dict:
            edges_info.append({"label": edge_info[-1],
                               "source": processing_node_dict[edge_info[1]],
                               "target": processing_node_dict[edge_info[0]],
                               })

    ret_dict["edges"] = edges_info

    ###Top Nodes
    try:
        if top_node_list in processing_node_dict:
            ret_dict["tops"] = [processing_node_dict[top_node_list]]
    except:
        ret_dict["tops"] = []

    return ret_dict


def get_carg_value(label, token):
    if label not in ['named', 'card', 'mofy', 'dofm', 'yofc', 'year_range', 'named_n', \
                     'dofw', 'numbered_hour', 'season', 'ord', 'fraction', 'excl', 'holiday', \
                     '_pm_x', 'timezone_p', '_am_x', 'polite', 'meas_np']:
        return False

    int_dict = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'eleven': '11',
        'twelve': '12',
        'hundred': '100',
        'thousand': '1000',
        'million': '1000000',
        'billion': '1000000000',
        'trillion': '1000000000000',
        "both": '2',
        "dozen": '12',
        "ones": '2',
    }

    punctuation = {".", "?", "!", ";", ",", ":",
                   "“", "\"", "”", "‘", "'", "’",
                   "(", ")", "[", "]", "{", "}",
                   " ", "\t", "\n", "\f"}

    punctuation_str = ''.join(list(punctuation))

    if label == '_am_x' and token == 'a.m.':
        value = "am_time"
    elif label == '_pm_x' and token == 'p.m.':
        value = "pm_time"
    elif label in ['card']:
        value = token.strip(punctuation_str).lower()
        for value_sp in value.split('-'):
            if value_sp in int_dict:
                value = int_dict[value_sp]
                break

            elif len(re.findall('\d+', value_sp)) > 0:
                value = value_sp.strip(punctuation_str.replace(',', ''))
                break

    elif label == 'dofm':
        value = token.strip(punctuation_str).lower()
        for value_sp in value.split('-'):
            if value_sp in int_dict:
                value = int_dict[value_sp]
                break
            elif len(re.findall('\d+', value_sp)) > 0:
                value = value_sp.strip(punctuation_str.replace(',', ''))
                break

    elif label == 'dofw':
        value = token.strip(punctuation_str + 's').lower()
        week_dict = {"sunday": "Sun",
                     "monday": "Mon",
                     "tuesday": "Tue",
                     "wednesday": "Wed",
                     "thursday": "Thu",
                     "friday": "Fri",
                     "saturday": "Sat"}
        if value in week_dict:
            value = week_dict[value]

    elif label == 'excl':
        value = token.strip(punctuation_str)
    elif label == 'fraction':
        value = token.strip(punctuation_str)
    elif label == 'holiday':
        value = token.strip(punctuation_str).replace(' ', '+')
    elif label == 'meas_np':
        value = token.strip(punctuation_str)
    elif label == 'mofy':
        value = token.strip(punctuation_str + 's')

        month_dict = {"january": "Jan",
                      "february": "Feb",
                      "march": "Mar",
                      "april": "Apr",
                      "may": "May",
                      "june": "Jun",
                      "july": "Jul",
                      "august": "Aug",
                      "september": "Sep",
                      "october": "Oct",
                      "november": "Nov",
                      "december": "Dec"}

        if value in list(month_dict.values()):
            return value

        for value_sp in value.lower().split():
            if value in month_dict:
                value = month_dict[value_sp]

    elif label == 'named':
        # W.D.) -> W.D.
        if len(re.findall(r'\.', token)) > 1:
            value = token.strip(punctuation_str.replace('.', ''))
        # James. -> James
        else:
            for value_sp in token.strip(punctuation_str).replace(' ', '+').split('-'):
                value = value_sp
                break

    elif label == 'named_n':
        if len(re.findall("U\.S\.", token, re.I)) > 0:
            value = 'US'
        elif len(re.findall("U\.N\.", token, re.I)) > 0:
            value = 'UN'
        elif len(re.findall("U\.K\.", token, re.I)) > 0:
            value = 'UK'
        elif len(re.findall("underground", token, re.I)) > 0:
            value = 'Underground'
        else:
            for value_sp in token.strip(punctuation_str).replace(' ', '+').split('-'):
                value = value_sp
                break

    elif label == 'numbered_hour':
        value = token.strip(punctuation_str).lower()
        hour_dict = {
            "midnight": '0',
            "noon": '12',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
            'eleven': '11',
            'twelve': '12'
        }
        if value in hour_dict:
            value = hour_dict[value]

    elif label == 'ord':
        value = token.strip(punctuation_str).lower()
        ord_dict = {
            "first": "1",
            "second": "2",
            "third": "3",
            "fourth": "4",
            "fifth": "5",
            "sixth": "6",
            "seventh": "7",
            "eighth": "8",
            "ninth": "9",
            "tenth": "10"
        }

        for value_sp in value.split('-'):
            if value_sp in ord_dict:
                value = ord_dict[value_sp]
                break
            elif len(re.findall("th", value, re.I)) > 0:
                value = value_sp
                break
            else:
                pass

    elif label == 'polite':
        value = token.strip(punctuation_str).lower()

    elif label == 'season':
        value = token.strip(punctuation_str).lower()
        if len(re.findall("Christmas", token, re.I)) > 0:
            value = 'Christmas'

    elif label == 'timezone_p':
        value = token.strip(punctuation_str).lower()

    elif label == 'yofc' or label == 'year_range':
        value = token.strip(punctuation_str).lower().split('-')
        if value[0] == 'mid':
            value = value[1]
        else:
            value = value[0]

    else:
        value = token.strip(punctuation_str).lower()

    return value
