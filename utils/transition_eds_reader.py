import copy
import json
import logging
from collections import OrderedDict
from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from conllu.parser import parse_line, DEFAULT_FIELDS
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

label_prior = ['ARG1', 'ARG2', 'ARG3', 'R-HNDL', 'L-HNDL']
label_prior_dict = {label_prior[idx]: idx for idx in range(len(label_prior))}


class Relation(object):
    type = None

    def __init__(self, node, rel, remote=False):
        self.node = node
        self.rel = rel
        self.remote = remote

    def show(self):
        print("Node:{},Rel:{},Is remote:{} || ".format(self.node, self.rel, self.remote), )


class Head(Relation): type = 'HEAD'


class Child(Relation): type = 'CHILD'


class Node(object):
    def __init__(self, info):
        self.id = info["id"]
        self.anchored = False
        self.anchors = []
        self.label = info["label"]
        if "properties" in info:
            assert len(info["properties"]) == 1
        self.properties = info["properties"][0] if "properties" in info else None
        self.values = info["values"][0] if "values" in info else None

        if "anchors" in info:
            self.anchors = [(anc["from"], anc["to"]) for anc in info["anchors"]]
            self.anchored = True

        self.heads, self.childs = [], []
        self.head_ids, self.child_ids = [], []

        return

    def add_head(self, edge):
        assert edge["target"] == self.id
        remote = False
        if "properties" in edge and "remote" in edge["properties"]:
            remote = True
        if edge["source"] in self.head_ids:
            self.heads.append(Head(edge["source"], edge["label"], remote))
            # print("Multiple arcs between two nodes!")
            return True
        self.heads.append(Head(edge["source"], edge["label"], remote))
        self.head_ids.append(edge["source"])
        return False

    def add_child(self, edge):
        assert edge["source"] == self.id
        remote = False
        if "properties" in edge and "remote" in edge["properties"]:
            remote = True
        if edge["target"] in self.child_ids:
            self.childs.append(Child(edge["target"], edge["label"], remote))
            # print("Multiple arcs between two nodes!")
            return True
        self.childs.append(Child(edge["target"], edge["label"], remote))
        self.child_ids.append(edge["target"])
        return False


class Graph(object):
    def __init__(self, js):
        self.id = js["id"]
        self.input = js["input"]
        self.top = js["tops"] if "tops" in js else None
        self.companion = js["companion"]

        self.nodes = {}
        if 'nodes' in js:
            for node in js["nodes"]:
                self.nodes[node["id"]] = Node(node)

        self.edges = {}
        self.multi_arc = False
        if 'edges' in js:
            for edge in js["edges"]:
                multi_arc_child = self.nodes[edge["source"]].add_child(edge)
                multi_arc_head = self.nodes[edge["target"]].add_head(edge)

                if multi_arc_child or multi_arc_head:
                    self.multi_arc = True

        self.meta_info = json.dumps(js)

        self.gold_mrps = copy.deepcopy(js)
        self.gold_mrps.pop('companion')

        self.prediction = True if "prediction" in js else False

    def get_childs(self, id):
        childs = self.nodes[id].childs
        child_ids = [c.node for c in childs]
        return childs, child_ids

    def extract_token_info_from_companion_data(self):
        annotation = []
        for line in self.companion:
            line = '\t'.join(line)
            annotation.append(parse_line(line, DEFAULT_FIELDS))

        tokens = [x["form"] for x in annotation if x["form"] is not None]
        lemmas = [x["lemma"] for x in annotation if x["lemma"] is not None]
        pos_tags = [x["upostag"] for x in annotation if x["upostag"] is not None]
        token_range = [tuple([int(i) for i in list(x["misc"].values())[0].split(':')]) for x in annotation]

        return {"tokens": tokens,
                "lemmas": lemmas,
                "pos_tags": pos_tags,
                "token_range": token_range}

    def has_cross_arc(self):
        tokens_range = []
        for node_id, node_info in self.nodes.items():
            tokens_range.append(node_info.anchors[0])

        for i in range(len(tokens_range)):
            for j in range(i + 1, len(tokens_range)):
                if i == j:
                    continue
                if (tokens_range[i][1] > tokens_range[j][0] \
                    and tokens_range[i][1] < tokens_range[j][1] \
                    and tokens_range[i][0] < tokens_range[j][0]) or \
                        (tokens_range[j][1] > tokens_range[i][0] \
                         and tokens_range[j][1] < tokens_range[i][1] \
                         and tokens_range[j][0] < tokens_range[i][0]):
                    return True
        return False

    def get_arc_info(self):
        tokens, arc_indices, arc_tags = [], [], []
        concept_node = []

        token_info = self.extract_token_info_from_companion_data()

        tokens = token_info["tokens"]
        lemmas = token_info["lemmas"]
        pos_tags = token_info["pos_tags"]
        token_range = token_info["token_range"]

        # Step1: Construct the alignment between token and node
        # Attention: multiple nodes can have overlapping anchors
        alignment_dict = {}
        node_label_dict = {}
        for node_id, node_info in self.nodes.items():
            concept_node.append(node_id + len(tokens))

            alignment_dict[node_id + len(tokens)] = []
            node_label_dict[node_id + len(tokens)] = node_info.label

            assert len(node_info.anchors) == 1
            node_anchored_begin, node_anchored_end = node_info.anchors[0][0], node_info.anchors[0][1]
            for token_idx in range(len(token_range)):
                token_anchored_begin, token_anchored_end = token_range[token_idx][0], token_range[token_idx][1]
                if node_anchored_begin > token_anchored_end or node_anchored_end < token_anchored_begin:
                    continue
                if token_anchored_begin >= node_anchored_begin and token_anchored_end <= node_anchored_end:
                    alignment_dict[node_id + len(tokens)].append(token_idx)

                # check if suffix alignment exists
                # Example case:
                # Node anchor: 'of child'
                # Sentence: 'Take of children'
                if (node_anchored_end > token_anchored_begin and node_anchored_end < token_anchored_end) or \
                        (node_anchored_begin < token_anchored_end and node_anchored_begin > token_anchored_begin):
                    print((node_anchored_begin, node_anchored_end), '-->',
                          self.input[node_anchored_begin:node_anchored_end], \
                          (token_anchored_begin, token_anchored_end), '-->',
                          self.input[token_anchored_begin:token_anchored_end])

        # Step2: Link node and its align token(s) via alignment_dict
        # Add Terminal Edge
        # for node_id,alignment_tokens in alignment_dict.items():
        #     for token_idx in alignment_tokens:
        #         arc_indices.append((token_idx,node_id))
        #         arc_tags.append('Terminal')

        # Step3: Multi-Label Arc
        childs_dict = {node_id: {} for node_id in self.nodes.keys()}
        for node_id, node_info in self.nodes.items():
            for child_of_node_info in node_info.childs:
                child_node = child_of_node_info.node
                arc_tag = child_of_node_info.rel

                # the arc with one label
                if child_node not in childs_dict[node_id]:
                    childs_dict[node_id][child_node] = arc_tag
                # the arc with multi label
                # aggregate the multi-label by label prior, defined in the start in this file
                else:
                    # expand the label_prior_dict. this only happens when occur n-label arc (n>2)
                    if childs_dict[node_id][child_node] not in label_prior_dict:
                        label_prior_dict[childs_dict[node_id][child_node]] = len(label_prior_dict)

                    if label_prior_dict[arc_tag] < label_prior_dict[childs_dict[node_id][child_node]]:
                        arc_tag = arc_tag + '+' + childs_dict[node_id][child_node]
                    else:
                        arc_tag = childs_dict[node_id][child_node] + '+' + arc_tag

                    childs_dict[node_id][child_node] = arc_tag

        # Step4: Add Label between node
        for node_id, node_info in self.nodes.items():
            for child_node in childs_dict[node_id]:
                arc_tag = childs_dict[node_id][child_node]

                arc_indices.append((child_node + len(tokens), node_id + len(tokens)))
                arc_tags.append(arc_tag)

        # Step 5: rank node by interval
        node_range_dict = {}
        for node_id, node_info in self.nodes.items():
            node_anchored_begin, node_anchored_end = node_info.anchors[0][0], node_info.anchors[0][1]
            node_range_dict[node_id + len(tokens)] = (node_anchored_begin, node_anchored_end)

        node_range_dict = OrderedDict(sorted(node_range_dict.items(), key=lambda x: (x[1][0], -x[1][1])))

        node_info_dict = {"alignment_dict": alignment_dict,
                          "node_range_dict": node_range_dict,
                          "node_label_dict": node_label_dict,
                          "graph_id": self.id}

        ret = {"tokens": tokens,
               "arc_indices": arc_indices,
               "arc_tags": arc_tags,
               "concept_node": concept_node,
               "root_id": self.top[0] + len(tokens) if self.top is not None else None,
               "lemmas": lemmas,
               "pos_tags": pos_tags,
               "node_info_dict": node_info_dict,
               "graph_id": self.id,
               "meta_info": self.meta_info,
               "tokens_range": token_range,
               "gold_mrps": self.gold_mrps}

        return ret


def parse_sentence(sentence_blob: str):
    graph = Graph(json.loads(sentence_blob))
    if graph.has_cross_arc() and graph.prediction == False:
        return False
    ret = graph.get_arc_info()

    return ret


def lazy_parse(text: str):
    for sentence in text.split("\n"):
        if sentence:
            ret = parse_sentence(sentence)
            if ret == False:
                continue
            yield ret


@DatasetReader.register("eds_reader_conll2019")
class EDSDatasetReaderConll2019(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 action_indexers: Dict[str, TokenIndexer] = None,
                 arc_tag_indexers: Dict[str, TokenIndexer] = None,
                 concept_label_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._lemma_indexers = None
        if lemma_indexers is not None and len(lemma_indexers) > 0:
            self._lemma_indexers = lemma_indexers

        self._action_indexers = None
        if action_indexers is not None and len(action_indexers) > 0:
            self._action_indexers = action_indexers

        self._arc_tag_indexers = None
        if arc_tag_indexers is not None and len(arc_tag_indexers) > 0:
            self._arc_tag_indexers = arc_tag_indexers

        self._concept_label_indexers = concept_label_indexers or {
            'concept_label': SingleIdTokenIndexer(namespace='concept_label')}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r', encoding='utf8') as eds_file:
            logger.info("Reading EDS instances from conllu dataset at: %s", file_path)
            for ret in lazy_parse(eds_file.read()):
                tokens = ret["tokens"]
                arc_indices = ret["arc_indices"]
                arc_tags = ret["arc_tags"]
                root_id = ret["root_id"]
                lemmas = ret["lemmas"]
                pos_tags = ret["pos_tags"]
                meta_info = ret["meta_info"]
                node_info_dict = ret["node_info_dict"]
                tokens_range = ret["tokens_range"]
                gold_mrps = ret["gold_mrps"]

                concept_node = ret["concept_node"]
                gold_actions = get_oracle_actions(tokens, arc_indices, arc_tags, root_id, concept_node, node_info_dict) if arc_indices else None

                # if len(gold_actions) / len(tokens) > 20:
                #     print(len(gold_actions) / len(tokens))

                if gold_actions and gold_actions[-1] == '-E-':
                    print('-E-', ret["graph_id"])
                    continue

                concept_label_list = list(node_info_dict["node_label_dict"].values())
                yield self.text_to_instance(tokens, lemmas, pos_tags, arc_indices, arc_tags, gold_actions,
                                            [root_id], [meta_info], concept_label_list, tokens_range, [gold_mrps])

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         lemmas: List[str] = None,
                         pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         gold_actions: List[str] = None,
                         root_id: List[int] = None,
                         meta_info: List[str] = None,
                         concept_label: List[int] = None,
                         tokens_range: List[Tuple[int, int]] = None,
                         gold_mrps: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)

        fields["tokens"] = token_field
        meta_dict = {"tokens": tokens}

        if lemmas is not None and self._lemma_indexers is not None:
            fields["lemmas"] = TextField([Token(l) for l in lemmas], self._lemma_indexers)
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")

        if arc_indices is not None and arc_tags is not None:
            meta_dict["arc_indices"] = arc_indices
            meta_dict["arc_tags"] = arc_tags
            fields["arc_tags"] = TextField([Token(a) for a in arc_tags], self._arc_tag_indexers)

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        if meta_info is not None:
            meta_dict["meta_info"] = meta_info[0]

        if gold_mrps is not None:
            meta_dict["gold_mrps"] = gold_mrps[0]

        if tokens_range is not None:
            meta_dict["tokens_range"] = tokens_range

        if concept_label is not None:
            meta_dict["concept_label"] = concept_label
            fields["concept_label"] = TextField([Token(a) for a in concept_label], self._concept_label_indexers)

        if root_id is not None:
            meta_dict["root_id"] = root_id[0]

        fields["metadata"] = MetadataField(meta_dict)

        return Instance(fields)


def get_oracle_actions(tokens, arc_indices, arc_tags, root_id, concept_node, node_info_dict):
    actions = []
    stack = []
    buffer = []
    deque = []
    generated_order = {-1: -1}

    total_node_num = len(tokens) + len(concept_node)
    N = len(tokens)
    for i in range(N - 1, -1, -1):
        buffer.append(i)

    graph = {}
    for token_idx in range(total_node_num):
        graph[token_idx] = []

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    whole_graph = [[False for i in range(total_node_num)] for j in range(total_node_num)]
    for arc, arc_tag in zip(arc_indices, arc_tags):
        graph[arc[0]].append((arc[1], arc_tag))
        whole_graph[arc[0]][arc[1]] = True

    # i:head_point j:child_point
    top_down_graph = [[] for i in range(total_node_num)]  # N real point, 1 root point, concept_node

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = [[False for i in range(total_node_num)] for j in range(total_node_num)]
    sub_graph_arc_list = []

    for i in range(total_node_num):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    # auxiliary list for START and END op
    alignment_dict = node_info_dict["alignment_dict"]
    node_range_dict = node_info_dict["node_range_dict"]
    node_label_dict = node_info_dict["node_label_dict"]

    begin_dict = {}
    end_dict = {}
    # key:token id, value: list of node_id

    node_begin_dict = {}
    node_end_dict = {}
    for token_id in range(len(tokens)):
        node_begin_dict[token_id] = {}
        node_end_dict[token_id] = {}

    for order_node_id in node_range_dict.keys():
        begin_dict[order_node_id] = alignment_dict[order_node_id][0]
        end_dict[order_node_id] = alignment_dict[order_node_id][-1]

        node_begin_dict[begin_dict[order_node_id]][order_node_id] = False
        node_end_dict[end_dict[order_node_id]][order_node_id] = False

    node_align_begin_flag = {}
    node_align_end_flag = {}
    for node_id in range(len(concept_node)):
        node_align_begin_flag[node_id + len(tokens)] = False
        node_align_end_flag[node_id + len(tokens)] = False

    # return if w1 is one head of w0
    def has_head(w0, w1):
        if w0 < 0 or w1 < 0:
            return False
        for w in graph[w0]:
            if w[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    # return if w has any unfound head
    def lack_head(w):
        if w < 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return the relation between child: w0, head: w1
    def get_arc_label(w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_node_label(w0):
        return node_label_dict[w0]

    def check_graph_finish():
        return whole_graph == sub_graph

    def check_sub_graph(w0, w1):
        if w0 < 0 or w1 < 0:
            return False
        else:
            return sub_graph[w0][w1] == False

    def is_surface_token(token):
        return token < len(tokens) and token >= 0

    def is_concept_node(token):
        return token >= len(tokens)

    def start_generate_node(token):
        if is_surface_token(token):
            for concept_node_id, concept_node_status in node_begin_dict[token].items():
                if concept_node_status == False:
                    return concept_node_id
        return -1

    def end_generate_node(token):
        if is_surface_token(token):
            concept_node_id_list = []
            for concept_node_id, concept_node_status in node_end_dict[token].items():
                if concept_node_status == False:
                    concept_node_id_list.append(concept_node_id)
            if len(concept_node_id_list) > 0:
                return concept_node_id_list
        return [-1]

    def finish_alignment_token(token):
        if not is_surface_token(token):
            return False
        return start_generate_node(token) == -1 and end_generate_node(token) == [-1]

    def finish_alignment_node(node):
        if not is_concept_node(node):
            return False

        begin_align_flag = node_align_begin_flag[node]
        end_align_flag = node_align_end_flag[node]
        return begin_align_flag and end_align_flag

    def lack_end_align(node):
        if not is_concept_node(node):
            return False
        return node_align_end_flag[node] == False

    def generate_all_concept_node():
        for node in concept_node:
            if node_align_end_flag[node] == False:
                return False
            if node_align_begin_flag[node] == False:
                return False
        return True

    def find_end_align_of_node(node):
        if not is_concept_node(node):
            return -1, -1

        buffer_token = alignment_dict[node][-1]
        buffer_position = buffer.index(buffer_token)

        return buffer_position, buffer_token

    def find_end_align_of_token(token):
        if not is_surface_token(token) or end_generate_node(token) == [-1]:
            return False

        end_generate_node_list = end_generate_node(token)

        for node in stack:
            if node in end_generate_node_list:
                stack_token = node
                stack_position = stack.index(stack_token)
                return stack_token, stack_position

        return False

    def find_all_greater_edge(node):

        for node_id, node_order in generated_order.items():
            # skip self-node and symbol-node in generate_order dict, i.e. -1
            if node_id == node or node_id == -1:
                continue

            if (has_head(node_id, node) and check_sub_graph(node_id, node)) or \
                    (has_head(node, node_id) and check_sub_graph(node, node_id)):
                return False

        return True

    def get_oracle_actions_onestep(sub_graph, stack, buffer, actions, root_id):

        s0 = stack[-1] if len(stack) > 0 else -1
        s1 = stack[-2] if len(stack) > 1 else -1
        b0 = buffer[-1] if len(buffer) > 0 else -1

        # LEFT
        if has_head(s0, b0) and check_sub_graph(s0, b0) and is_concept_node(b0):
            actions.append("LEFT-EDGE#SPLIT_TAG#" + get_arc_label(s0, b0))
            sub_graph[s0][b0] = True
            sub_graph_arc_list.append((s0, b0))
            return

        # RIGHT_EDGE
        elif has_head(b0, s0) and check_sub_graph(b0, s0) and is_concept_node(b0):
            actions.append("RIGHT-EDGE#SPLIT_TAG#" + get_arc_label(b0, s0))
            sub_graph[b0][s0] = True
            sub_graph_arc_list.append((b0, s0))
            return

        # SELF-EDGE
        elif has_head(s0, s0) and check_sub_graph(s0, s0) and is_concept_node(s0):
            actions.append("SELF-EDGE#SPLIT_TAG#" + get_arc_label(s0, s0))
            sub_graph[s0][s0] = True
            sub_graph_arc_list.append((s0, s0))
            return

        # TOP
        elif b0 == root_id and "TOP" not in actions:
            actions.append("TOP")

        # REDUCE
        elif not has_unfound_child(s0) and not lack_head(s0) and is_concept_node(s0) and finish_alignment_node(s0):
            actions.append("REDUCE")
            stack.pop()
            return

        # DROP
        elif finish_alignment_token(b0) and is_surface_token(b0):
            actions.append("DROP")
            buffer.pop()

            while len(deque) != 0:
                stack.append(deque.pop())

            return

        # SHIFT
        elif len(buffer) != 0 and is_concept_node(b0) and find_all_greater_edge(b0):
            while len(deque) != 0:
                stack.append(deque.pop())
            if buffer[-1] not in generated_order:
                num_of_generated_node = len(generated_order)
                generated_order[buffer[-1]] = num_of_generated_node

            stack.append(buffer.pop())
            actions.append("SHIFT")

        # START
        elif start_generate_node(b0) != -1 and is_surface_token(b0):
            node_id = start_generate_node(b0)
            buffer.append(node_id)
            node_begin_dict[b0][node_id] = True
            node_align_begin_flag[node_id] = True

            actions.append("START#SPLIT_TAG#" + get_node_label(node_id))

        # END
        elif s0 in end_generate_node(b0) and s0 != -1 and is_surface_token(b0):
            node_end_dict[b0][s0] = True
            node_align_end_flag[s0] = True

            actions.append("END")

        # PASS
        elif len(stack) != 0:
            deque.append(stack.pop())
            actions.append("PASS")

        # ERROR
        else:
            actions.append('-E-')

    cnt = 0
    while not (len(stack) == 0 and len(buffer) == 0):
        get_oracle_actions_onestep(sub_graph, stack, buffer, actions, root_id)
        remain_unfound_arc = sorted(list(set(arc_indices) - set(sub_graph_arc_list)), key=lambda x: x[0])

        cnt += 1
        if actions[-1] == '-E-' or cnt > 10000:
            print(node_info_dict["graph_id"])
            break

    if not check_graph_finish():
        print(node_info_dict["graph_id"])

    # actions.append('FINISH')

    return actions


def check_cross_arc(file_path):
    # check if cross arc exists

    cross_arc_num = 0
    err_sentence = []
    err_range = []
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            tokens_range = []
            for node_id, node_info in graph.nodes.items():
                tokens_range.append(node_info.anchors[0])

            for i in range(len(tokens_range)):
                for j in range(i + 1, len(tokens_range)):
                    if i == j:
                        continue
                    if (tokens_range[i][1] > tokens_range[j][0] \
                        and tokens_range[i][1] < tokens_range[j][1] \
                        and tokens_range[i][0] < tokens_range[j][0]) or \
                            (tokens_range[j][1] > tokens_range[i][0] \
                             and tokens_range[j][1] < tokens_range[i][1] \
                             and tokens_range[j][0] < tokens_range[i][0]):
                        cross_arc_num += 1
                        err_sentence.append(sentence)
                        tmp = [tokens_range[i][0], tokens_range[i][1], tokens_range[j][0], tokens_range[j][1]]
                        tmp = list(map(lambda x: str(x), tmp))

                        tmp = ','.join(tmp) + '\n' + graph.input[tokens_range[i][0]:tokens_range[i][1]] + \
                              '\n' + graph.input[tokens_range[j][0]:tokens_range[j][1]] + '\n'

                        err_range.append(tmp)

    return cross_arc_num


def check_uncontinuous(file_path):
    # check if un-continuous exists

    cross_arc_num = 0
    err_sentence = []
    err_range = []
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            uncontinuous_num = 0
            total_num = 0
            for node_id, node_info in graph.nodes.items():
                if len(node_info.anchors) > 1:
                    uncontinuous_num += 1
                total_num += 1
            if uncontinuous_num > 0:
                print(uncontinuous_num, total_num)


def check_top_nodes(file_path):
    # check if un-continuous exists
    err_none = 0
    err_multi = 0
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            if graph.top == None:
                err_none += 1
                print(graph.id)
                continue

            if len(graph.top) > 1:
                print(graph.id)
                err_multi += 1
                continue
    print(err_multi, err_none)


def check_carg(file_path, output_file_path):
    # check carg property of node in EDS

    value_label_dict = {}

    triple_list = []
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            for node_id, node_info in graph.nodes.items():
                if node_info.values is not None:
                    if node_info.label not in value_label_dict:
                        value_label_dict[node_info.label] = []

                    info_tuple = '---'.join(
                        [node_info.values, graph.input[node_info.anchors[0][0]:node_info.anchors[0][1]]])
                    if info_tuple not in value_label_dict[node_info.label]:
                        value_label_dict[node_info.label].append(info_tuple)

                    triple_list.append('\t'.join([graph.input[node_info.anchors[0][0]:node_info.anchors[0][1]], \
                                                  node_info.label, \
                                                  node_info.values, \
                                                  ]))
    print(list(value_label_dict.keys()))


def check_longest_sentence(file_path):
    max_len = -1

    cnt = 0
    triple_list = []
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            max_len = max(len(graph.extract_token_info_from_companion_data()["tokens"]), max_len)
            cnt += len(graph.extract_token_info_from_companion_data()["tokens"])

    cnt = cnt / 35656.0
    print(max_len, cnt)
