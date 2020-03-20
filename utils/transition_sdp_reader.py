import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import logging
from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

import json
import copy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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

        if "anchors" in info:
            self.anchors = [(anc["from"], anc["to"]) for anc in info["anchors"]]
            self.anchored = True

        self.properties = info["properties"]
        self.values = info["values"]
        self.label = info["label"]

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

        self.testing = False
        if 'nodes' not in js:
            self.testing = True

        self.gold_mrps = copy.deepcopy(js)
        self.gold_mrps.pop('companion')

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

    def get_arc_info(self):
        tokens, arc_indices, arc_tags = [], [], []

        token_info = self.extract_token_info_from_companion_data()

        ###Step 1: Extract surface token
        tokens = token_info["tokens"]
        lemmas_companion = token_info["lemmas"]
        pos_tags_companion = token_info["pos_tags"]
        tokens_range = token_info["token_range"]

        # extract frame
        frame = ['non'] * len(tokens)
        for node_id, node_info in self.nodes.items():
            try:
                frame[node_id] = node_info.values[1]
            except:
                continue

        # extract pos_tag
        pos_tag = ['non'] * len(tokens)
        for node_id, node_info in self.nodes.items():
            try:
                pos_tag[node_id] = node_info.values[0]
            except:
                continue

        # extract node_label
        node_label = ['non'] * len(tokens)
        for node_id, node_info in self.nodes.items():
            try:
                node_label[node_id] = node_info.label
            except:
                continue

        # MRP-Testing mode: tokenization from Companion data
        if self.testing == True:
            ret = {"tokens": tokens,
                   "tokens_range": tokens_range,
                   "arc_indices": arc_indices,
                   "arc_tags": arc_tags,
                   "lemmas": lemmas_companion,
                   "mrp_pos_tags": pos_tags_companion,
                   "meta_info": self.meta_info,
                   "frame": frame,
                   "pos_tag": pos_tag,
                   "node_label": node_label,
                   "gold_mrps": self.gold_mrps}
            return ret

        ###Step 2: Add arc label
        # link root and top token(s), if top(s) exists in data_item
        if self.top is not None:
            for top_node in self.top:
                arc_indices.append((top_node + 1, 0))  # 0 represents root
                arc_tags.append('ROOT')

        # link other tokens
        for node_id, node_info in self.nodes.items():
            for child_of_node_info in node_info.childs:
                arc_indices.append((child_of_node_info.node + 1, node_id + 1))
                arc_tags.append(child_of_node_info.rel)

        ###Step 3: extract lemma feature and pos_tag feature
        ### Due to the unperfect tokenization of MRP-Companion data,
        ### we need to align the companion data and original data

        # key:gold-token/layer0-node
        # value:companion-token
        align_dict = {}
        node_info_flag = [False] * len(tokens)

        mrp_lemmas = []
        mrp_pos_tags = []

        if len(tokens) != len(mrp_pos_tags):
            mrp_pos_tags = pos_tags_companion

        if len(tokens) != len(mrp_lemmas):
            mrp_lemmas = lemmas_companion

        ret = {"tokens": tokens,
               "tokens_range": tokens_range,
               "arc_indices": arc_indices,
               "arc_tags": arc_tags,
               "lemmas": mrp_lemmas,
               "mrp_pos_tags": mrp_pos_tags,
               "meta_info": self.meta_info,
               "frame": frame,
               "pos_tag": pos_tag,
               "node_label": node_label,
               "gold_mrps": self.gold_mrps}

        return ret


def parse_sentence(sentence_blob: str):
    graph = Graph(json.loads(sentence_blob))
    ret = graph.get_arc_info()
    return ret


def lazy_parse(text: str):
    for sentence in text.split("\n"):
        if sentence:
            yield parse_sentence(sentence)


@DatasetReader.register("sdp_reader_conll2019")
class SDPDatasetReaderConll2019(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 action_indexers: Dict[str, TokenIndexer] = None,
                 arc_tag_indexers: Dict[str, TokenIndexer] = None,
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

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r', encoding='utf8') as ucca_file:
            logger.info("Reading SDP instances from conllu dataset at: %s", file_path)
            for ret in lazy_parse(ucca_file.read()):

                tokens = ret["tokens"]
                arc_indices = ret["arc_indices"]
                arc_tags = ret["arc_tags"]
                lemmas = ret["lemmas"]
                mrp_pos_tags = ret["mrp_pos_tags"]

                meta_info = ret["meta_info"]
                tokens_range = ret["tokens_range"]
                frame = ret["frame"]
                pos_tag = ret["pos_tag"]
                gold_mrps = ret["gold_mrps"]
                node_label = ret["node_label"]

                #In CoNLL2019, gold actions is not avaiable in test set.
                gold_actions = get_oracle_actions(tokens, arc_indices, arc_tags) if arc_indices else None

                if gold_actions and gold_actions[-1] == '-E-':
                    print('-E-')
                    continue

                yield self.text_to_instance(tokens, lemmas, mrp_pos_tags, arc_indices, arc_tags, gold_actions,
                                            [meta_info],
                                            tokens_range, frame, pos_tag, node_label, [gold_mrps])

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         lemmas: List[str] = None,
                         mrp_pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         gold_actions: List[str] = None,
                         meta_info: List[str] = None,
                         tokens_range: List[Tuple[int, int]] = None,
                         frame: List[str] = None,
                         pos_tag: List[str] = None,
                         node_label: List[str] = None,
                         gold_mrps: List[str] = None) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        meta_dict = {"tokens": tokens}

        if lemmas is not None and self._lemma_indexers is not None:
            fields["lemmas"] = TextField([Token(l) for l in lemmas], self._lemma_indexers)

        if mrp_pos_tags is not None:
            fields["mrp_pos_tags"] = SequenceLabelField(mrp_pos_tags, token_field, label_namespace="pos")

        if frame is not None:
            fields["frame"] = SequenceLabelField(frame, token_field, label_namespace="frame")

        if pos_tag is not None:
            fields["pos_tag"] = SequenceLabelField(pos_tag, token_field, label_namespace="pos_tag")

        if node_label is not None:
            fields["node_label"] = SequenceLabelField(node_label, token_field, label_namespace="node_label")

        if arc_indices is not None and arc_tags is not None:
            meta_dict["arc_indices"] = arc_indices
            meta_dict["arc_tags"] = arc_tags
            fields["arc_tags"] = TextField([Token(a) for a in arc_tags], self._arc_tag_indexers)

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        if meta_info is not None:
            meta_dict["meta_info"] = meta_info[0]

        if tokens_range is not None:
            meta_dict["tokens_range"] = tokens_range

        if gold_mrps is not None:
            meta_dict["gold_mrps"] = gold_mrps[0]

        fields["metadata"] = MetadataField(meta_dict)
        return Instance(fields)


def get_oracle_actions(annotated_sentence, directed_arc_indices, arc_tags):
    graph = {}
    for token_idx in range(len(annotated_sentence) + 1):
        graph[token_idx] = []

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    for arc, arc_tag in zip(directed_arc_indices, arc_tags):
        graph[arc[0]].append((arc[1], arc_tag))

    N = len(graph)  # N-1 point, 1 root point

    # i:head_point j:child_point
    top_down_graph = [[] for i in range(N)]  # N-1 real point, 1 root point => N point

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = [[False for i in range(N)] for j in range(N)]

    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    actions = []
    stack = [0]
    buffer = []
    deque = []

    for i in range(N - 1, 0, -1):
        buffer.append(i)

    # return if w1 is one head of w0
    def has_head(w0, w1):
        if w0 <= 0:
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

    # return if w has other head except the present one
    def has_other_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    # return if w has any unfound head
    def lack_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return if w has any unfound child in stack sigma
    # !!! except the top in stack
    def has_other_child_in_stack(stack, w):
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack \
                    and c != stack[-1] \
                    and not sub_graph[c][w]:
                return True
        return False

    # return if w has any unfound head in stack sigma
    # !!! except the top in stack
    def has_other_head_in_stack(stack, w):
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack \
                    and h[0] != stack[-1] \
                    and not sub_graph[w][h[0]]:
                return True
        return False

    # return the relation between child: w0, head: w1
    def get_arc_label(w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions):
        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        if s0 > 0 and has_head(s0, b0):
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        elif s0 >= 0 and has_head(b0, s0):
            if not has_other_child_in_stack(stack, b0) and not has_other_head_in_stack(stack, b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        elif len(buffer) != 0 and not has_other_head_in_stack(stack, b0) \
                and not has_other_child_in_stack(stack, b0):
            actions.append("NS")
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    while len(buffer) != 0:
        get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions)

    return actions


def check_pos_frame(file_path):
    pos = []
    frame = []
    with open(file_path, 'r', encoding='utf8') as eds_file:
        for sentence in eds_file.read().split("\n"):
            try:
                graph = Graph(json.loads(sentence))
                for node_id, node_info in graph.nodes.items():
                    if 'pos' not in node_info.properties:
                        pos.append(sentence)
                    if 'frame' not in node_info.properties:
                        frame.append(sentence)

                    if 'pos' not in node_info.properties or 'frame' not in node_info.properties:
                        break
            except:
                print(sentence)

    print(len(pos), len(frame))
