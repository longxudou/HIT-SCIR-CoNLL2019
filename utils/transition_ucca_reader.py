import copy
import json
import logging
from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from conllu.parser import parse_line, DEFAULT_FIELDS
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# for aggregate multi-label arc
label_prior = ['P', 'S', 'C', 'H', 'A', 'E', 'R', 'T', 'Q', 'D', 'F', 'U', 'G', 'L']
label_prior_dict = {label_prior[idx]: idx for idx in range(len(label_prior))}
for idx in range(len(label_prior)):
    label_prior_dict[label_prior[idx] + '*'] = idx + len(label_prior)


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
        # assert self.anchored ==False
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
        assert len(js["tops"]) == 1
        self.top = js["tops"][0]

        self.companion = {}
        if 'companion' in js:
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
        if 'nodes' not in js or len(js['nodes']) == 0:
            self.testing = True

        self.lay_0_node = []
        self.lay_1_node = []

        self.gold_mrps = copy.deepcopy(js)
        if 'companion' in self.gold_mrps:
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
        concept_node_expect_root = []

        lay_0_node_info = []
        childs_dict = {node_id: {} for node_id in self.nodes.keys()}

        ###Step 1: Extract surface token node
        # MRP-Testing mode: tokenization from Companion data
        if self.testing == True and len(self.companion) != 0:
            token_info = self.extract_token_info_from_companion_data()
            tokens = token_info["tokens"]
            token_range = token_info["token_range"]
            self.lay_0_node = token_info["tokens"]

            ret = {"tokens": tokens,
                   "tokens_range": token_range,
                   "layer_0_node": self.lay_0_node,
                   "meta_info": self.meta_info}

            return ret

        # MRP-Training mode: tokenization from mrp data
        else:
            for node_id, node_info in self.nodes.items():
                if node_info.anchored == True:
                    for anchor_idx in range(len(node_info.anchors)):
                        token_begin_idx = node_info.anchors[anchor_idx][0]
                        token_end_idx = node_info.anchors[anchor_idx][1]
                        lay_0_node_info.append((token_begin_idx, token_end_idx))
                        self.lay_0_node.append(self.input[token_begin_idx:token_end_idx])

                        arc_indices.append(("layer_0:" + str(len(self.lay_0_node) - 1), node_id))
                        arc_tags.append("Terminal")

                if node_id != self.top:
                    concept_node_expect_root.append(node_id)

                self.lay_1_node.append(node_id)

        tokens = self.lay_0_node

        ###Step 2: Add arc label
        for node_id, node_info in self.nodes.items():
            for _child_of_node_info in node_info.childs:
                _child_node = _child_of_node_info.node
                _rel = _child_of_node_info.rel
                _remote = _child_of_node_info.remote

                _arc_tag = _rel if _remote == False else _rel + '*'

                # the arc with one label
                if _child_node not in childs_dict[node_id]:
                    childs_dict[node_id][_child_node] = _arc_tag
                # the arc with multi label
                # aggregate the multi-label by label prior, defined in the start in this file
                else:
                    # expand the label_prior_dict. this only happens when occur n-label arc (n>2)
                    if childs_dict[node_id][_child_node] not in label_prior_dict:
                        label_prior_dict[childs_dict[node_id][_child_node]] = len(label_prior_dict)

                    if label_prior_dict[_arc_tag] < label_prior_dict[childs_dict[node_id][_child_node]]:
                        _arc_tag = _arc_tag + '+' + childs_dict[node_id][_child_node]
                    else:
                        _arc_tag = childs_dict[node_id][_child_node] + '+' + _arc_tag

                    childs_dict[node_id][_child_node] = _arc_tag

        for node_id, node_info in self.nodes.items():
            for _child_node in childs_dict[node_id]:
                _arc_tag = childs_dict[node_id][_child_node]

                arc_indices.append((_child_node, node_id))
                arc_tags.append(_arc_tag)

        ###Step 3: trans arc_indices and concept_node_expect_root, add node's index with len(tokens)
        # add layer1_node_idx in arc_indices with len(layer0_node)
        trans_arc_indices = arc_indices[:]
        arc_indices = []
        for arc_info in trans_arc_indices:
            if isinstance(arc_info[0], int):
                arc_indices.append((arc_info[0] + len(tokens), arc_info[1] + len(tokens)))
            else:
                arc_indices.append((int(arc_info[0][8:]), arc_info[1] + len(tokens)))

        # add layer1_node_idx in concept_node_expect_root with len(layer0_node)
        trans_concept_node_expect_root = concept_node_expect_root[:]
        concept_node_expect_root = []
        for node_id in trans_concept_node_expect_root:
            concept_node_expect_root.append(node_id + len(tokens))

        ###Step 4: extract lemma feature and pos_tag feature
        ### Due to the unperfect tokenization of MRP-Companion data,
        ### we need to align the companion data and original data

        # key:gold-token/layer0-node
        # value:companion-token
        align_dict = {}
        node_info_flag = [False] * len(lay_0_node_info)

        mrp_lemmas = []
        mrp_pos_tags = []

        if len(self.companion) != 0:
            token_info = self.extract_token_info_from_companion_data()

            lemmas = token_info["lemmas"]
            pos_tags = token_info["pos_tags"]
            token_range = token_info["token_range"]

            for companion_token_idx in range(len(token_range)):
                companion_token_info = token_range[companion_token_idx]

                for node_idx in range(len(lay_0_node_info)):
                    node_info = lay_0_node_info[node_idx]

                    if companion_token_info[0] <= node_info[0] and node_info[1] <= companion_token_info[1] \
                            and node_info_flag[node_idx] == False:
                        align_dict[node_idx] = companion_token_idx
                        node_info_flag[node_idx] = True

                        mrp_lemmas.append(lemmas[companion_token_idx])
                        mrp_pos_tags.append(pos_tags[companion_token_idx])

        if len(tokens) != len(mrp_pos_tags):
            mrp_pos_tags = tokens

        if len(tokens) != len(mrp_lemmas):
            mrp_lemmas = tokens

        ret = {"tokens": tokens,
               "tokens_range": lay_0_node_info,
               "arc_indices": arc_indices,
               "arc_tags": arc_tags,
               "concept_node_expect_root": concept_node_expect_root,
               "root_id": self.top + len(tokens),
               "layer_0_node": self.lay_0_node,
               "layer_1_node": self.lay_1_node,
               "lemmas": mrp_lemmas,
               "pos_tags": mrp_pos_tags,
               "meta_info": self.meta_info,
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


def expand_arc_with_descendants(arc_indices, total_node_num, len_tokens):
    ###step 1: construct graph
    graph = {}
    for token_idx in range(total_node_num):
        graph[token_idx] = {"in_degree": 0, "head_list": []}

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    for arc in arc_indices:
        graph[arc[0]]["head_list"].append((arc[1], 'Arc_label_place_holder'))
        graph[arc[1]]["in_degree"] += 1

    # i:head_point j:child_point›
    top_down_graph = [[] for i in range(total_node_num)]  # N real point, 1 root point, concept_node_expect_root
    step2_top_down_graph = [[] for i in range(total_node_num)]

    topological_stack = []
    for i in range(total_node_num):
        if graph[i]["in_degree"] == 0:
            topological_stack.append(i)
        for head_tuple_of_point_i in graph[i]["head_list"]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)
            step2_top_down_graph[head].append(i)

    ###step 2: construct topological order
    topological_order = []
    # step2_top_down_graph=top_down_graph[:]
    while len(topological_stack) != 0:
        stack_0_node = topological_stack.pop()
        topological_order.append(stack_0_node)

        for i in graph:
            if stack_0_node in step2_top_down_graph[i]:
                step2_top_down_graph[i].remove(stack_0_node)
                graph[i]["in_degree"] -= 1
                if graph[i]["in_degree"] == 0 and \
                        i not in topological_stack and \
                        i not in topological_order:
                    topological_stack.append(i)

    ###step 3: expand arc indices using the nodes indices ordered by topological way
    expand_node_dict = {}
    for node_idx in range(total_node_num):
        expand_node_dict[node_idx] = top_down_graph[node_idx][:]

    for node_idx in topological_order:
        if len(expand_node_dict[node_idx]) == 0:  # no childs
            continue
        expand_childs = expand_node_dict[node_idx][:]
        for child in expand_node_dict[node_idx]:
            expand_childs += expand_node_dict[child]
        expand_node_dict[node_idx] = expand_childs

    ###step 4: delete duplicate and concept node
    token_filter = set(list(i for i in range(len_tokens)))
    for node_idx in expand_node_dict:
        expand_node_dict[node_idx] = list(set(expand_node_dict[node_idx]) & token_filter)

    ###step 5: expand arc indices using expand_node_dict
    arc_descendants = []
    for arc_info in arc_indices:
        arc_info_0 = arc_info[0] if arc_info[0] < len_tokens else \
            '-'.join([str(i) for i in sorted(expand_node_dict[arc_info[0]])])

        arc_info_1 = arc_info[1] if arc_info[1] < len_tokens else \
            '-'.join([str(i) for i in sorted(expand_node_dict[arc_info[1]])])

        arc_descendants.append((arc_info_0, arc_info_1))

    return arc_descendants


@DatasetReader.register("ucca_reader_conll2019")
class UCCADatasetReaderConll2019(DatasetReader):
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
            logger.info("Reading UCCA instances from conllu dataset at: %s", file_path)
            for ret in lazy_parse(ucca_file.read()):
                tokens = ret["tokens"] if "tokens" in ret else None
                arc_indices = ret["arc_indices"] if "arc_indices" in ret else None
                arc_tags = ret["arc_tags"] if "arc_tags" in ret else None
                root_id = ret["root_id"] if "root_id" in ret else None
                lemmas = ret["lemmas"] if "lemmas" in ret else None
                pos_tags = ret["pos_tags"] if "pos_tags" in ret else None
                meta_info = ret["meta_info"] if "meta_info" in ret else None
                tokens_range = ret["tokens_range"] if "tokens_range" in ret else None
                gold_mrps = ret["gold_mrps"] if "gold_mrps" in ret else None

                concept_node_expect_root = ret["concept_node_expect_root"] if "concept_node_expect_root" in ret else None

                # In CoNLL2019, gold actions is not avaiable in test set.
                gold_actions = get_oracle_actions(tokens, arc_indices, arc_tags, root_id, \
                                                  concept_node_expect_root,
                                                  len(ret["layer_0_node"]) + len(ret["layer_1_node"])) if "layer_1_node" in ret else None

                if gold_actions and tokens and len(gold_actions) / len(tokens) > 20:
                    print(len(gold_actions) / len(tokens))

                arc_descendants = expand_arc_with_descendants(arc_indices,
                                                              len(ret["layer_0_node"]) + len(ret["layer_1_node"]),
                                                              len(tokens)) if "layer_1_node" in ret else None

                if gold_actions and gold_actions[-2] == '-E-':
                    print('-E-')
                    continue
                yield self.text_to_instance(tokens, lemmas, pos_tags, arc_indices, arc_tags, gold_actions,
                                            arc_descendants, [root_id], [meta_info], tokens_range, [gold_mrps])

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         lemmas: List[str] = None,
                         pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         gold_actions: List[str] = None,
                         arc_descendants: List[str] = None,
                         root_id: List[int] = None,
                         meta_info: List[str] = None,
                         tokens_range: List[Tuple[int, int]] = None,
                         gold_mrps: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)

        fields["tokens"] = token_field
        meta_dict = {"tokens": tokens}

        if arc_indices is not None and arc_tags is not None:
            meta_dict["arc_indices"] = arc_indices
            meta_dict["arc_tags"] = arc_tags
            fields["arc_tags"] = TextField([Token(a) for a in arc_tags], self._arc_tag_indexers)

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        if arc_descendants is not None:
            meta_dict["arc_descendants"] = arc_descendants

        if root_id is not None:
            meta_dict["root_id"] = root_id[0]

        if meta_info is not None:
            meta_dict["meta_info"] = meta_info[0]

        if tokens_range is not None:
            meta_dict["tokens_range"] = tokens_range

        if gold_mrps is not None:
            meta_dict["gold_mrps"] = gold_mrps[0]

        fields["metadata"] = MetadataField(meta_dict)

        return Instance(fields)


def get_oracle_actions(tokens, arc_indices, arc_tags, root_id, concept_node_expect_root, total_node_num):
    actions = []
    stack = [root_id]
    buffer = []
    concept_node_expect_root = {i: False for i in concept_node_expect_root}
    generated_order = {root_id: 0}

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
    # i:head_point j:child_point›
    top_down_graph = [[] for i in range(total_node_num)]  # N real point, 1 root point, concept_node_expect_root

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = [[False for i in range(total_node_num)] for j in range(total_node_num)]
    sub_graph_arc_list = []

    for i in range(total_node_num):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    def has_find_primary_head(w0):
        if w0 < 0:
            return False
        for node_info in graph[w0]:
            if '*' not in node_info[1] and sub_graph[w0][node_info[0]] == True:
                return True
        return False

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

    # head:w1, child:w0
    def has_remote_edge(w0, w1):
        if w0 < 0 or w1 < 0:
            return False

        for node_info in graph[w0]:
            if node_info[0] == w1:
                return '*' in node_info[1]
        return False

    def get_conpect_node_id(w0):
        """
        return True only if find the new head+concept node of w0
        """
        if w0 < 0:
            return -1
        for head_node_info_of_w0 in graph[w0]:
            head_node_id = head_node_info_of_w0[0]
            if sub_graph[w0][head_node_id] == False and head_node_id in concept_node_expect_root:
                if concept_node_expect_root[head_node_id] == True:
                    return -1
                return head_node_id
        return -1

    def check_graph_finish():
        return whole_graph == sub_graph

    def check_sub_graph(w0, w1):
        if w0 < 0 or w1 < 0:
            return False
        else:
            return sub_graph[w0][w1] == False

    def get_oracle_actions_onestep(sub_graph, stack, buffer, actions):

        s0 = stack[-1] if len(stack) > 0 else -1
        s1 = stack[-2] if len(stack) > 1 else -1

        # RIGHT_EDGE/REMOTE-EDGE
        if s0 != -1 and has_head(s0, s1) and check_sub_graph(s0, s1):
            if has_remote_edge(s0, s1):
                actions.append("RIGHT-REMOTE:" + get_arc_label(s0, s1))
            else:
                actions.append("RIGHT-EDGE:" + get_arc_label(s0, s1))
            sub_graph[s0][s1] = True
            sub_graph_arc_list.append((s0, s1))
            return

        # LEFT_EDGE/REMOTE-EDGE
        elif s1 != root_id and has_head(s1, s0) and check_sub_graph(s1, s0):
            if has_remote_edge(s1, s0):
                actions.append("LEFT-REMOTE:" + get_arc_label(s1, s0))
            else:
                actions.append("LEFT-EDGE:" + get_arc_label(s1, s0))
            sub_graph[s1][s0] = True
            sub_graph_arc_list.append((s1, s0))
            return

        # NODE
        elif s0 != root_id and get_conpect_node_id(s0) != -1 and has_head(s0, get_conpect_node_id(
                s0)) and not has_find_primary_head(s0):
            concept_node_id = get_conpect_node_id(s0)
            buffer.append(concept_node_id)

            actions.append("NODE:" + get_arc_label(s0, concept_node_id))

            concept_node_expect_root[concept_node_id] = True
            sub_graph[s0][concept_node_id] = True
            sub_graph_arc_list.append((s0, concept_node_id))

            return

        # REDUCE
        elif s0 != -1 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("REDUCE")
            stack.pop()
            return

        # SWAP
        elif len(stack) > 2 and generated_order[s0] > generated_order[s1] and (
                has_unfound_child(stack[-3]) or lack_head(stack[-3])):
            buffer.append(stack.pop(-2))
            actions.append("SWAP")
            return

        # SHIFT
        elif len(buffer) != 0:

            if buffer[-1] not in generated_order:
                num_of_generated_node = len(generated_order)
                generated_order[buffer[-1]] = num_of_generated_node

            stack.append(buffer.pop())
            actions.append("SHIFT")
            return

        # REMOTE-NODE
        elif s0 != root_id and get_conpect_node_id(s0) != -1 and has_remote_edge(s0, get_conpect_node_id(s0)):
            concept_node_id = get_conpect_node_id(s0)
            buffer.append(concept_node_id)
            actions.append("REMOTE-NODE:" + get_arc_label(s0, concept_node_id))
            concept_node_expect_root[concept_node_id] = True
            sub_graph[s0][concept_node_id] = True
            sub_graph_arc_list.append((s0, concept_node_id))

            return

        else:
            remain_unfound_edge = set(arc_indices) - set(sub_graph_arc_list)
            actions.append('-E-')
            return

    while not (check_graph_finish() and len(buffer) == 0):
        get_oracle_actions_onestep(sub_graph, stack, buffer, actions)
        if actions[-1] == '-E-':
            break

    actions.append('FINISH')
    stack.pop()

    return actions


def count_multi_label_arc(file_path):
    total_arc_list = []
    multi_label_arc_list = {}

    total_sentence = 0
    has_multi_label_arc = 0

    flag = False
    with open(file_path, 'r', encoding='utf8') as ucca_file:
        for tokens, arc_indices, arc_tags, root_id, concept_node_expect_root in lazy_parse(ucca_file.read()):
            flag = False
            for arc_info in arc_tags:
                total_arc_list.append(arc_info)
                if '+' in arc_info:
                    if arc_info not in multi_label_arc_list:
                        multi_label_arc_list[arc_info] = 1
                    else:
                        multi_label_arc_list[arc_info] += 1
                    flag = True
            if flag:
                has_multi_label_arc += 1
            total_sentence += 1

    print(total_sentence, has_multi_label_arc, has_multi_label_arc / total_sentence)


def count_continuous_anchors(file_path):
    # count continuous spans
    output_list = []

    with open(file_path, 'r', encoding='utf8') as ucca_file:
        for sentence in ucca_file.read().split("\n"):
            graph = Graph(json.loads(sentence))
            uncontinuous_num = 0
            total_num = 0

            for node_id, node_info in graph.nodes.items():
                output_tmp = []
                if len(node_info.anchors) > 1:
                    uncontinuous_num += 1
                    for anchor_info in node_info.anchors:
                        output_tmp.append(graph.input[anchor_info[0]:anchor_info[1]])

                if len(node_info.anchors) > 0:
                    total_num += 1

                if len(output_tmp) > 0:
                    output_list.append(' '.join(output_tmp))

            if uncontinuous_num > 0 and total_num > 50:
                print(uncontinuous_num, total_num)
