import json
import logging
from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def parse_sentence(sentence_blob: str) -> Tuple[List[str], List[List[str]], str, str, str, str, str]:
    tokens = []
    id = ''
    actions = []
    amr = []
    input = ''
    mrp = ''
    companion = ''

    for line in sentence_blob.split('\n'):
        if line.startswith('#'):
            line = line.split(' ', maxsplit=2)
            if len(line) != 3:
                continue
            if line[1] == '::id':
                id = line[-1].split(maxsplit=1)[0]
            elif line[1] == '::tok':
                tokens = line[-1].split()
            elif line[1] == '::action':
                actions.append(line[-1].split())
            elif line[1] == '::snt':
                input = line[-1]
            elif line[1] == '::mrp':
                mrp = line[-1]
            elif line[1] == '::companion':
                companion = line[-1]
        else:
            amr.append(line)
    amr = ' '.join(map(str.strip, amr))

    return tokens, actions, id, amr, input, mrp, companion


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            result = parse_sentence(sentence)
            if result[3]:
                yield result


@DatasetReader.register("amr_list-based_arc-eager")
class AMRDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super(AMRDatasetReader, self).__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._lemma_indexers = None
        if lemma_indexers is not None and len(lemma_indexers) > 0:
            self._lemma_indexers = lemma_indexers
        self._action_indexers = None

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading AMR data from: %s", file_path)

        with open(file_path, encoding='utf8') as sdp_file:
            for tokens, actions, id, amr, input, mrp, companion in lazy_parse(sdp_file.read()):
                lemmas = None
                pos_tags = None
                yield self.text_to_instance(tokens, lemmas, pos_tags, actions, id, amr, input, mrp, companion)

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         lemmas: List[str] = None,
                         pos_tags: List[str] = None,
                         gold_actions: List[List[str]] = None,
                         id: str = None,
                         amr: str = None,
                         input: str = None,
                         mrp: str = None,
                         companion: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        meta_dict = {"tokens": tokens}
        if id:
            meta_dict["id"] = id
        if amr:
            meta_dict["amr"] = amr
        if input:
            meta_dict["input"] = input
        if mrp:
            meta_dict["mrp"] = json.loads(mrp)
        if companion:
            meta_dict["companion"] = json.loads(companion)

        if lemmas is not None and self._lemma_indexers is not None:
            fields["lemmas"] = TextField([Token(l) for l in lemmas], self._lemma_indexers)
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")

        if gold_actions is not None:
            meta_dict["gold_actions"] = ['@@:@@'.join(a) for a in gold_actions]
            fields["gold_actions"] = TextField([Token('@@:@@'.join(a)) for a in gold_actions],
                                               {'actions': SingleIdTokenIndexer(namespace='actions')})
            fields["gold_newnodes"] = TextField(
                [Token(a[1] if a[0] == 'NEWNODE' else DEFAULT_PADDING_TOKEN) for a in gold_actions],
                {'newnodes': SingleIdTokenIndexer(namespace='newnodes')})
            fields["gold_entities"] = TextField(
                [Token(a[1] if a[0] == 'ENTITY' else DEFAULT_PADDING_TOKEN) for a in gold_actions],
                {'entities': SingleIdTokenIndexer(namespace='entities')})
            fields["gold_relations"] = TextField(
                [Token(a[1] if a[0] in ['LEFT', 'RIGHT'] else DEFAULT_PADDING_TOKEN) for a in gold_actions],
                {'relations': SingleIdTokenIndexer(namespace='relations')})
        fields["metadata"] = MetadataField(meta_dict)

        return Instance(fields)
