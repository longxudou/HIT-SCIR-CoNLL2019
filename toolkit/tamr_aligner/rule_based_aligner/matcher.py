#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import absolute_import
import string
import os
from .stemmer import Stemmer, lemmatizer
from system.misc import parse_all_dates
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

_word2vec = None


def init_word2vec(filename, binary=False):
    global _word2vec
    _word2vec = KeyedVectors.load_word2vec_format(filename, binary=binary)


def get_concept_name(name):
    if '-' in name and name.rsplit('-', 1)[1].isdigit():
        return name.rsplit('-', 1)[0]
    return name


class CommonPrefixMatchHelper(object):
    def __init__(self, minimal_match_length=4):
        self.minimal_match_length = minimal_match_length

    def is_fuzzy_match_word(self, stemmed_word, concept):
        longest_common_prefix = CommonPrefixMatchHelper._max_prefix(stemmed_word, concept)
        if longest_common_prefix >= self.minimal_match_length:
            return True
        return False

    def is_fuzzy_match_wordlist(self, stemmed_wordlist, concept):
        for stemmed_word in stemmed_wordlist:
            longest_common_prefix = CommonPrefixMatchHelper._max_prefix(stemmed_word, concept)
            if longest_common_prefix >= self.minimal_match_length:
                return True
        return False

    def match_lengths(self, stemmed_words, graph):
        max_match_lengths = [self.minimal_match_length for _ in stemmed_words]
        for node in graph.true_nodes():
            concept_name = get_concept_name(node.name)
            for i, stemmed_candidates in enumerate(stemmed_words):
                payload = max([self._max_prefix(stemmed_word, concept_name) for stemmed_word in stemmed_candidates])
                if payload > max_match_lengths[i]:
                    max_match_lengths[i] = payload
        return max_match_lengths

    @staticmethod
    def _max_prefix(str1, str2):
        len1, len2 = len(str1), len(str2)
        ret = 0
        for i in range(min(len1, len2)):
            if str1[:i + 1] == str2[:i + 1]:
                ret += 1
            else:
                break
        return ret


class SemanticMatchHelper(object):
    def __init__(self, similarity_threshold=0.7, lower=True, exclude_stop_words=True, exclude_punctuations=True):
        self.similarity_threshold = similarity_threshold
        self.lower = lower
        self.exclude_stop_words = exclude_stop_words
        self.exclude_punctuations = exclude_punctuations
        self.stopwords = set(stopwords.words('english'))
        assert _word2vec is not None

    @staticmethod
    def _all_punctuation(word):
        return len([c for c in word if c not in string.punctuation]) == 0

    def is_semantic_match_word(self, word, concept):
        if self.lower:
            word = word.lower()
            concept = concept.lower()
        if self.exclude_stop_words and word in self.stopwords:
            return False
        if self.exclude_punctuations and self._all_punctuation(word):
            return False
        try:
            if _word2vec.similarity(word, concept) > self.similarity_threshold:
                return True
            else:
                return False
        except:
            return False

    def is_semantic_match_wordlist(self, wordlist, concept):
        if self.lower:
            concept = concept.lower()
        for word in wordlist:
            if self.exclude_stop_words and word in self.stopwords:
                continue
            if self.exclude_punctuations and self._all_punctuation(word):
                continue
            try:
                if _word2vec.similarity(word, concept) > self.similarity_threshold:
                    return True
            except:
                continue
        return False


class Matcher(object):
    def __init__(self):
        pass

    def match(self, words, stemmed_words, postags, graph, align_results):
        raise NotImplemented


class WordMatcher(Matcher):
    def __init__(self):
        super(WordMatcher, self).__init__()

    def match(self, words, stemmed_words, postags, graph, align_results):
        for start, stemmed_wordlist in enumerate(stemmed_words):
            for stemmed_word in stemmed_wordlist:
                for node in graph.true_nodes():
                    if graph.is_entity_token(node, consider_alignment=False) or \
                            graph.is_date_entity_attributes(node) or \
                            graph.is_url_entity_attributes(node) or \
                            graph.is_ordinal_entity_attributes(node):
                        continue
                    concept_name = get_concept_name(node.name)
                    if stemmed_word == concept_name:
                        align_results.add(start, start + 1, node.level, None)


class FuzzyWordMatcher(WordMatcher):
    kFunctionalConcept = ('have-org-role-91',
                          'have-rel-role-91',
                          'multi-sentence',
                          'rate-entity-91',
                          '-',
                          'byline-91',
                          'monetary-quantity',
                          'temporal-quantity',
                          'amr-unknown',
                          'date-entity',
                          'date-interval',
                          'include-91',
                          'be-located-at-91')

    def __init__(self):
        super(FuzzyWordMatcher, self).__init__()
        self.helper = CommonPrefixMatchHelper()

    def skip(self, node, graph):
        return graph.is_entity_token(node, consider_alignment=False) or \
               graph.is_date_entity_attributes(node) or \
               graph.is_url_entity_attributes(node) or \
               graph.is_ordinal_entity_attributes(node) or \
               node.name in self.kFunctionalConcept

    def match(self, words, stemmed_words, postags, graph, align_results):
        for start, stemmed_wordlist in enumerate(stemmed_words):
            for node in graph.true_nodes():
                if self.skip(node, graph):
                    continue
                concept_name = get_concept_name(node.name)
                if self.helper.is_fuzzy_match_wordlist(stemmed_wordlist, concept_name):
                    align_results.add(start, start + 1, node.level, None)


class MorphosemanticLinkMatcher(FuzzyWordMatcher):
    def __init__(self):
        super(MorphosemanticLinkMatcher, self).__init__()
        path = os.path.join(os.path.dirname(__file__), 'morphosemantic-links.dic')
        lexicon = {}
        for line in open(path, 'r'):
            verb, noun = line.strip().split(',')
            if verb not in lexicon:
                lexicon[verb] = set()
            lexicon[verb].add(noun)
        self.lexicon = lexicon

    def match(self, words, stemmed_words, postags, graph, align_results):
        for start, (word, postag) in enumerate(zip(words, postags)):
            for node in graph.true_nodes():
                if self.skip(node, graph):
                    continue
                concept = get_concept_name(node.name)
                if concept not in self.lexicon:
                    continue

                candidates = set()
                if postag is not None:
                    candidates.add(lemmatizer.lemmatize(word.lower(), postag))
                else:
                    candidates.add(lemmatizer.lemmatize(word.lower(), 'n'))
                    candidates.add(lemmatizer.lemmatize(word.lower(), 'v'))
                    candidates.add(lemmatizer.lemmatize(word.lower(), 'a'))
                    candidates.add(lemmatizer.lemmatize(word.lower(), 's'))
                for candidate in candidates:
                    if candidate in self.lexicon[concept]:
                        align_results.add(start, start + 1, node.level, None)


class SemanticWordMatcher(FuzzyWordMatcher):
    def __init__(self, threshold=0.7, lower=True):
        super(SemanticWordMatcher, self).__init__()
        self.helper = SemanticMatchHelper(threshold, lower)

    def match(self, words, stemmed_words, postags, graph, align_results):
        for start, wordlist in enumerate(stemmed_words):
            for node in graph.true_nodes():
                if self.skip(node, graph):
                    continue
                concept = get_concept_name(node.name)
                if self.helper.is_semantic_match_wordlist(wordlist, concept):
                    align_results.add(start, start + 1, node.level, None)


class FuzzySpanMatcher(Matcher):
    def __init__(self):
        super(FuzzySpanMatcher, self).__init__()

    @staticmethod
    def normalize_million_and_billion(surface):
        tokens = surface.replace('-', ' ').split()
        if len(tokens) != 2:
            return None
        if tokens[1] not in ('billion', 'million'):
            return None
        power = (10 ** 9 if tokens[1] == 'billion' else 10 ** 6)
        try:
            number = float(tokens[0])
            return str(int(number * power))
        except:
            if tokens[0].lower() in Stemmer.kNumbers:
                return str((Stemmer.kNumbers.index(tokens[0].lower()) + 1) * power)
            return None

    def match(self, words, stemmed_words, postags, graph, align_results):
        n_words = len(words)
        for start in range(n_words):
            for end in range(start + 2, n_words + 1):
                surface = ' '.join(words[start: end]).lower()
                norm_number = FuzzySpanMatcher.normalize_million_and_billion(surface)
                for node in graph.true_nodes():
                    concept_name = get_concept_name(node.name).replace('-', ' ')
                    if concept_name == surface:
                        align_results.add(start, end, node.level, None)
                    elif concept_name == norm_number:
                        align_results.add(start, end, node.level, None)


class EntityMatcher(Matcher):
    def __init__(self):
        super(EntityMatcher, self).__init__()

    def add(self, start, end, root_level, levels, align_results):
        align_results.add(start, end, root_level, None)
        for level in levels:
            align_results.add(start, end, level, root_level)


class NamedEntityMatcher(EntityMatcher):
    def __init__(self):
        super(NamedEntityMatcher, self).__init__()

    @classmethod
    def collect_entities(cls, graph):
        entities = {}
        for node in graph.true_nodes():
            if graph.is_entity_name(node, consider_alignment=False):
                levels, surfaces = [], []
                for edge in graph.edges_by_parents[node.level]:
                    if edge.relation.startswith('op'):
                        child_node = graph.get_node_by_level(edge.tgt_level)
                        rank = int(edge.relation[2:])
                        surfaces.append((child_node.name[1:-1], rank))
                        levels.append((edge.tgt_level, rank))
                levels = [level for level, rank in sorted(levels, key=lambda x: x[1])]
                surfaces = [surface for surface, rank in sorted(surfaces, key=lambda x: x[1])]
                entity_surface = '\t'.join(surfaces)
                if entity_surface in entities:
                    print('Multiple entities: {0}'.format(graph.n))
                    continue
                entities[entity_surface] = node.level, levels
        return entities

    def match(self, words, stemmed_words, postags, graph, align_results):
        """
        :param words: list[str]
        :param stemmed_words: list[str]
        :param postags: list[str]
        :param graph: Alignment
        :param align_results: AlignResults
        :return:
        """
        entities = self.collect_entities(graph)

        n_words = len(words)
        for start in range(n_words):
            for end in range(start + 1, n_words + 1):
                form = '\t'.join(words[start: end])
                if form in entities:
                    self.add(start, end, entities[form][0], entities[form][1], align_results)
                else:
                    for surface in entities:
                        if form.lower() == surface.lower():
                            self.add(start, end, entities[surface][0], entities[surface][1], align_results)


class FuzzyNamedEntityMatcher(NamedEntityMatcher):
    def __init__(self):
        super(FuzzyNamedEntityMatcher, self).__init__()
        self.helper = CommonPrefixMatchHelper()

    def entity_equal(self, words, entity):
        compressed_surface = ''.join(words)
        compressed_entity = entity.replace('\t', '')
        if compressed_entity == compressed_surface:
            return True
        if compressed_entity.lower() == compressed_surface.lower():
            return True
        if compressed_entity in ('UnitedStates', 'UnitedStatesOfAmerica') and \
                compressed_surface in ('US', 'U.S.', 'u.s.', 'America', 'American', 'Americans'):
            return True
        if compressed_entity == 'GreatBritain' and compressed_surface == 'British':
            return True
        if compressed_entity == 'UnitedNations' and compressed_surface == 'UN':
            return True
        if compressed_entity == 'China' and compressed_surface in ('Chinese', 'Sino'):
            return True
        if compressed_entity == 'SovietUnion' and compressed_surface == 'Soviet':
            return True
        if compressed_entity == 'SaudiArabia' and compressed_surface == 'Saudi':
            return True
        # Korea , People 's Republic => People's Republic of Korea
        comma_removed_compressed_surface = compressed_surface.replace(',', '')
        if compressed_entity == comma_removed_compressed_surface:
            return True
        comma_swapped_compressed_surface = ''.join(compressed_surface.split(',', 1)[::-1])
        if compressed_entity == comma_swapped_compressed_surface:
            return True
        return False

    def match(self, words, stemmed_words, postags, graph, align_results):
        entities = self.collect_entities(graph)

        n_words = len(words)
        for entity in entities:
            surfaces = entity.split('\t')
            for start in range(n_words):
                for end in reversed(range(start + 1, n_words + 1)):
                    # to gureentee maximum matching, only one fuzzy matched entity is possible
                    # which has the maximum length.
                    if len(surfaces) == end - start:
                        if all(self.helper.is_fuzzy_match_wordlist(wordlist, surface)
                               for wordlist, surface in zip(stemmed_words[start: end], surfaces)):
                            self.add(start, end, entities[entity][0], entities[entity][1], align_results)
                            break
                        elif all(self.helper.is_fuzzy_match_wordlist(wordlist, surface.lower())
                                 for wordlist, surface in zip(stemmed_words[start: end], surfaces)):
                            self.add(start, end, entities[entity][0], entities[entity][1], align_results)
                            break
                    if self.entity_equal(words[start: end], entity):
                        self.add(start, end, entities[entity][0], entities[entity][1], align_results)
                        break


class SemanticNamedEntityMatcher(NamedEntityMatcher):
    def __init__(self, threshold=0.6, lower=True):
        super(SemanticNamedEntityMatcher, self).__init__()
        self.helper = SemanticMatchHelper(threshold, lower)

    def match(self, words, stemmed_words, postags, graph, align_results):
        entities = self.collect_entities(graph)

        n_words = len(words)
        for start in range(n_words):
            for end in range(start + 1, n_words + 1):
                for entity in entities:
                    surfaces = entity.split('\t')
                    if len(surfaces) != end - start:
                        continue
                    if all(self.helper.is_semantic_match_word(word, surface)
                           for word, surface in zip(words[start: end], surfaces)):
                        self.add(start, end, entities[entity][0], entities[entity][1], align_results)


class DateEntityMatcher(EntityMatcher):
    def __init__(self):
        super(DateEntityMatcher, self).__init__()

    def match(self, words, stemmed_words, postags, graph, align_results):
        for node in graph.true_nodes():
            if not graph.is_date_entity(node, consider_alignment=False):
                continue
            year, month, day, weekday, decade = None, None, None, None, None
            levels = []
            for edge in graph.edges_by_parents[node.level]:
                if edge.relation == 'year':
                    year_node = graph.get_node_by_level(edge.tgt_level)
                    year = int(year_node.name)
                    levels.append(year_node.level)
                elif edge.relation == 'month':
                    month_node = graph.get_node_by_level(edge.tgt_level)
                    month = int(month_node.name)
                    levels.append(month_node.level)
                elif edge.relation == 'day':
                    day_node = graph.get_node_by_level(edge.tgt_level)
                    day = int(day_node.name)
                    levels.append(day_node.level)
                elif edge.relation == 'weekday':
                    weekday_node = graph.get_node_by_level(edge.tgt_level)
                    weekday = weekday_node.name
                    levels.append(weekday_node.level)
                elif edge.relation == 'decade':
                    decade_node = graph.get_node_by_level(edge.tgt_level)
                    decade = decade_node.name
                    levels.append(decade_node.level)
            if any([year, month, day]):
                n_words = len(words)
                for start in range(n_words):
                    for end in range(start + 1, n_words + 1):
                        form = ' '.join(words[start: end])
                        all_dates = parse_all_dates(form)
                        if len(all_dates) > 0:
                            success = False
                            for date_result, flags in all_dates:
                                if self.is_date_equal(year, 'year', date_result, flags[0]) and \
                                        self.is_date_equal(month, 'month', date_result, flags[1]) and \
                                        self.is_date_equal(day, 'day', date_result, flags[2]):
                                    success = True
                                    break
                            if success:
                                self.add(start, end, node.level, levels, align_results)
            elif weekday is not None:
                for start, word in enumerate(words):
                    if word.lower() == weekday:
                        self.add(start, start + 1, node.level, levels, align_results)
            elif decade is not None:
                for start, word in enumerate(words):
                    if word[:4] == decade:
                        self.add(start, start + 1, node.level, levels, align_results)

    @staticmethod
    def is_date_equal(field, field_name, data, flag):
        if field is None:
            if flag:
                return False
        else:
            if not flag:
                return False
            elif field != getattr(data, field_name, ""):
                return False
        return True


class URLEntityMatcher(EntityMatcher):
    def __init__(self):
        super(URLEntityMatcher, self).__init__()

    @classmethod
    def collect_entities(cls, graph):
        entities = {}
        for node in graph.true_nodes():
            if node.name != 'url-entity':
                continue
            levels, surface = [], None
            if node.level in graph.edges_by_parents:
                for edge in graph.edges_by_parents[node.level]:
                    if edge.relation == 'value':
                        surface = edge.tgt_name[1:-1]
                    levels.append(edge.tgt_level)
            if surface is not None:
                entities[surface] = node.level, levels
        return entities

    def match(self, words, stemmed_words, postags, graph, align_results):
        """
        :param words: list[str]
        :param stemmed_words: list[str]
        :param postags: list[str]
        :param graph: Alignment
        :param align_results: AlignResults
        :return:
        """
        entities = self.collect_entities(graph)

        n_words = len(words)
        for start in range(n_words):
            for end in range(start + 1, n_words + 1):
                # to cope the erroneous tokenization.
                word = ''.join(words[start: end])
                for surface in entities:
                    if word != surface and word.lower() != surface:
                        continue
                    self.add(start, start + 1, entities[surface][0], entities[surface][1], align_results)


class OrdinalEntityMatcher(EntityMatcher):
    kOrdinalNumber = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                      'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10}

    def __init__(self):
        super(OrdinalEntityMatcher, self).__init__()

    @classmethod
    def collect_entities(cls, graph):
        entities = {}
        for node in graph.true_nodes():
            if node.name != 'ordinal-entity':
                continue
            levels, surface = [], None
            for edge in graph.edges_by_parents[node.level]:
                if edge.relation == 'value':
                    surface = edge.tgt_name
                levels.append(edge.tgt_level)
            if surface is not None:
                entities[surface] = node.level, levels
        return entities

    def normalize_ordinal_word(self, word):
        word = word.lower()
        if word in self.kOrdinalNumber:
            return str(self.kOrdinalNumber[word])
        for number, suffix in (('1', 'st'), ('2', 'nd'), ('3', 'rd')):
            if len(word) > 2 and word.endswith(suffix) and word[:-2].isdigit() and word[-3] == number:
                return word[:-2]
        if len(word) > 2 and word.endswith('th') and word[:-2].isdigit() and word[-3] not in ('1', '2', '3'):
            return word[:-2]
        return word

    def match(self, words, stemmed_words, postags, graph, align_results):
        entities = self.collect_entities(graph)
        for start, word in enumerate(words):
            ordinal_word = self.normalize_ordinal_word(word)
            if ordinal_word is None:
                continue
            if ordinal_word in entities:
                self.add(start, start + 1, entities[ordinal_word][0], entities[ordinal_word][1], align_results)


class MinusPolarityMatcher(Matcher):
    kMinusPolarityTokens = ("no", "not", "non", "nt", "n't")

    def __init__(self):
        super(MinusPolarityMatcher, self).__init__()

    def match(self, words, stemmed_words, postags, graph, align_results):
        for node in graph.true_nodes():
            if node.name != '-':
                continue
            for start, word in enumerate(words):
                if word.lower() in self.kMinusPolarityTokens:
                    align_results.add(start, start + 1, node.level, None)


class BelocatedAtMatcher(Matcher):
    kLocationTokens = ('in', 'on', 'at', 'away', 'near')

    def __init__(self):
        super(BelocatedAtMatcher, self).__init__()

    def match(self, words, stemmed_words, postags, graph, align_results):
        for node in graph.true_nodes():
            if node.name != 'be-located-at-91':
                continue
            for start, word in enumerate(words):
                if word.lower() in self.kLocationTokens:
                    align_results.add(start, start + 1, node.level, None)


class TemporalQuantityMatcher(Matcher):
    kTemporalExpressions = {'daily': 'day',
                            'monthly': 'month',
                            'yearly': 'year',
                            'annual': 'year',
                            'annually': 'year'}

    def __init__(self):
        super(TemporalQuantityMatcher, self).__init__()

    def match(self, words, stemmed_words, postags, graph, align_results):
        for node in graph.true_nodes():
            if node.name != 'temporal-quantity':
                continue
            if node.level not in graph.edges_by_parents:
                continue
            unit = [e for e in graph.edges_by_parents[node.level] if e.relation == 'unit']
            quant = [e for e in graph.edges_by_parents[node.level] if e.relation == 'quant']
            if len(unit) != 1 or unit[0].tgt_name not in ('day', 'month', 'year')\
                    or len(quant) != 1 or quant[0].tgt_name != '1':
                continue
            for start, word in enumerate(words):
                word = word.lower()
                if word in self.kTemporalExpressions and self.kTemporalExpressions[word] == unit[0]:
                    align_results.add(start, start + 1, node.level, None)
                    align_results.add(start, start + 1, unit[0].tgt_level, node.level)
                    align_results.add(start, start + 1, quant[0].tgt_level, node.level)


__all__ = [
    'init_word2vec',
    'WordMatcher',
    'FuzzyWordMatcher',
    'MorphosemanticLinkMatcher',
    'SemanticWordMatcher',
    'FuzzySpanMatcher',
    'NamedEntityMatcher',
    'FuzzyNamedEntityMatcher',
    'SemanticNamedEntityMatcher',
    'DateEntityMatcher',
    'URLEntityMatcher',
    'OrdinalEntityMatcher',
    'MinusPolarityMatcher',
    'BelocatedAtMatcher',
    'TemporalQuantityMatcher'
]
