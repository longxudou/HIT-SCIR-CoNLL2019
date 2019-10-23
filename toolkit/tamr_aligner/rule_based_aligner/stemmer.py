#!/usr/bin/env python
from __future__ import unicode_literals
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


class Stemmer(object):
    kMinusPrefix2 = ('un', 'in', 'il', 'im', 'ir', 'il', 'Un', 'In', 'Il', 'Im', 'Ir', 'Il')
    kMinusPrefix3 = ('non', 'Non')

    kMonths = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
               'September', 'October', 'November', 'December']

    kNumbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    kExceptions = {'.': ('multi-sentence', ),
                   ';': ('and', 'multi-sentence', ),
                   ':': ('mean', ),
                   '!': ('expressive', ),
                   '..': ('expressive', ),
                   '...': ('expressive', ),
                   '....': ('expressive', ),
                   '?': ('interrogative', ),
                   '%': ('percentage-entity', ),
                   '$': ('dollar', ),
                   'also': ('include',),
                   'anti': ('oppose', 'counter'),
                   'but': ('contrast', 'have-concession'),
                   'while': ('contrast', ),
                   'because': ('cause',),
                   'whereby': ('cause', ),
                   'if': ('cause', 'interrogative'),
                   'by': ('cause', ),
                   'for': ('cause', ),
                   'so': ('infer', 'cause'),
                   'since': ('cause', ),
                   'on': ('cause', ),
                   'in': ('cause', ),
                   'against': ('-', ),
                   'no': ('-',),
                   'non': ('-', ),
                   'not': ('-', ),
                   'n\'t': ('-', ),
                   'never': ('-', ),
                   'yet': ('-', ),
                   'neither': ('-', ),
                   'of': ('include', 'have-manner', ),
                   'might': ('possible', ),
                   'may': ('possible', ),
                   'maybe': ('possible', ),
                   'could': ('possible', ),
                   'can': ('possible', ),
                   'cant': ('possible', ),
                   'cannot': ('possible', ),
                   'can\'t': ('possible', ),
                   'should': ('recommend', ),
                   'who': ('amr-unknown', ),
                   'what': ('amr-unknown', ),
                   'how': ('amr-unknown', ),
                   'as': ('and', 'same', 'contrast',),
                   'with': ('and', ),
                   'plus': ('and', ),
                   '-': ('and', ),
                   'without': ('-', ),
                   'me': ('i', ),
                   'my': ('i', ),
                   'her': ('she', ),
                   'his': ('he', ),
                   'him': ('he', ),
                   'us': ('we', ),
                   'our': ('we', ),
                   'ours': ('we', ),
                   'your': ('you', ),
                   'yourself': ('you', ),
                   'these': ('this', ),
                   'those': ('that', ),
                   'o.k.': ('okay', ),
                   'death': ('die',),
                   'deaths': ('die', ),
                   'like': ('resemble', ),
                   'similar': ('resemble', ),
                   'right': ('entitle', ),
                   'rights': ('entitle',),
                   'must': ('obligate',),
                   'etc': ('et-cetera',),
                   'according': ('say', ),}

    def __init__(self):
        pass

    def stem(self, word, postag):
        ret = set()
        ret.add(word)
        ret.add(word.lower())

        # lemmatize
        if postag is not None:
            ret.add(lemmatizer.lemmatize(word.lower(), postag))
        else:
            ret.add(lemmatizer.lemmatize(word.lower(), 'n'))
            ret.add(lemmatizer.lemmatize(word.lower(), 'v'))
            ret.add(lemmatizer.lemmatize(word.lower(), 'a'))
            ret.add(lemmatizer.lemmatize(word.lower(), 's'))

        # normalize month
        month_normalized_word = self._normalize_month(word)
        if month_normalized_word is not None:
            ret.add(month_normalized_word)

        # normalize number
        number_normalized_word = self._normalize_number(word)
        if number_normalized_word is not None:
            ret.add(number_normalized_word)

        # normalize exceptions
        exception_normalized_words = self._normalize_exceptions(word)
        if exception_normalized_words is not None:
            for exception_normalized_word in exception_normalized_words:
                ret.add(exception_normalized_word)

        other_normalized_word = self._normalize_others(word)
        if other_normalized_word is not None:
            ret.add(other_normalized_word)
        return ret

    def _normalize_number(self, word):
        if word.lower() in self.kNumbers:
            return str(self.kNumbers.index(word.lower()) + 1)
        elif ',' in word and word.replace(',', '').isdigit():
            return word.replace(',', '')
        return None

    def _normalize_month(self, word):
        if word in self.kMonths:
            return str(self.kMonths.index(word) + 1)
        return None

    def _normalize_exceptions(self, word):
        if word.lower() in self.kExceptions:
            return self.kExceptions[word.lower()]
        return None

    def _normalize_others(self, word):
        if word[:3] in self.kMinusPrefix3:
            return word[3:]
        elif word[:2] in self.kMinusPrefix2:
            return word[2:]
        elif word.endswith('er'):
            return word[:-2]
        elif word.endswith('ers'):
            return word[:-3]
        return None

