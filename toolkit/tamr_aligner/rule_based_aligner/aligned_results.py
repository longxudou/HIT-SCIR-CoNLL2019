#!/usr/bin/env python


class AlignedResults(object):
    def __init__(self, multiple=True):
        self.spans_to_levels = {}
        self.levels_to_spans = {}
        self.multiple = multiple

    def add(self, start, end, level, dependent):
        if self.multiple:
            return self._mutualisticly_add(start, end, level, dependent)
        else:
            return self._exclusively_add(start, end, level, dependent)

    def _mutualisticly_add(self, start, end, level, dependent):
        added = False
        if (start, end) not in self.spans_to_levels:
            self.spans_to_levels[start, end] = set()
        if (level, dependent) not in self.spans_to_levels[start, end]:
            added = True
        self.spans_to_levels[start, end].add((level, dependent))

        if level not in self.levels_to_spans:
            self.levels_to_spans[level] = set()
        self.levels_to_spans[level].add((start, end, dependent))
        return added

    def _exclusively_add(self, start, end, level, dependent):
        # first check if the concept is aligned.
        if level in self.levels_to_spans:
            return False
        self.levels_to_spans[level] = {(start, end, dependent)}
        added = False
        if dependent is not None:
            if (start, end) not in self.spans_to_levels:
                self.spans_to_levels[start, end] = set()
            if (level, dependent) not in self.spans_to_levels[start, end]:
                added = True
            self.spans_to_levels[start, end].add((level, dependent))
        else:
            overlap = False
            for new_start, new_end in self.spans_to_levels:
                if (start < new_start < end) or (start < new_end < end):
                    overlap = True
                    break
            if not overlap:
                if (start, end) not in self.spans_to_levels:
                    self.spans_to_levels[start, end] = set()
                if (level, dependent) not in self.spans_to_levels[start, end]:
                    added = True
                self.spans_to_levels[start, end].add((level, dependent))
        return added

    def contains(self, level):
        return level in self.levels_to_spans

    def get_spans_by_level(self, level):
        return set([(start, end) for start, end, _ in self.levels_to_spans.get(level, set())])

    def get_levels_by_span(self, start, end):
        return set([level for level, _ in self.spans_to_levels.get((start, end), set())])
