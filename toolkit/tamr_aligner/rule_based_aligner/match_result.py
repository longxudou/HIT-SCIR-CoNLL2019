#!/usr/bin/env python
from __future__ import unicode_literals


class MatchResult(object):
    def __init__(self, level, signature):
        self.level = level
        self.signature = signature

    def __eq__(self, other):
        if isinstance(other, MatchResult):
            return self.level == other.level
        return False

    def __str__(self):
        return '{0}={1}'.format(self.signature, self.level)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.level.__hash__()


class EntityMatchResult(MatchResult):
    def __init__(self, level, children_levels, signature):
        super(EntityMatchResult, self).__init__(level, signature)
        self.children_levels = children_levels

    def __str__(self):
        return '{0}=({1}, {2})'.format(self.signature, self.level, self.children_levels)


class WordMatchResult(MatchResult):
    def __init__(self, level, signature='word'):
        super(WordMatchResult, self).__init__(level, signature)

    def __eq__(self, other):
        if isinstance(other, WordMatchResult) or \
                isinstance(other, FuzzyWordMatchResult) or \
                isinstance(other, SemanticWordMatchResult):
            return self.level == other.level
        return False


class FuzzyWordMatchResult(WordMatchResult):
    def __init__(self, level):
        super(FuzzyWordMatchResult, self).__init__(level, '(fuzzy)word')


class MorphosemanticLinkMatchResult(WordMatchResult):
    def __init__(self, level):
        super(MorphosemanticLinkMatchResult, self).__init__(level, '(morph)word')


class SemanticWordMatchResult(WordMatchResult):
    def __init__(self, level):
        super(SemanticWordMatchResult, self).__init__(level, '(sem)word')


class FuzzySpanMatchResult(MatchResult):
    def __init__(self, level):
        super(FuzzySpanMatchResult, self).__init__(level, '(fuzzy)span')


class NamedEntityMatchResult(EntityMatchResult):
    def __init__(self, level, children_levels, signature='entity'):
        super(NamedEntityMatchResult, self).__init__(level, children_levels, signature)

    def __eq__(self, other):
        if isinstance(other, FuzzyNamedEntityMatchResult) or \
                isinstance(other, NamedEntityMatchResult) or \
                isinstance(other, SemanticNamedEntityMatchResult):
            return self.level == other.level
        return False


class FuzzyNamedEntityMatchResult(NamedEntityMatchResult):
    def __init__(self, level, children_levels):
        super(FuzzyNamedEntityMatchResult, self).__init__(level, children_levels, '(fuzzy)entity')


class SemanticNamedEntityMatchResult(NamedEntityMatchResult):
    def __init__(self, level, children_levels):
        super(SemanticNamedEntityMatchResult, self).__init__(level, children_levels, '(sem)entity')


class URLEntityMatchResult(EntityMatchResult):
    def __init__(self, level, children_levels):
        super(URLEntityMatchResult, self).__init__(level, children_levels, 'url-entity')


class OrdinalEntityMatchResult(EntityMatchResult):
    def __init__(self, level, children_levels):
        super(OrdinalEntityMatchResult, self).__init__(level, children_levels, 'ordinal-entity')


class DateEntityMatchResult(MatchResult):
    def __init__(self, level, children_levels):
        super(DateEntityMatchResult, self).__init__(level, 'date-entity')
        self.children_levels = children_levels

    def __str__(self):
        return 'date-entity=({0}, {1})'.format(self.level, self.children_levels)


class MinusPolarityMatchResult(MatchResult):
    def __init__(self, level):
        super(MinusPolarityMatchResult, self).__init__(level, 'minus')


class EntityTypeMatchResult(MatchResult):
    def __init__(self, level):
        super(EntityTypeMatchResult, self).__init__(level, 'entity_type')


class QuantityMatchResult(MatchResult):
    def __init__(self, level):
        super(QuantityMatchResult, self).__init__(level, 'quantity')


class PersonOfUpdateResult(MatchResult):
    def __init__(self, level):
        super(PersonOfUpdateResult, self).__init__(level, 'person_of')


class PersonUpdateResult(MatchResult):
    def __init__(self, level):
        super(PersonUpdateResult, self).__init__(level, 'person')


class GovernmentOrganizationUpdateResult(MatchResult):
    def __init__(self, level):
        super(GovernmentOrganizationUpdateResult, self).__init__(level, 'gov_org')


class MinusPolarityPrefixesUpdateResult(MatchResult):
    def __init__(self, level):
        super(MinusPolarityPrefixesUpdateResult, self).__init__(level, 'minus_prefix')


class DegreeUpdateResult(MatchResult):
    def __init__(self, level):
        super(DegreeUpdateResult, self).__init__(level, 'degree')


class RelativePositionUpdateResult(MatchResult):
    def __init__(self, level):
        super(RelativePositionUpdateResult, self).__init__(level, 'relative_position')


class HaveOrgRoleUpdateResult(MatchResult):
    def __init__(self, level):
        super(HaveOrgRoleUpdateResult, self).__init__(level, 'have-org-role-91')


class CauseUpdateResult(MatchResult):
    def __init__(self, level):
        super(CauseUpdateResult, self).__init__(level, 'cause01')


class BelocatedAtMatchResult(MatchResult):
    def __init__(self, level):
        super(BelocatedAtMatchResult, self).__init__(level, 'be-located-91')


class ImperativeUpdateResult(MatchResult):
    def __init__(self, level):
        super(ImperativeUpdateResult, self).__init__(level, 'imperative')


class PossibleUpdateResult(MatchResult):
    def __init__(self, level):
        super(PossibleUpdateResult, self).__init__(level, 'possible')


__all__ = [
    'EntityTypeMatchResult',
    'QuantityMatchResult',
    'DateEntityMatchResult',
    'URLEntityMatchResult',
    'OrdinalEntityMatchResult',
    'MinusPolarityMatchResult',
    'BelocatedAtMatchResult',
    'FuzzyWordMatchResult',
    'FuzzySpanMatchResult',
    'MorphosemanticLinkMatchResult',
    'SemanticWordMatchResult',
    'NamedEntityMatchResult',
    'FuzzyNamedEntityMatchResult',
    'SemanticNamedEntityMatchResult',

    'PersonOfUpdateResult',
    'PersonUpdateResult',
    'RelativePositionUpdateResult',
    'GovernmentOrganizationUpdateResult',
    'MinusPolarityPrefixesUpdateResult',
    'DegreeUpdateResult',
    'HaveOrgRoleUpdateResult',
    'CauseUpdateResult',
    'ImperativeUpdateResult',
    'WordMatchResult',
    'PossibleUpdateResult',
]
