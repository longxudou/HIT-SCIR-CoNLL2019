#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import sys
from datetime import datetime
_DATE_FORMATS = {
    '%y0000': (True, False, False),
    '%y%m00': (True, True, False),
    '%y%m%d': (True, True, True),
    '%Y0000': (True, False, False),
    '%Y%m00': (True, True, False),
    '%d %B %Y': (True, True, True),
    '%d %B': (True, True, False),
    '%d %Y': (True, False, True),
    '%Y%m%d': (True, True, True),
    '%Y-%m-%d': (True, True, True),
    '%m/%d': (False, True, True),
    '%m/%d/%Y': (True, True, True),
    '%m - %d - %Y': (True, True, True),
    '%B %Y': (True, True, False),
    '%B , %Y': (True, True, False),
    '%B %d %Y': (True, True, True),
    '%B %d , %Y': (True, True, True),
    '%B %d': (False, True, True),
    '%B %dst': (False, True, True),
    '%B %dnd': (False, True, True),
    '%B %drd': (False, True, True),
    '%B %dth': (False, True, True),
    '%B': (False, True, False),
    '%Y': (True, False, False),
    '%y': (True, False, False),
}


def parse_date(expression):
    results = []
    for format_ in _DATE_FORMATS:
        try:
            result = datetime.strptime(expression, format_)
            results.append((result, _DATE_FORMATS[format_]))
        except:
            continue
    results = list(filter(lambda result: 1900 <= result[0].year < 2100, results))
    if len(results) > 1:
        return results[0]
    elif len(results) == 1:
        return results[0]
    else:
        return None, (False, False, False)


def parse_all_dates(expression):
    results = []
    for format_ in _DATE_FORMATS:
        try:
            result = datetime.strptime(expression, format_)
            results.append((result, _DATE_FORMATS[format_]))
        except:
            continue
    results = list(filter(lambda r: 1900 <= r[0].year < 2100, results))
    return results


def test():
    for line in open(sys.argv[1], 'r'):
        expression, fields = line.strip().split('|||')
        expression = expression.strip()
        result = parse_date(expression)
        slots = result[1]
        for field in fields:
            if field == 'year':
                assert slots[0]
            if field == 'month':
                assert slots[1]
            if field == 'day':
                assert slots[2]
        print('{0} ||| {1} ||| {2}'.format(expression, slots, fields), file=sys.stderr)


if __name__ == "__main__":
    test()
