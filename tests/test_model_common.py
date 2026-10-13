"""Unit tests for model/networks/common.py."""
from pidu.model.networks.common import select_dict_from_subset


def test_returns_value_for_matching_prefix():
  d = {'system1': 'a', 'system2': 'b'}
  assert select_dict_from_subset(d, 'system1_modelA') == 'a'
  assert select_dict_from_subset(d, 'system2_0') == 'b'


def test_returns_none_when_no_key_matches():
  d = {'system1': 'a', 'system2': 'b'}
  assert select_dict_from_subset(d, 'system3_modelA') is None


def test_returns_none_for_empty_dict():
  assert select_dict_from_subset({}, 'system1') is None


def test_first_match_wins_when_multiple_keys_are_prefixes():
  # dict order is preserved in Python 3.7+, so the first inserted matching
  # key should be the one returned.
  d = {'sys': 'first', 'system1': 'second'}
  assert select_dict_from_subset(d, 'system1_modelA') == 'first'
