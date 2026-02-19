"""Test script for Phase 1 generator refactoring."""

from nirs4all.pipeline.config.generator import OR_KEYWORD, RANGE_KEYWORD, count_combinations, expand_spec, is_generator_node, sample_with_seed


def test_basic_expand():
    """Test basic OR expansion."""
    print('=== Basic OR test ===')
    result = expand_spec({'_or_': ['A', 'B', 'C']})
    print(f'expand_spec: {result}')
    assert result == ['A', 'B', 'C'], f"Expected ['A', 'B', 'C'], got {result}"
    print('PASS')

def test_pick():
    """Test OR with pick."""
    print('=== Pick test ===')
    result = expand_spec({'_or_': ['A', 'B', 'C'], 'pick': 2})
    print(f'expand_spec with pick=2: {result}')
    assert len(result) == 3, f"Expected 3 combinations, got {len(result)}"
    print('PASS')

def test_range():
    """Test range expansion."""
    print('=== Range test ===')
    result = expand_spec({'_range_': [1, 5]})
    print(f'expand_spec range [1,5]: {result}')
    assert result == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {result}"
    print('PASS')

def test_keyword_constants():
    """Test keyword constants are accessible."""
    print('=== Keyword constants ===')
    print(f'OR_KEYWORD: {OR_KEYWORD}')
    print(f'RANGE_KEYWORD: {RANGE_KEYWORD}')
    assert OR_KEYWORD == '_or_', f"Expected '_or_', got {OR_KEYWORD}"
    assert RANGE_KEYWORD == '_range_', f"Expected '_range_', got {RANGE_KEYWORD}"
    print('PASS')

def test_is_generator_node():
    """Test is_generator_node function."""
    print('=== is_generator_node ===')
    or_node = {'_or_': ['A', 'B']}
    normal_node = {'class': 'Test'}
    print(f'is_generator_node with _or_: {is_generator_node(or_node)}')
    print(f'is_generator_node without: {is_generator_node(normal_node)}')
    assert is_generator_node(or_node) is True
    assert is_generator_node(normal_node) is False
    print('PASS')

def test_sample_with_seed():
    """Test deterministic sampling with seed."""
    print('=== sample_with_seed ===')
    items = ['A', 'B', 'C', 'D', 'E']
    s1 = sample_with_seed(items, 2, seed=42)
    s2 = sample_with_seed(items, 2, seed=42)
    s3 = sample_with_seed(items, 2, seed=99)
    print(f'seed=42 run1: {s1}')
    print(f'seed=42 run2: {s2}')
    print(f'seed=99: {s3}')
    assert s1 == s2, f"Same seed should give same result: {s1} vs {s2}"
    print(f'Deterministic (s1==s2): {s1 == s2}')
    print('PASS')

def test_count_combinations():
    """Test counting combinations."""
    print('=== count_combinations ===')
    count = count_combinations({'_or_': ['A', 'B', 'C']})
    print(f'count for 3 choices: {count}')
    assert count == 3

    count = count_combinations({'_or_': ['A', 'B', 'C'], 'pick': 2})
    print(f'count for 3 choices, pick 2: {count}')
    assert count == 3  # C(3,2) = 3

    count = count_combinations({'_range_': [1, 10]})
    print(f'count for range [1,10]: {count}')
    assert count == 10
    print('PASS')

if __name__ == '__main__':
    test_basic_expand()
    test_pick()
    test_range()
    test_keyword_constants()
    test_is_generator_node()
    test_sample_with_seed()
    test_count_combinations()
    print('\n=== All tests passed! ===')
