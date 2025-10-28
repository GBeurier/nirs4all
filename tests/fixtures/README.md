# Test Fixtures

This directory contains test data and fixtures used across the test suite.

## Structure

- `datasets/` - Sample datasets for testing
  - `sample_data/` - General sample datasets
  - `regression_data/` - Regression-specific test data
- `pipelines/` - Pipeline configurations for testing
  - `test_configs/` - Test pipeline configurations

## Usage

Test fixtures should be:
- Small and focused
- Well-documented
- Representative of real-world data
- Reusable across multiple tests

## Adding New Fixtures

When adding new test fixtures:
1. Place them in the appropriate subdirectory
2. Document their purpose in a comment or README
3. Keep file sizes minimal
4. Use descriptive names
