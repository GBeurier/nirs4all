# Dataset and Indexer Test Suite

This directory contains comprehensive tests for the nirs4all dataset management system, specifically testing the `SpectroDataset` and `Indexer` classes.

## Test Structure

### Core Test Files

- **`test_dataset.py`** - Tests for the main `SpectroDataset` class
- **`test_indexer.py`** - Tests for the `Indexer` class
- **`test_utils.py`** - Shared test utilities, fixtures, and sample data generators
- **`run_tests.py`** - Test runner script with various execution options

### Test Categories

#### Unit Tests (`@pytest.mark.unit`)
- Individual method testing
- Input validation
- Error handling
- Edge cases

#### Integration Tests (`@pytest.mark.integration`)
- Complete workflow scenarios
- Multi-component interactions
- End-to-end data processing

#### Dataset Tests (`@pytest.mark.dataset`)
- Feature and target management
- Data retrieval with filters
- Multi-source handling
- Sample augmentation
- Cross-validation folds

#### Indexer Tests (`@pytest.mark.indexer`)
- Index management
- Filtering operations
- Column value retrieval
- Sample tracking
- Augmentation tracking

## Sample Data

The test suite includes realistic sample data generators:

### Spectral Data
- **NIR spectra**: Simulates Near-Infrared spectroscopy data (700-2500 nm)
- **Visible spectra**: Simulates visible light spectra (400-700 nm)
- **Raman spectra**: Simulates Raman spectroscopy data (0-4000 cm⁻¹)

### Target Data
- **Classification**: Multi-class targets with configurable class balance
- **Regression**: Continuous targets with realistic noise patterns
- **Binary classification**: Two-class targets

### Metadata
- Sample IDs, batch information, instrument details
- Environmental conditions (temperature, humidity)
- Quality flags and operator information

## Running Tests

### Using the Test Runner Script

```bash
# Run all tests
python tests/run_tests.py all

# Run only dataset tests
python tests/run_tests.py dataset

# Run only indexer tests
python tests/run_tests.py indexer

# Run integration tests
python tests/run_tests.py integration

# Run tests with coverage
python tests/run_tests.py coverage

# Quick development tests (stop on first failure)
python tests/run_tests.py quick
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dataset.py -v

# Run tests with specific marker
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "dataset"

# Run tests with coverage
pytest tests/ --cov=nirs4all.dataset --cov-report=html

# Run specific test method
pytest tests/test_dataset.py::TestSpectroDataset::test_add_features_single_partition
```

## Test Implementation Guidelines

### Adding New Tests

1. **Follow naming conventions**: `test_<functionality>_<scenario>`
2. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
3. **Include docstrings**: Describe what the test validates
4. **Use fixtures**: Leverage existing fixtures from `test_utils.py`
5. **Test edge cases**: Include error conditions and boundary cases

### Test Data

- Use `SampleDataGenerator` for consistent test data
- Set random seeds for reproducibility
- Create realistic data that mimics actual use cases
- Test with different data sizes and configurations

### Assertions

- Validate data shapes and types
- Check data consistency between features and targets
- Verify indexer state after operations
- Test filter logic thoroughly
- Validate error handling

## Expected Test Implementation

Each test skeleton includes TODO comments indicating what should be implemented:

```python
def test_add_features_single_partition(self):
    """Test adding features to a single partition."""
    # TODO: Implement test
    # - Add train features with partition="train"
    # - Verify features are stored correctly
    # - Check indexer is updated properly
    pass
```

### Implementation Steps for Each Test:

1. **Setup**: Create test data using `SampleDataGenerator`
2. **Action**: Perform the operation being tested
3. **Verification**: Assert expected outcomes
4. **Validation**: Check data consistency and integrity

### Example Implementation:

```python
def test_add_features_single_partition(self):
    """Test adding features to a single partition."""
    # Setup
    features = self.sample_data.create_spectral_features(n_samples=50)
    filter_dict = {"partition": "train"}

    # Action
    self.dataset.add_features(filter_dict, features)

    # Verification
    retrieved_features = self.dataset.features(filter_dict)
    assert retrieved_features.shape == features.shape
    assert np.allclose(retrieved_features, features)

    # Validation
    train_indices = self.dataset.index_column("sample", {"partition": "train"})
    assert len(train_indices) == 50
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines with:

- Automated test execution on code changes
- Coverage reporting and thresholds
- Performance regression detection
- Multi-platform compatibility testing

## Contributing

When adding new functionality to the dataset system:

1. Add corresponding test skeletons
2. Implement tests alongside code changes
3. Ensure all tests pass before submitting
4. Update this documentation if needed

## Test Data Compliance Verification

The tests include comprehensive validation to ensure:

- **Data Integrity**: Features and targets remain consistent
- **Index Accuracy**: Sample indices correctly track data
- **Filter Correctness**: Filtering returns expected subsets
- **Augmentation Tracking**: Augmented samples are properly indexed
- **Cross-validation Consistency**: Folds contain correct sample assignments
