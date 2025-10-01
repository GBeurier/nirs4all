# Deprecated Files Archive

This directory contains files that have been deprecated and moved from the main codebase as part of the architecture simplification.

## Date: October 1, 2025

## Moved Files:

### 📁 controllers/models/
- `abstract_model_controller.py` - Replaced by simplified `BaseModelController`
- `old_base_model_controller.py` - Previous version of base controller
- `cv_averaging.py` - Functionality integrated into `BaseModelController`
- `model_naming.py` - Functionality integrated into `model_utils.py`
- `optuna_manager.py` - Functionality integrated into `optuna_controller.py`
- `enums.py` - ModelMode enum only used by deprecated controllers
- `prediction_store_backup.py` - Backup file no longer needed

### 📁 core/
- Entire `core/` module - Replaced by simplified controller architecture
  - `model/` - Model management classes
  - `finetuner/` - Finetuning classes
  - `utils.py` - Utility functions

### 📁 tests/
- `test_optuna_integration.py` - Tests for deprecated AbstractModelController
- `test_finetuning_focused.py` - Tests for deprecated controllers
- `run_finetuning_tests.py` - Test runner for deprecated functionality
- `test_scaling_fix.py` - Legacy test file
- `test_modular_integration.py` - Legacy test file
- `core/` - Tests for deprecated core module

### 📁 controllers/models/deprec/
- All files from the existing deprec folder moved to main deprec

## Reason for Deprecation:

The architecture was simplified following user requirements:
1. **External prediction storage** - Predictions moved outside dataset as controller arguments
2. **Simplified controllers** - Complex inheritance replaced with clean BaseModelController
3. **Integrated functionality** - Separate manager classes integrated into main controllers
4. **Better separation of concerns** - Logic properly separated into 3 files maximum

## Current Active Architecture:

### ✅ Active Files:
- `BaseModelController` - Simplified base class
- `model_utils.py` - Model utilities and naming
- `optuna_controller.py` - Integrated optuna functionality
- `prediction_store.py` - External prediction storage functions
- Framework-specific controllers (sklearn, tensorflow, torch)

### ✅ Verification:
- Q1.py runs successfully ✅
- External prediction store working ✅
- Average/weighted-average calculations working ✅
- Enhanced output format working ✅

## Note:
These files are kept for reference and potential rollback if needed. They can be safely deleted once the new architecture is fully validated in production.