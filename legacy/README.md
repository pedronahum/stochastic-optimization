# Legacy Code Archive

This directory contains the original implementations and migration documentation from the JAX modernization effort completed in November 2024.

## Contents

### `old_problems/`
Original problem implementations before JAX migration:
- **AdaptiveMarketPlanning** - Original adaptive market planning implementation
- **AssetSelling** - Original asset selling implementation
- **BloodManagement** - Original blood management implementation
- **ClinicalTrials** - Original clinical trials implementation
- **EnergyStorage_I** - Original energy storage implementation
- **MedicalDecisionDiabetes** - Original medical decision diabetes implementation
- **StochasticShortestPath_Dynamic** - Original SSP dynamic implementation
- **StochasticShortestPath_Static** - Original SSP static implementation
- **TwoNewsvendor** - Original two newsvendor implementation

### `migration_docs/`
Documentation created during the JAX migration process:
- **MIGRATION_COMPLETE.md** - Final migration summary
- **ANALYSIS_INDEX.md** - Problem analysis index
- **ASSET_SELLING_MIGRATION.md** - Asset selling migration notes
- **LEGACY_PROBLEM_ANALYSIS.md** - Detailed legacy code analysis
- **MIGRATION_QUICK_REFERENCE.md** - Quick reference guide
- **MYPY_COMPLIANCE.md** - Type safety compliance tracking
- **TYPING_STATUS.md** - Type annotation status
- **Modernization/** - Additional modernization notes

## Migration Details

All problems were successfully migrated to JAX-native implementations in November 2024:
- ✅ 9/9 problems migrated
- ✅ 230/230 tests passing
- ✅ 100% mypy strict compliance
- ✅ Full JIT compilation support

The new implementations are located in `problems/` and follow modern JAX best practices.

## For Historical Reference Only

These files are kept for historical reference and should not be used for new development. All new development should use the JAX-native implementations in the main repository (`problems/`, `tests/`, etc.).
