# Legacy Problem Migration Analysis - Complete Documentation

## Overview

This directory now contains comprehensive analysis of the 5 remaining legacy problem directories to determine optimal migration sequence to JAX-native implementation.

**Key Finding**: Start with **AdaptiveMarketPlanning** - it's the smallest, simplest, and most JAX-friendly.

---

## Documents Generated

### 1. LEGACY_PROBLEM_ANALYSIS.md (20 KB)

**Complete technical analysis of all 5 problems**

Includes for each problem:
- File structure and line counts
- State space complexity analysis
- Decision variable analysis
- Problem dynamics explanation
- Special features and challenges
- Migration difficulty rating
- Code examples and patterns

Key Sections:
- 1. BloodManagement (VERY HIGH complexity, do last)
- 2. AdaptiveMarketPlanning (VERY LOW complexity, do first)
- 3. MedicalDecisionDiabetes (MODERATE complexity)
- 4. StochasticShortestPath_Dynamic (MODERATE-HIGH complexity)
- 5. StochasticShortestPath_Static (MODERATE complexity)

Plus:
- Comparison matrix of all 5 problems
- Migration priority tier ranking
- Key migration insights from AdaptiveMarketPlanning
- Modernization directory status
- Complete action plan

**Use this for**: Deep understanding of each problem, detailed complexity analysis, algorithm specifics

---

### 2. MIGRATION_QUICK_REFERENCE.md (9.8 KB)

**Quick reference guide for migration planning**

Key Sections:
- TL;DR recommendation
- Problems at a glance (size, complexity, time estimates)
- Why start with AdaptiveMarketPlanning
- Why wait on others
- What you'll learn from each problem
- Step-by-step migration plan for AdaptiveMarketPlanning
- Success criteria and timelines
- Code patterns and templates
- Next steps after first migration

**Use this for**: Quick lookups, timeline planning, understanding rationale, getting started

---

## The Recommendation

### Start Here: AdaptiveMarketPlanning

**Why:**
- Smallest codebase (543 lines)
- Simplest state (2 dimensions)
- Single decision variable
- No external dependencies (pure math)
- Can complete in 1-2 days
- Perfect template for other problems
- Minimal risk, maximum learning

**Timeline**: 1-2 days including full testing and documentation

### Then Proceed In Order:
1. **Tier 2** (Days 3-8): MedicalDecisionDiabetes - moderate complexity, Bayesian learning
2. **Tier 3** (Days 8-13): StochasticShortestPath_Static - graph-based, Bellman iteration
3. **Tier 4** (Days 13-19): StochasticShortestPath_Dynamic - lookahead policies
4. **Tier 5** (Days 19+): BloodManagement - maximum complexity, do last

---

## Quick Size Comparison

| Problem | LOC | State Dims | Decision Vars | Complexity | Time |
|---------|-----|-----------|---------------|-----------|------|
| AdaptiveMarketPlanning | 543 | 2 | 1 | ⭐ | 1-2 days |
| MedicalDecisionDiabetes | 721 | 15 | 5 | ⭐⭐ | 3-4 days |
| SSP_Static | 683 | 1-10 | 1 | ⭐⭐ | 4-5 days |
| SSP_Dynamic | 598 | 1 | 1 | ⭐⭐⭐ | 5-6 days |
| BloodManagement | 2,167 | 160-200+ | 160+ | ⭐⭐⭐⭐⭐ | 10-14 days |

---

## How to Use These Documents

### If You Have 5 Minutes
Read the TL;DR section of MIGRATION_QUICK_REFERENCE.md

### If You Have 15 Minutes
Read MIGRATION_QUICK_REFERENCE.md completely (323 lines, quick read)

### If You Have 1 Hour
Read MIGRATION_QUICK_REFERENCE.md + skim LEGACY_PROBLEM_ANALYSIS.md comparison matrix

### If You Have 2+ Hours
Read both documents completely for comprehensive understanding

---

## Key Findings Summary

### Complexity Ranking (Easiest to Hardest)
1. **AdaptiveMarketPlanning** - Pure gradient descent, 2D state, no dependencies
2. **MedicalDecisionDiabetes** - Bayesian learning, 15D state, 5 policy variants
3. **SSP_Static** - Bellman iteration, graph structure, value approximation
4. **SSP_Dynamic** - Lookahead policies, graph structures, horizon planning
5. **BloodManagement** - LP integration, 160+D state, advanced algorithms

### Time-to-Value Ranking (Best First)
1. **AdaptiveMarketPlanning** - 1-2 days, quick win, great template
2. **MedicalDecisionDiabetes** - 3-4 days, good follow-up
3. **SSP_Static** - 4-5 days, graph patterns
4. **SSP_Dynamic** - 5-6 days, more complex graphs
5. **BloodManagement** - 10-14 days, major project, do last

### Learning Value
1. **AdaptiveMarketPlanning** - JAX basics, policy patterns, testing
2. **MedicalDecisionDiabetes** - Policy vectorization, Bayesian updates
3. **SSP_Static** - Graph structures, Bellman iteration
4. **SSP_Dynamic** - Lookahead, horizon-based planning
5. **BloodManagement** - LP integration, multi-dimensional optimization

---

## Migration Path Overview

```
Day 1-2:     AdaptiveMarketPlanning  (1-2 days, 543 LOC)
             ↓
Day 3-8:     MedicalDecisionDiabetes (3-4 days, 721 LOC)
             ↓
Day 8-13:    SSP_Static             (4-5 days, 683 LOC)
             ↓
Day 13-19:   SSP_Dynamic            (5-6 days, 598 LOC)
             ↓
Day 19+:     BloodManagement        (10-14 days, 2,167 LOC)

Total: ~5-6 weeks to migrate all 5 problems (4,312 total LOC)
```

---

## Related Resources

### In Modernization/ Directory
- **FINAL_SUMMARY.md** - Executive overview of JAX modernization
- **JAX_MODERNIZATION_PLAN.md** - Complete 13-week roadmap (30 KB)
- **NUMPY_VS_JAX_COMPARISON.md** - Performance and pattern comparison
- **QUICK_START.md** - Environment setup and workflow
- **base_protocols_jax.py** - Type-safe JAX protocol templates
- **energy_storage_model_jax.py** - Complete JAX example (16 KB)

### Already Migrated
- **stochopt/problems/energy_storage/** - JAX-native example
- **stochopt/problems/asset_selling/** - JAX-native example
- **stochopt/problems/clinical_trials/** - JAX-native example (partial)
- **stochopt/problems/two_newsvendor/** - JAX-native example

---

## Next Steps

1. **Read MIGRATION_QUICK_REFERENCE.md** (5-10 min)
   - Understand the recommendation
   - Get timeline overview
   - See code patterns

2. **Skim LEGACY_PROBLEM_ANALYSIS.md** (10-15 min)
   - Review comparison matrix
   - Understand why AdaptiveMarketPlanning is first
   - Note key metrics for each problem

3. **Review Modernization/ directory** (15-20 min)
   - Study base_protocols_jax.py template
   - Check energy_storage_model_jax.py example
   - Read JAX_MODERNIZATION_PLAN.md overview

4. **Start Migration** (1-2 days)
   - Create stochopt/problems/adaptive_market_planning/
   - Follow Phase 1-5 plan in MIGRATION_QUICK_REFERENCE.md
   - Use templates and examples from Modernization/

---

## Questions?

- **For quick answers**: See MIGRATION_QUICK_REFERENCE.md
- **For technical details**: See LEGACY_PROBLEM_ANALYSIS.md
- **For code patterns**: See Modernization/base_protocols_jax.py
- **For complete plans**: See Modernization/JAX_MODERNIZATION_PLAN.md

---

## Summary

You have comprehensive analysis of all 5 legacy problems with clear recommendation:

**Start with AdaptiveMarketPlanning for 1-2 day quick win that sets up the entire migration sequence.**

Then proceed through the tiers based on increasing complexity and interdependencies.

Good luck with the modernization!

---

**Report Generated**: November 14, 2025
**Analysis Scope**: 5 legacy problems, 4,312 total lines of code
**Primary Recommendation**: AdaptiveMarketPlanning migration first
**Estimated Total Timeline**: 5-6 weeks for all 5 problems
