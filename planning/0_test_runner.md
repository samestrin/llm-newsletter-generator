# Test Runner Instructions (Optimized)

**CRITICAL: Poetry Usage Enforcement**
- ALL tests MUST use Poetry: `poetry run pytest` (NEVER `pytest` or `python -m pytest`)
- ALL dependency management MUST use Poetry: `poetry add`, `poetry install`
- If Poetry fails, stop immediately and report the issue

---

## Core Mission

Autonomous test runner for Python CLI application. Complete each test suite following the retry strategy. Create review documents for persistent failures.

### Key Constraints
- **LLM Tests**: Attempt but don't get stuck - note instability and skip if needed
- **API Keys**: Load from `.env` or tests will fail
- **Loop Prevention**: Track iterations clearly - if stuck, document and move on
- **Individual Mode**: Auto-trigger after 5+ suite failures
- **Warning Logging**: Log ALL warnings to `/planning/pending/99_test_warnings.md`

---

## Execution Strategy

### Retry Logic (Core Algorithm)
For each unchecked test suite:

1. **Suite Attempt 1-3**: Run full suite with `poetry run pytest tests/[test_file].py`
2. **Individual Mode Trigger**: If >5 errors OR 3 failed attempts
3. **Individual Attempts**: Run each test separately, 3 attempts each
4. **Final Resolution**: Mark complete OR create review document

### Poetry Commands (Required)
```bash
# Basic execution
poetry run pytest tests/test_file.py          # Full suite
poetry run pytest tests/test_file.py::TestClass::test_method  # Individual

# Debugging
poetry run pytest -v --tb=long               # Verbose output
poetry run pytest --lf                       # Last failed only
```

### Iteration Tracking (Anti-Loop)
```
[ATTEMPT 1/3] Running full suite: test_file.py
[ATTEMPT 2/3] Fixing issue: <specific problem>  
[ATTEMPT 3/3] Final suite attempt
[INDIVIDUAL MODE] Processing test by test
[WARNING CHECK] Logging any warnings to 99_test_warnings.md
[MOVING ON] Creating review document: ##.#_review_and_correct_test_file.md
```

### Warning Management
- **Log ALL warnings** encountered during test execution to `/planning/pending/99_test_warnings.md`
- **Use standardized format** with test file, priority, and resolution tracking
- **Preserve existing warnings** - never overwrite, only append new ones
- **High priority warnings** may require immediate attention before continuing

---

## Key Documentation References

**Architecture & Design:**
- `/README.md` - Overview


**Implementation Status:**
- `/planning/done/` - Completed tasks
- `/planning/pending/` - Pending tasks and reviews

---

## Review Document Template

When creating `/planning/pending/##.#_review_and_correct_<test_name>.md` (_If this file already exists, update it._):

```markdown
# Review and Correct: [Test Name]

## Test Failures Summary
- **Suite**: [test_file.py]  
- **Failed Tests**: [count] of [total]
- **Attempts Made**: [suite attempts] + [individual attempts]

## Detailed Analysis
### [Test Name 1]
- **Error**: [specific error message]
- **Root Cause**: [analysis]  
- **Suggested Fix**: [specific action]

## Recommended Actions
1. [Priority 1 action]
2. [Priority 2 action]

## Next Steps
- [ ] Human review required
- [ ] Implementation needed
- [ ] Re-run after fixes
```

---

## Current Test Suite Status

### Phase 1: Foundation & Testing Infrastructure ✅
1. [x] `test_newsletter_generator.py` - Basic instantiation (EXISTING) - ⚠️ 11.6s execution time
2. [ ] `test_feed_processing.py` - Feed loading and parsing
3. [ ] `test_content_generation.py` - Text generation and caching
4. [ ] `test_template_system.py` - Template loading and processing
5. [ ] `test_model_management.py` - Model configuration and loading
6. [ ] `test_cache_manager.py` - Cache operations and validation
7. [ ] `test_config_manager.py` - Configuration management
8. [ ] `test_error_handling.py` - Exception handling and validation

### Phase 2: Enhanced Metadata & Content Processing
9. [ ] `test_metadata_extraction.py` - RSS field extraction
10. [ ] `test_content_analysis.py` - Sentiment and topic analysis
11. [ ] `test_advanced_templates.py` - Dynamic template system
12. [ ] `test_content_quality.py` - Quality scoring and validation
13. [ ] `test_duplicate_detection.py` - Content similarity and deduplication

### Phase 3: Multi-Model Architecture
14. [ ] `test_model_providers.py` - Local and API model providers
15. [ ] `test_two_tier_processing.py` - Fast and quality model tiers
16. [ ] `test_api_integrations.py` - OpenAI, Gemini, Claude integrations
17. [ ] `test_model_fallback.py` - Fallback mechanisms
18. [ ] `test_cost_management.py` - API usage and cost tracking

### Phase 4: Advanced Generation Pipeline
19. [ ] `test_multi_stage_generation.py` - Theme extraction and planning
20. [ ] `test_quality_assurance.py` - Content validation and scoring
21. [ ] `test_content_enhancement.py` - Related articles and linking
22. [ ] `test_consistency_checking.py` - Cross-section coherence

### Phase 5: User Experience & Interface
23. [ ] `test_configuration_interface.py` - Interactive configuration
24. [ ] `test_output_formats.py` - HTML, PDF, email formatting
25. [ ] `test_preview_editing.py` - Content preview and editing
26. [ ] `test_user_workflows.py` - End-to-end user scenarios

### Phase 6: Performance & Scalability
27. [ ] `test_caching_system.py` - Redis and distributed caching
28. [ ] `test_parallel_processing.py` - Async operations and concurrency
29. [ ] `test_performance_monitoring.py` - Metrics and analytics
30. [ ] `test_scalability.py` - Large dataset processing

### Phase 7: Security & Production
31. [ ] `test_security_features.py` - Encryption and validation
32. [ ] `test_production_readiness.py` - Health checks and deployment
33. [ ] `test_compliance.py` - Privacy and audit features
34. [ ] `test_integration_end_to_end.py` - Full system integration


---

## Execution Instructions

1. **Process each unchecked test** following retry logic
2. **Track iterations clearly** to prevent loops  
3. **Use Poetry exclusively** for all test execution
4. **Log warnings immediately** to `/planning/pending/99_test_warnings.md` using the standardized format
5. **Create review documents** for persistent failures
6. **Mark completed tests** with [x] when passing
7. **Report final status** with summary of actions taken

**Warning Signs to Skip:**
- LLM instability or hallucination risks
- Endless retry loops (>3 suite + individual attempts)
- Missing API keys causing consistent failures
- Environmental issues requiring human intervention
- **Critical warnings** that require immediate human attention