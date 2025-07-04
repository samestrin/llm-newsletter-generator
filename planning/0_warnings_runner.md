# Warning Resolver Instructions (Optimized)

**CRITICAL: Poetry Usage Enforcement**
- ALL warning analysis MUST use Poetry: `poetry run pytest` (NEVER `pytest` or `python -m pytest`)
- ALL dependency management MUST use Poetry: `poetry add`, `poetry install`
- If Poetry fails, stop immediately and report the issue

---

## Core Mission

Autonomous warning resolver for Python BookSort CLI application. Process warnings systematically following the retry strategy. Create resolution documents for persistent warnings requiring human intervention.

### Key Constraints
- **Warning Priorities**: Critical > High > Medium > Low processing order
- **API Keys**: Load from `.env` or warnings may be incomplete
- **Loop Prevention**: Track iterations clearly - if stuck, document and move on
- **Batch Processing**: Process all warnings from same test file together
- **Persistent Logging**: All warnings, updates, changes logged to `/planning/pending/99_test_warnings.md`

---

## Execution Strategy

### Warning Resolution Logic (Core Algorithm)
For each warning category (Critical → High → Medium → Low):

1. **Scan Phase**: Run `poetry run pytest -v` to collect all current warnings
2. **Categorization**: Auto-classify warnings by impact and urgency
3. **Resolution Attempt 1-3**: Apply automated fixes where possible
4. **Verification**: Re-run tests to confirm warning resolution
5. **Documentation**: Update status in `/planning/pending/99_test_warnings.md`
6. **Escalation**: Create review document for unresolvable warnings

### Poetry Commands (Required)
```bash
# Warning collection
poetry run pytest -v --tb=short -W error::DeprecationWarning  # Capture warnings as errors
poetry run pytest -v --tb=short -W ignore::pytest.PytestUnraisableExceptionWarning  # Suppress noise
poetry run pytest tests/test_file.py -v --tb=short           # File-specific warnings

# Resolution verification
poetry run pytest tests/test_file.py -v --tb=short           # Confirm fixes
poetry run pytest -v --tb=short                             # Full suite verification
```

### Iteration Tracking (Anti-Loop)
```
[SCAN 1/3] Collecting warnings from: test_file.py
[CLASSIFY] Categorizing warnings by priority
[RESOLVE 1/3] Applying automated fix: <specific solution>
[VERIFY] Re-running tests to confirm resolution
[LOG] Updating warning status in 99_test_warnings.md
[ESCALATE] Creating review document: ##.#_warning_resolution_test_file.md
```

### Warning Management Protocol
- **Process by priority**: Critical → High → Medium → Low
- **Batch by test file**: Handle all warnings from same file together
- **Preserve existing entries** - never overwrite, only update status
- **Auto-resolve common patterns**: Deprecation warnings, import warnings, etc.
- **Escalate complex issues** requiring human judgment

---

## Warning Classification Matrix

| Priority | Type | Auto-Resolve | Tolerance | Action |
|----------|------|--------------|-----------|---------|
| **Critical** | API auth, DB connection, filesystem | Yes | Zero | Automated fixes/Immediate escalation |
| **High** | Core deprecations, performance, rate limits | Partial | Low | Automated fixes |
| **Medium** | Minor deprecations, config issues | Yes | Medium | Automated fixes |
| **Low** | Cosmetic, dev-only warnings | Yes | High | Automated fixes |

---

## Warning Resolution Template

When creating `/planning/pending/##.#_warning_resolution_<test_name>.md`:

```markdown
# Warning Resolution: [Test Name]

## Warning Summary
- **Test File**: [test_file.py]
- **Total Warnings**: [count]
- **Critical**: [count] | **High**: [count] | **Medium**: [count] | **Low**: [count]
- **Auto-Resolved**: [count] | **Escalated**: [count]

## Resolution Actions Taken
### Auto-Resolved Warnings
- **[Priority]** [Warning Type]: [Brief description]
  - **Action**: [Specific fix applied]
  - **Verification**: [Test results]

### Escalated Warnings
- **[Priority]** [Warning Type]: [Brief description]
  - **Reason**: [Why auto-resolution failed]
  - **Recommendation**: [Suggested human action]

## Updated Warning Status
- [x] Processed all warnings
- [x] Updated 99_test_warnings.md
- [ ] Human review required for escalated items

## Next Steps
1. [Priority 1 action for human review]
2. [Priority 2 action for human review]
```

---

## Execution Instructions

1. **Scan for warnings** in priority order (Critical → Low)
2. **Apply automated fixes** where patterns match
3. **Track iterations clearly** to prevent loops
4. **Use Poetry exclusively** for all operations
5. **Update warning status** in real-time
6. **Create escalation documents** for complex issues
7. **Verify all changes** with test re-runs
8. **Report completion metrics** with summary

**Skip Conditions:**
- Warnings requiring external dependencies
- Configuration changes affecting prod systems
- Complex refactoring beyond scope
- Warnings from third-party libraries (document only)
- **Critical warnings** requiring immediate human attention

**Success Metrics:**
- Warning count reduction
- Auto-resolution percentage
- Time to resolution
- Escalation accuracy