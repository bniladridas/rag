# Issue #124: Organize Tests with Pytest Markers for Better CI Execution

## Problem
Tests are not properly categorized, making it difficult to:
- Run unit tests separately from integration tests
- Get accurate test coverage reports
- Optimize CI pipeline execution
- Maintain clear test boundaries

Currently all tests run together, which is inefficient for CI/CD pipelines.

## Requirements
- Unit tests should run fast and not require external dependencies
- Integration tests should test end-to-end functionality
- CI should be able to run test types separately
- Clear test categorization for maintenance

## Solution Implemented
1. Added pytest markers to all test files:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests

2. Updated CI workflow to run tests by marker:
   - Unit tests job: `pytest -m unit`
   - Integration tests job: `pytest -m integration`

3. Verified test distribution:
   - 36 unit tests across 5 files
   - 19 integration tests in e2e file

## Files Changed
- `tests/test_cli.py` - added `pytestmark = pytest.mark.unit`
- `tests/test_config.py` - added `pytestmark = pytest.mark.unit`
- `tests/test_data_fetcher.py` - added `pytestmark = pytest.mark.unit`
- `tests/test_rag_engine.py` - added `pytestmark = pytest.mark.unit`
- `tests/test_tools.py` - added `pytestmark = pytest.mark.unit`
- `tests/integration/test_e2e.py` - added `pytestmark = pytest.mark.integration`
- `.github/workflows/ci.yml` - updated test execution with markers

## Verification
```bash
pytest -m unit  # 36 tests pass
pytest -m integration  # 19 tests pass
pytest  # 55 total tests pass
```

**Status: ✅ Resolved**
