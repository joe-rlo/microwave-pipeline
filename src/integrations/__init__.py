"""Third-party API clients used by tools.

Each module is a thin, pure-HTTP adapter around an external service.
Knows nothing about the Agent SDK or the pipeline — those wrappers
live in `src/tools/`. Keeping the split lets us unit-test the API
shape without dragging the SDK into pytest.
"""
