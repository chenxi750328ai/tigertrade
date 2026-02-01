PR title: feat: add start/end time and paging support to get_kline_data

Summary
-------
This change adds optional `start_time` and `end_time` parameters to `get_kline_data` and implements best-effort automatic paging via `QuoteClient.get_future_bars_by_page` for single-symbol historical/time-range requests. The function still supports the previous `quote_client.get_future_bars` path for normal, small recent requests.

Files changed
-------------
- tigertrade/tiger1.py
  - `get_kline_data(..., start_time=None, end_time=None)`
  - Adds a helper to convert datetimes to epoch ms and uses the by-page API when time-range or large counts are requested.
  - Normalizes page results (DataFrame or iterable of bars), concatenates pages, timezone normalizes to Asia/Shanghai, sorts and truncates to requested `count`.

- tests/test_kline_data.py
  - Adds:
    - `test_get_kline_data_with_paging` — validates multi-page concatenation and tz conversion
    - `test_get_kline_data_time_range_ms_arg` — validates integer ms `start_time`/`end_time` are passed through and used

Testing
-------
Run locally:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_kline_data.py -q

All tests pass locally (6 passed, some pandas FutureWarnings about freq alias). See `tests/test_kline_data.py` for mocking details.

Notes / Caveats
--------------
- Paging is implemented best-effort and currently only used when a single symbol is requested (the by-page API is per-identifier).
- The by-page API behavior varies between SDK versions; this implementation handles several response shapes (tuple (df, token), dict, DataFrame, iterable of bar objects).
- Added structured logging (module `logging`) and improved docstrings for `get_kline_data` and `place_tiger_order` to clarify parameters, return values and the `ALLOW_REAL_TRADING` safety guard.

Suggested branch & commit message
---------------------------------
Branch: feature/kline-paging
Commit message: feat(kline): add start/end time and paging support to get_kline_data; add tests

How to apply
------------
Option A (preferred, if you have an existing git repo):

1. Save the script `kline-paging-apply.sh` in project root and make it executable:

    chmod +x kline-paging-apply.sh
    ./kline-paging-apply.sh

2. Review changes, run tests, and commit:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_kline_data.py -q
    git checkout -b feature/kline-paging
    git add tigertrade/tiger1.py tests/test_kline_data.py
    git commit -m "feat(kline): add start/end time and paging support to get_kline_data; add tests"
    git push -u origin feature/kline-paging

3. Open a PR in GitHub with the above title and description.

Option B (manual): copy the updated files from `tigertrade/tiger1.py` and `tests/test_kline_data.py` and replace your local files; run tests as above.

If you want, I can also produce a minimal patch/diff file instead of the script—tell me which you prefer.
