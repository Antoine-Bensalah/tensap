#!/usr/bin/env bash
# Run pytest with the given arguments and turn the failed and erroneous tests
# into GitHub annotations: they are then readable on the summary of the run and
# through the public API (the logs need a signed-in account).
set -o pipefail
pytest -rfE "$@" 2>&1 | tee pytest.log
status=${PIPESTATUS[0]}
grep -E "^(FAILED|ERROR) " pytest.log | head -30 | sed 's/^/::error title=pytest::/'
if [ "$status" -ne 0 ] && ! grep -qE "^(FAILED|ERROR) " pytest.log; then
  # collection or internal error: report the end of the log
  tail -15 pytest.log | sed 's/^/::error title=pytest (end of log)::/'
fi
exit "$status"
