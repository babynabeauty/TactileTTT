#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-outputs/libero_eval_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="${OUT_DIR}/libero_eval_${TIMESTAMP}.txt"

TASK_SUITES=(
  libero_spatial
  libero_object
  libero_goal
  libero_10
)

mkdir -p "${OUT_DIR}"
overall_status=0

{
  echo "LIBERO evaluation started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Python: ${PYTHON_BIN}"
  echo "Output directory: ${OUT_DIR}"
  echo "Task suites: ${TASK_SUITES[*]}"
  echo
} | tee "${SUMMARY_FILE}"

for suite in "${TASK_SUITES[@]}"; do
  LOG_FILE="${OUT_DIR}/${suite}_${TIMESTAMP}.txt"

  {
    echo "============================================================"
    echo "Running ${suite}"
    echo "Log file: ${LOG_FILE}"
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Command: ${PYTHON_BIN} libero_test.py --args.task-suite-name ${suite} $*"
    echo "============================================================"
  } | tee -a "${SUMMARY_FILE}" | tee "${LOG_FILE}"

  "${PYTHON_BIN}" libero_test.py --args.task-suite-name "${suite}" "$@" 2>&1 | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}

  {
    echo
    echo "Finished ${suite} at $(date '+%Y-%m-%d %H:%M:%S') with exit code ${status}"
    echo
  } | tee -a "${SUMMARY_FILE}" | tee -a "${LOG_FILE}"

  if [[ "${status}" -ne 0 ]]; then
    overall_status="${status}"
    echo "${suite} failed. Continuing with the next suite. See ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  fi
done

echo "All LIBERO evaluations finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"
exit "${overall_status}"
