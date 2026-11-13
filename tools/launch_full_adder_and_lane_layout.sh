#!/usr/bin/env bash
# Launch the full-adder truth table (docs/public-full-adder-truth-table.md, 4 jobs) and the lane-layout check
# (docs/public-lane-layout-check.md, 16 jobs).
#
# Usage: launch_full_adder_and_lane_layout.sh MODE SOURCE_CHECKOUT FROZEN_COMMIT RUN_ROOT [JOB_SET]
#   MODE             preflight (no model load) or execute
#   SOURCE_CHECKOUT  an isolated checkout of the frozen commit (git worktree); must be clean and at FROZEN_COMMIT
#   RUN_ROOT         a new directory; an existing one is refused (preflight and execution use separate roots)
#   JOB_SET          adder-and-lanes (default, 20 jobs) or single-lane (the lane check's single-lane arm, 8 jobs)
#
# The work is done by tools/launch_controller.py (standard library only), which replaced the xargs queue after
# Codex's Task 8 recheck (R1: memory admission decided by one controller that counts budgets not yet resident;
# R2: jobs in their own sessions, signals sent only to those, a record for every job and the summary even when
# stopped). Its docstring lists the environment settings. exec keeps this process's PID for the controller, so a
# signal sent to the launcher reaches the controller directly.
exec "${CONTROLLER_PY:-/usr/bin/python3}" -I "$(dirname "$(readlink -f "$0")")/launch_controller.py" "$@"
