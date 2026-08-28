#!/bin/bash
# SIGKILL the dry-run python (crash-resume test). Exact pattern; this
# script cmdline does not contain the pattern.
pkill -9 -f "ace.train.grpo"; sleep 2; pgrep -f "ace.train.grpo" && echo STILL_ALIVE || echo KILLED
