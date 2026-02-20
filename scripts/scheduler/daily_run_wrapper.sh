#!/bin/bash

echo "git pull to latest."
git pull

./scripts/scheduler/daily_run_v7x.sh
