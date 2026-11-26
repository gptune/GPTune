#!/bin/bash
# Submits the staged jobs of stage_premium.queue (one "ENV=... sbatch-arguments" line per job, run from
# examples/GPTune-Demo) to the GPU premium QOS, one at a time, whenever the user has fewer than LIMIT jobs
# there (QOS MaxSubmitPU, pending and running). Submitted lines move to stage_premium.done with their job id.
# Run: nohup setsid bash stage_premium.sh > stage_premium.log 2>&1 &
HERE=$(cd $(dirname $0) && pwd)
DEMO_DIR=$(dirname $HERE)
QUEUE=$HERE/stage_premium.queue
DONE=$HERE/stage_premium.done
LIMIT=${LIMIT:-5}
cd $DEMO_DIR
while [ -s $QUEUE ]; do
    used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -c premium)
    if [ "$used" -lt "$LIMIT" ]; then
        line=$(head -n 1 $QUEUE)
        out=$(eval "$line" 2>&1)  # (the scripts request the premium QOS)
        if [[ $out =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            echo "$(date '+%F %T') submitted ${BASH_REMATCH[1]}: $line"
            echo "${BASH_REMATCH[1]} $line" >> $DONE
            sed -i 1d $QUEUE
            continue
        fi
        echo "$(date '+%F %T') submission failed ($used premium jobs): $out"
    fi
    sleep 60
done
echo "$(date '+%F %T') queue empty"
