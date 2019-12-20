#!/bin/bash
logfile=$1

while true; do
        sleep 9m
	line=$(tail -n 1  $logfile) 
        echo "$line" | grep "this is finished"
        if [ $? = 0 ]
        then
            exit 1
        else
            #echo "monitoring $logfile"
            echo "${line: -1}" 
        fi
done

