#!/bin/bash

# log system
LOG_LEVEL="INFO"
log()
{
    local LEVEL=$1
    shift
    local MESSAGE="$@"
    local TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    if [[ "$LEVEL" == "ERROR" ]] || [[ "$LEVEL" == "INFO" && "$LOG_LEVEL" == "INFO" ]] || \
    [[ "$LEVEL" == "DEBUG" && "$LOG_LEVEL" == "DEBUG" ]]; then
        echo "[$TIMESTAMP] [$LEVEL] : $MESSAGE"
    fi
}

# error check
if [ $# -ne 1 ]; then
    log "INFO" "usage: ./findlib.sh <libname>"
    exit 1
fi

if [ ! -e $1 ]; then
    log "ERROR" "$1 does not exsist"
    exit 1
fi

# wrap all dynamic library
LIBS=$(ldd $1 | grep "=> /" | awk '{print $3}')
DEST="./lib"

mkdir -p $DEST
for LIB in $LIBS; do
    cp $LIB $DEST
done
log "INFO" "wrap up completed!"
