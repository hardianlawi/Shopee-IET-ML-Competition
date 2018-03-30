#!/bin/bash

if [ -z "$1" ]
then
    echo "usage: ./clean.sh [train_folder]"
    exit 1
fi

find $1 -name *jpg | xargs file | grep -v JPEG | grep -o '^[^:]\+' | xargs rm
