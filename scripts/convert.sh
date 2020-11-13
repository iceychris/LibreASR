#!/bin/bash

###
# converts a directory containing .wav
# files to .mp3 files
###

OIFS="$IFS"
IFS=$'\n'

FORMAT=.mp3
DATA_DIR="$1"
N=${2:-1}

mkdir -p $DATA_DIR

FILES=$(ls "$DATA_DIR" | grep $FORMAT)

thread () {
    local FILE_N=$1
    FILENAME="${FILE_N:0:${#FILE_N}-4}"
    ffmpeg -i $DATA_DIR/$FILE_N -acodec pcm_s16le -ac 1 -ar 16000 $DATA_DIR/$FILENAME.wav
    rm $DATA_DIR/$FILE_N
}
for FILE in $FILES
do
   ((i=i%N)); ((i++==0)) && wait
   thread "$FILE" & 
done

IFS="$OIFS"
