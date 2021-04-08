#!/bin/bash

###
# convert a directory containing audio files to .wav format (pcm_s16le)
# 
# synopsis:
#  $ bash scripts/convert.sh <path> <mp3|mp4|ogg> <num_threads>
# 
# example usage:
#  $ bash scripts/convert.sh /data/common-voice/de/clips mp3 8
###

# first, make sure ffmpeg is available
if ! command -v ffmpeg &> /dev/null
then
	echo "ERROR: ffmpeg could not be found"
	exit
fi

# run conversion
OIFS="$IFS"
IFS=$'\n'

FORMAT=.${2}
DATA_DIR="$1"
N=${3:-1}

mkdir -p $DATA_DIR

FILES=$(ls "$DATA_DIR" | grep $FORMAT)

thread () {
    local FILE_N=$1
    FILENAME="${FILE_N:0:${#FILE_N}-4}"
    ffmpeg -y -i $DATA_DIR/$FILE_N -acodec pcm_s16le -ac 1 -ar 16000 $DATA_DIR/$FILENAME.wav
    rm $DATA_DIR/$FILE_N
}
for FILE in $FILES
do
   ((i=i%N)); ((i++==0)) && wait
   thread "$FILE" & 
done

IFS="$OIFS"
