#!/bin/bash
BASEDIR=data

pip install gdown

# https://github.com/luogen1996/RepAdapter
# https://drive.google.com/file/d/1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p/view
FILE_ID="1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p"
FILE_NAME="vtab-1k.zip"
gdown --id $FILE_ID -O $FILE_NAME

mkdir -p ${BASEDIR}/vtab-1k
mv $FILE_NAME ${BASEDIR}/vtab-1k
cd ${BASEDIR}/vtab-1k
unzip -q $FILE_NAME
