#!/bin/bash

git config --global user.email "miro@cs.washington.edu"
git config --global user.name "Miro Enev"

git add -A

if [ "$1" != "" ]; then
    git commit -a -m "$1"
else
    git commit -a -m "automated checkpoint"
fi

git push
