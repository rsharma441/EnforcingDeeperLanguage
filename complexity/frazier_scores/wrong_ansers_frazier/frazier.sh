#!/bin/bash
while read p; do
  splat frazier "$p"
done <wronganswer.txt > wrongout.txt
