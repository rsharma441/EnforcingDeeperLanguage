#!/bin/bash
while read p; do
  splat frazier "$p"
done <rightanswer.txt > rightout.txt