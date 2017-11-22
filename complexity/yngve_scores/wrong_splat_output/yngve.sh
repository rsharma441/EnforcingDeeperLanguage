#!/bin/bash
while read p; do
  splat yngve "$p"
done <wronganswer.txt > wrongout.txt