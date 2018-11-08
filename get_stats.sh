#!/bin/bash
cat Pos_Tagger.py | grep -A 9 Hyperparameters
cat build.logs | grep Time
cat run.logs | grep Time
cat eval.logs | grep Accuracy