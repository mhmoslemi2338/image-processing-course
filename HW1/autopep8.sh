#!/bin/sh
python3 -m autopep8 *.py --recursive --aggressive --in-place --pep8-passes 60000 --verbose
