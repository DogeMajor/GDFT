#!/bin/bash

echo "Testing all the modules"

python test/testGDFT.py
python test/testCorrelations.py
python test/testCorrelationCalculator.py
python test/testOptimizer.py
python test/testPolynomeOptimizer.py