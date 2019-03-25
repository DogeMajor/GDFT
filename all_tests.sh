#!/bin/bash

echo "Testing all the modules"

python test/testGDFT.py TestGDFT
python test/testCorrelation.py TestCorrelation
python test/testCorrelationAnalyzer.py TestCorrelationAnalyzer
python test/testDAO.py TestDAO
python test/testSequenceFinder.py TestSequenceFinder
python test/testOptimizer.py TestOptimizer
python test/testThetasAnalyzer.py TestThetasAnalyzer
python test/testClassifier.py TestClassifier
python test/testPCA.py TestPCA
