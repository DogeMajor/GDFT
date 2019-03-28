#!/bin/bash

echo "Testing all the modules"

python tests/testGDFT.py TestGDFT
python tests/testCorrelation.py TestCorrelation
python tests/testCorrelationAnalyzer.py TestCorrelationAnalyzer
python tests/testDAO.py TestThetasDAO TestSortedThetasDAO TestThetaGroupsDAO
python tests/testSequenceFinder.py TestSequenceFinder
python tests/testOptimizer.py TestOptimizer
python tests/testThetasAnalyzer.py TestThetasAnalyzer
python tests/testClassifier.py TestClassifier
python tests/testPCA.py TestPCA

#python -m unittest discover -s tests