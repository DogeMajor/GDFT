#!/bin/bash

echo "Testing all the modules"

python test/testGDFT.py
python test/testCorrelations.py
python test/testCorrelationAnalyzer.py
python test/testDAO.py
python test/testSequenceFinder.py
python test/testOptimizer.py
python test/testThetasAnalyzer.py TestThetasAnalyzer
python test/testClassifier.py TestClassifier
python test/testPCA.py TestPCA
