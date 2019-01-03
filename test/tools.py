
def EqualMatrices(matA, matB):
    are_equal = True
    if matA.shape != matB.shape:
        return False
    for row in range(matA.shape[0]):
        for col in range(matA.shape[1]):
            are_equal = are_equal and bool(matA[row, col] == matB[row, col])
    return are_equal

def AlmostEqualMatrices(matA, matB, decimals=7):
    are_equal = True
    if matA.shape != matB.shape:
        return False
    for row in range(matA.shape[0]):
        for col in range(matA.shape[1]):
            are_equal = are_equal and bool(round(matA[row, col], decimals) == round(matB[row, col], decimals))
    return are_equal
