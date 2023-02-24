import math

def count(left, right, data):
    sum = 0
    for i in data:
        if left <= i < right + 1:
            sum += 1
    return sum



