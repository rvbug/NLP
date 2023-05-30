# Calculate Average

import numpy as np

a = np.array([1,2,3,4])
print(a, len(a))
total = 0
for i in range(0, len(a)):
  total = total + a[i]

avg = total/len(a)
print(avg)


# Calculate weighted average

inputs = np.random.random(size=(3,))
weights = np.random.random(size=(3,))
print(inputs, weights)

total_inputs  = 0
total_wt = 0

for i in range(0, len(inputs)):
  total_wt = total_wt + weights[i]
  total_inputs = total_inputs + (weights[i] * inputs[0])

print("numerator -> ", total_inputs)
print("Weight total -> " ,total_wt)

avg_wt = total_inputs/total_wt
print("average weight -> ", avg_wt)
