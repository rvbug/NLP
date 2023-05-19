import numpy as np

### common functions
def print_dec(x):
  print(f"{x:.2f}")

#sigmoid
def fn_sigmoid(x):
  return 1.0/ (1.0 + np.exp(-x))


####### init phase
# k => number of lstm layer required

k = np.random.random((4))
bi = np.random.random(k.shape[0])

ct1 = np.random.random((4,1)) # to double check later
ht1 = np.random.random((3,1))
xt = np.random.random((4,1))

conct = np.concatenate((ht1, xt))
print(conct.shape)

print(wf.shape)
print(k.shape)
print(bi.shape)

# (7, 1)
# (4, 7)
# (4,)
# (4,)

####### Input Gates (it)

# 1> concat x and ht
# 2> multiply it with output of 1 & 2  

# wi => `weight in forget layer` = k * (shape[0] of x, shape[0] of ht)
wi = np.random.random((k.shape[0], conct.shape[0]))
print(wi.shape, conct.shape , bi.shape)
it = fn_sigmoid(np.dot(wi, conct) + bi)
print(it.shape)
print(it)

# (4, 7) (7, 1) (4,)
# (4, 4)
# [[0.92526928 0.93732615 0.9587864  0.93053626]
# [0.88615609 0.90386775 0.93600232 0.89386353]
# [0.70688378 0.74444295 0.81920987 0.7229335 ]
# [0.89850803 0.91448345 0.9432921  0.90546843]]


####### Input Gates (c_hat)
bc = np.random.random(k.shape[0])
print(conct.shape)
c_hat = np.random.random((k.shape[0], conct.shape[0]))
print(wf.shape)
print(k.shape)
print(bc.shape)
wc = np.random.random((k.shape[0], conct.shape[0]))

print(c_hat.shape, conct.shape , wc.shape, bc.shape)
c_hat = np.tanh(np.dot(wc, conct) + bc)
print(c_hat.shape)
print(c_hat)


##### output
# (7, 1)
# (4, 7)
# (4,)
# (4,)
# (4, 7) (7, 1) (4, 7) (4,)
# (4, 4)
#[[0.93977877 0.9776522  0.96581094 0.93424888]
# [0.88955214 0.95833487 0.93658629 0.87969617]
# [0.98084967 0.9929868  0.9892265  0.97905044]
# [0.90207254 0.96320985 0.94393309 0.89327074]]


####### Forget Gate (ft)
k = np.random.random((4))
wf = np.random.random((k.shape[0], conct.shape[0]))
bf = np.random.random(k.shape[0])
ft = fn_sigmoid(np.dot(wf, conct) + bf )

print(conct.shape)
print(wf.shape)
print(bf.shape)
print(ft.shape)
print(ft)

##### output
# (7, 1)
# (4, 7)
# (4,)
# (4, 4)
# [[0.89007057 0.92407829 0.90318102 0.8959198 ]
# [0.92480399 0.94868606 0.93407884 0.92895261]
# [0.90448884 0.93436505 0.91604193 0.90964876]
# [0.90886656 0.93746808 0.91993699 0.91381252]]


####### Update Cell Gate (ft)
ft.shape, it.shape, c_hat.shape
# ((4, 4), (4, 4), (4, 4))
ct1 = np.random.random((4,1))
ct = np.multiply(ft, ct1) + np.multiply(it , c_hat)
print(ct)

##### output
#[[1.36235807 1.42801784 1.42607496 1.36540067]
# [1.64638873 1.74647436 1.74335957 1.64828443]
# [1.24911259 1.31334541 1.37324881 1.26672475]
# [1.21163415 1.29457705 1.29640511 1.212126  ]]


####### Output Gate (ot)
wo = np.random.random((k.shape[0], conct.shape[0]))
bo = np.random.random(k.shape[0])

ot = fn_sigmoid(np.dot(wo, conct) + bo)
print(ot)

##### output
#[[0.89520767 0.89839338 0.78617099 0.83451351]
# [0.92652805 0.9288373  0.84441648 0.88157444]
# [0.95218332 0.95372649 0.89551033 0.92159855]
# [0.90345344 0.90641469 0.80109015 0.84671737]]

####### Final Gate (ot)
ht = np.multiply(ot, np.tanh(ct))
print(ht)

# [[0.78504239 0.80070179 0.70036783 0.73240224]
# [0.86015195 0.87401077 0.79427083 0.81864927]
# [0.80748441 0.82508734 0.78753909 0.7860383 ]
# [0.75634322 0.7798072  0.68957427 0.70897007]]


print("next cell state ct \n", ct)
print("next hidden state ht\n", ht)

# next cell state ct 
# [[1.36235807 1.42801784 1.42607496 1.36540067]
# [1.64638873 1.74647436 1.74335957 1.64828443]
# [1.24911259 1.31334541 1.37324881 1.26672475]
# [1.21163415 1.29457705 1.29640511 1.212126  ]]
# next hidden state ht
# [[0.78504239 0.80070179 0.70036783 0.73240224]
# [0.86015195 0.87401077 0.79427083 0.81864927]
# [0.80748441 0.82508734 0.78753909 0.7860383 ]
# [0.75634322 0.7798072  0.68957427 0.70897007]]
