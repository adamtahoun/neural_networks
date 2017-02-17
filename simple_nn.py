import numpy as np

# sigmoid function to calculate propability
def sigmoid(x, derivative = False):
    if(derivative == True):
        return (x * (1-x))
    return 1 / (1 + np.exp(-x))

#inputs
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

# seed for error analysis
np.random.seed(1)

#weights
theta0 = 2 * np.random.random((2,4)) - 1
theta1 = 2 * np.random.random((4,1)) - 1

for i in xrange(60000):

    # forwardpropogation
    l0 = X
    l1 = sigmoid(np.dot(l0,theta0))
    l2 = sigmoid(np.dot(l1,theta1))

    # backpropogation
    l2_error = y - l2
    if(i % 10000 == 0):
        print "Error: " + str(np.mean(np.abs(l2_error)))
        print (str(l2))

    # calculate deltas
    l2_delta = l2_error * sigmoid(l2, derivative = True)
    l1_error = l2_delta.dot(theta1.T)
    l1_delta = l1_error * sigmoid(l1, derivative = True)

    # update weights
    theta1 += l1.T.dot(l2_delta)
    theta0 += l0.T.dot(l1_delta)

print ("output after training")
print (str(l2))
