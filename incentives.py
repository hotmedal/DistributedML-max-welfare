import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Randomly select 100 training samples and 10 test samples for each agent
train_indices = np.random.permutation(len(x_train))
test_indices = np.random.permutation(len(x_test))

x_train_splits = [x_train[train_indices[i:i+100]] for i in range(0, len(x_train), 100)]
y_train_splits = [y_train[train_indices[i:i+100]] for i in range(0, len(y_train), 100)]

x_test_splits = [x_test[test_indices[i:i+10]] for i in range(0, len(x_test), 10)]
y_test_splits = [y_test[test_indices[i:i+10]] for i in range(0, len(y_test), 10)]


# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Hyperparameters
num_epochs = 50     # Communication rounds
num_agents = 10

# compute gradients on a single device
def compute_gradients(x_batch, y_batch, model):
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

# Initialize the model and optimizer
model = create_model()
optimizer = tf.keras.optimizers.legacy.Adam()

# Vanilla FedAVG
# Train the model
#for epoch in range(num_epochs):
#    total_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
#    for i in range(num_agents):
#        x_batch, y_batch = x_train_splits[i], y_train_splits[i]
#        gradients = compute_gradients(x_batch, y_batch, model)

#        for layer, grads in enumerate(gradients):
#            total_gradients[layer] += grads / num_agents  # Accumulate gradients

#    optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))      # at the server, end of round

#for i in range(num_agents):
#    pred = model.predict(x_test_splits[i], verbose=0)
#    pred_labels = np.argmax(pred, axis=1)

#    acc = np.mean(pred_labels == y_test_splits[i])
#    print("Device ", i, " accuracy: ", acc)




H = 50     # Contribution updating rounds
s = np.random.randint(1, 10, size=num_agents) * 10
rand_cost = np.random.randint(1, 10, size=num_agents) * 0.005
print("Starting S:  ", s)
tau = 100
ds = 10
n = num_agents


def avg_acc(i, x_train, y_train, x_test, y_test):
    M = create_model()
    M.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])     # from_logits=True  ?

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)      # shuffle data

    M.fit(x_train[indices[:i]], y_train[indices[:i]], epochs=5, validation_data=(x_test, y_test), verbose=0)    # train on only i samples
    test_loss, test_acc = M.evaluate(x_test, y_test, verbose=0)                     # should ideally do it multiple times for average
    return test_acc

def cost(s):
    # TODO: different cost for every device
    if s != 0:
        return 0.005*s       # add small value so cost is never zero
    else:
        return 0.00001

def welfare(p, u):
    return np.sum(u**p)**(1/p)

def relu(x):
    if x>=0:
        return x
    else:
        return 0

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def convex_func(x):
    return np.log(x + 1) ** 0.25

def non_decreasing(arr):
    nd_arr = np.copy(arr)
    for i in range(1, len(nd_arr)):
        if nd_arr[i] < nd_arr[i-1]:
            nd_arr[i] = nd_arr[i-1]
    return nd_arr


def interpolate_non_monotonic(arr, epsilon=0):
    interpolated_arr = np.copy(arr)
    for i in range(1, len(interpolated_arr)):
        if interpolated_arr[i] - interpolated_arr[i-1] <= epsilon:
            # Find the next element greater than current by epsilon
            next_greater_index = i + 1
            while next_greater_index < len(interpolated_arr) and \
                    interpolated_arr[next_greater_index] - interpolated_arr[i-1] <= epsilon:
                next_greater_index += 1
            
            # Interpolate between previous and next greater element
            prev_value = interpolated_arr[i-1]
            next_value = interpolated_arr[next_greater_index] if next_greater_index < len(interpolated_arr) else interpolated_arr[-1]
            for j in range(i, next_greater_index):
                interpolated_arr[j] = prev_value + (next_value - prev_value) * (j - (i-1)) / (next_greater_index - (i-1))
    return interpolated_arr
    


filename = 'data10.npy'
if os.path.exists(filename):
    # Load array from the npy file
    a = np.load(filename)
    print("  Loading  ", filename)
else:
    a = np.zeros(tau*n + ds+1)
    for i in range(10, tau*n + ds + 1,  ds):
        a[i] = avg_acc(i, x_train[train_indices[:tau*n]], y_train[train_indices[:tau*n]], x_test, y_test)   # create sparse accuracy vs data-size array using only deployed data
        if i%100 == 0:
            print("Accuracy table iteration:  ", i)
    np.save(filename, a)


## linear interpolate
a[-1] = np.max(a)
b = np.interp(np.arange(len(a)), np.arange(0, len(a), 10), a[::10])
del a 
a = b

## non decreasing with max
#a = non_decreasing(a)

## force monotonic increase, almost concave
#a = interpolate_non_monotonic(a, 0)

## make a strictly convex a
#x = np.linspace(0, 1, tau*n + ds+1)
#a = convex_func(x)


## Train the model
beta = 0
delta = 3300  # lambda/n^2/L^2 ... how to ensure that ds remains 10?

welfare1 = []
welfare01 = []
welfare02 = []
welfare04 = []
welfare06 = []
welfare08 = []


# FedBR-BG
for h in range(H):
    modelB = create_model()
    ## model training, not necessary to actually do it
#    for epoch in range(num_epochs):
#        total_gradients = [tf.zeros_like(var) for var in modelB.trainable_variables]
#        for i in range(num_agents):
#            #print("Game ", h, " Device ", i, " Data ", s[i])
#            if s[i] <= 0:
#                continue
#            x_batch, y_batch = x_train_splits[i], y_train_splits[i]

#            indices = np.arange(x_batch.shape[0])
#            np.random.shuffle(indices)
#            selected_indices = indices[:s[i]]    # Select s_i samples from the shuffled indices for local gradients

#            gradients = compute_gradients(x_batch[selected_indices], y_batch[selected_indices], modelB)     # returns vector of gradients for every layer
#            
#            for layer, grads in enumerate(gradients):
#                total_gradients[layer] += grads / num_agents  # Accumulate gradients in each layer

#        optimizer.apply_gradients(zip(total_gradients, modelB.trainable_variables))    # update model at the server, end of round

    temp_s = s.copy()
    u = (s*0).astype(float)
    c = np.ones(n)*0.005        # cost per sample c
    #c = rand_cost
    #c = np.array([cost(s_i) for s_i in s])      # c(s)
    
    
    A = np.sum(c**(-1))**(-1)           # is this even c(s)??? no
    C = np.sum(c)
    betaPlus  = (A*n*(n-2)+C + np.sqrt((A*n*(n-2)+C)**2 - 4*C*A*(n-1)**2)) / (2*C)
    betaMinus = (A*n*(n-2)+C - np.sqrt((A*n*(n-2)+C)**2 - 4*C*A*(n-1)**2)) / (2*C)
    if ( betaPlus >= 0 and betaPlus <= (1-1/n) ):
        beta = betaPlus
    else:
        beta = betaMinus
    
#    print(beta)
#    beta = 0.9


    for i in range(num_agents):
#        pred = modelB.predict(x_test_splits[i], verbose=0)
#        pred_labels = np.argmax(pred, axis=1)
#        acc = np.mean(pred_labels == y_test_splits[i])
#        print("Game iteration ", h, "Device ", i, " local accuracy: ", acc)
        
        u[i] = a[np.sum(s)] - (1-beta)*c[i]*s[i] - beta/(n-1)*(np.sum(c*s) - c[i]*s[i])
        #du_ds = relu(a[np.sum(s)+ds] - a[np.sum(s)]) / ds - (1-beta)*0.005 #*np.heaviside(s[i], 0)   # ??????
        du_ds = relu(a[np.sum(s)+ds] - a[np.sum(s)]) / ds - (1-beta)*0.005 # (1-beta)*c[i]
        print(du_ds)

        if (s[i] == 0 and du_ds < 0) or (s[i] == tau and du_ds > 0):
            temp_s[i] = s[i]        # do nothing if value tries to go out of bounds
        else:
            # choose between different update methods
            temp_s[i] = s[i] + round(delta*du_ds)    
            #temp_s[i] = round(s[i] + delta*du_ds, -1)  # round to nearest 10
            #temp_s[i] = s[i] + np.sign(du_ds)
            #temp_s[i] = round(s[i] + (sigmoid(du_ds)-0.5)*12000)
        
        if temp_s[i] < 0:
            temp_s[i] = 0
        if temp_s[i] > tau:
            temp_s[i] = tau
    
    s = temp_s.copy()

    welfare1.append(welfare(1, u))
    welfare01.append(welfare(0.1,u))
    welfare02.append(welfare(0.2,u))
    welfare04.append(welfare(0.4,u))
    welfare06.append(welfare(0.6,u))
    welfare08.append(welfare(0.8,u))
    
    del modelB
    print("Game ", h, " S: ", s)


plt.plot(welfare1)
plt.xlabel('Game iterations')
plt.ylabel('Welfare')
plt.title('1.0 mean Welfare')
plt.show()


#fig, axs = plt.subplots(3, 2)
#axs[0, 0].plot(welfare1)
#axs[0, 0].set_title('1.0 Welfare')
#axs[0, 1].plot(welfare01, 'tab:orange')
#axs[0, 1].set_title('0.1 Welfare')
#axs[1, 0].plot(welfare02, 'tab:green')
#axs[1, 0].set_title('0.2 Welfare')
#axs[1, 1].plot(welfare04, 'tab:red')
#axs[1, 1].set_title('0.4 Welfare')
#axs[2, 0].plot(welfare06, 'tab:purple')
#axs[2, 0].set_title('0.6 Welfare')
#axs[2, 1].plot(welfare08, 'tab:cyan')
#axs[2, 1].set_title('0.8 Welfare')

#plt.show()

plt.plot(a)
plt.xlabel('Data samples')
plt.ylabel('Accuracy')
plt.show()



