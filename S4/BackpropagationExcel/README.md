# NN BackPropogation using Excel

Backpropagation is used for training a neural network. Backpropagation adjusts the weights so that the neural network can map inputs to outputs. This backpropagation exercise shows the example calculations using excel sheet to help understand backpropagation. A neural network with two inputs, two hidden neurons, two output neurons are used and biases are ignored.

Initial weights,

w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3
w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55

Inputs i1=0.05,i2=0.1
Outputs t1=0.01, t2=0.9
## Learning rate graph

![image](https://user-images.githubusercontent.com/43727228/137796580-6ff18e08-9e85-4c86-9d26-a47259d35005.png)


![image](https://user-images.githubusercontent.com/43727228/137797021-4f0814a0-fb38-4545-afc7-b84808075b48.png)

## Forward Propogation
  h1 =w1*i1+w2+i2
  h2 =w3*i1+w4*i2
  a_h1 = σ(h1) = 1/(1+exp(-h1))
  a_h2 = σ(h2) = 1/(1+exp(-h2))
  o1 = w5 * a_h1 + w6 * a_h2
  o2 = w7 * a_h1 + w8 * a_h2
  
  a_o1 = σ(o1) = 1/(1+exp(-o1))
  a_o2 = σ(o2) = 1/(1+exp(-o2))

### Calculating the Error (Loss)
E1 = ½ * ( t1 - a_o1)²
E2 = ½ * ( t2 - a_o2)²
E_Total = E1 + E2


## Back Propogation
Back propogation is when the network learns and improves by updating the weights with the goal of minimizing error

Calculate the partial derivative of E_total with respect to w5

δE_total/δw5 = δ(E1 +E2)/δw5

δE_total/δw5 = δ(E1)/δw5       # removing E2 as there is no impact from E2 wrt w5	
             = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)	# Using Chain Rule
             = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))
                * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                   
                * (1 - a_o1 )) * a_h1                                       
             = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
Similarly
δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2

Backpropagation through the hidden layers

δE_total/δa_h1 = δ(E1+E2)/δa_h1 
               = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
               
δE_total/δa_h2 = δ(E1+E2)/δa_h2 
               = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
             = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
             

δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2


Updating the weights
    w1 = w1 - learning_rate * δE_total/δw1
    w2 = w2 - learning_rate * δE_total/δw2
    w3 = w3 - learning_rate * δE_total/δw3
    w4 = w4 - learning_rate * δE_total/δw4
    w5 = w5 - learning_rate * δE_total/δw5
    w8 = w6 - learning_rate * δE_total/δw6
    w7 = w7 - learning_rate * δE_total/δw7
    w8 = w8 - learning_rate * δE_total/δw8

Link to Excel Sheet - https://github.com/srija-palakurthi/ExtensiveVisionAI7/blob/f373b541e2cf0fb73aa5013d5c6a90eb475174d8/S4/BackpropagationExcel/Backpropagation_S4.xlsx
