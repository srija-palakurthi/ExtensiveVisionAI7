
# Training MNIST Dataset with Neural Network with less than 20k parameters and achieve greater than 99.4% accuracy.
MNIST dataset is used to create a NN with less than 20k parameters and achieve 99.4% accuracy under 20 epoches

## Number of parameters used 
17504

## Acheived accuracy 
99.47%

## Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         # input =28x28x1 , RF: 3x3
        nn.Conv2d(1, 16, 3) ,     ## out=28x28x16
         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1),
         nn.Conv2d(16, 16, 3),    ## out=28x28x32,input=28x28x16,RF=5x5
         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1),     
         nn.Conv2d(16, 32, 3),    ## out=28x28x32,input=28x28x16,RF=5x5
                 nn.ReLU(),          
         nn.BatchNorm2d(32),  
         nn.Dropout2d(0.1)      
        )

        self.transition_layer1= nn.Sequential( 
            nn.Conv2d(32,16,1),              ## out= 14x14x16, input=14x14x32
            nn.ReLU(),                         # input =28x28x32
            nn.MaxPool2d(2, 2),               ## out= 14x14x32

            )      

        self.conv2 = nn.Sequential(       # input=14x14x16
        nn.Conv2d(16, 16, 3),              ## out=12 x12 x 32
                         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1) , 
        nn.Conv2d(16, 16, 3),             ## out= 10x10x16, input= 12x12x32
                         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1) , 
          nn.Conv2d(16, 32, 3,padding=1),             ## out= 10x10x16, input= 12x12x32
                         nn.ReLU(),          
         nn.BatchNorm2d(32),  
         nn.Dropout2d(0.1)  
          )

       
        self.conv_final=nn.Conv2d(32, 10, 1, bias=False) #input-5x5x16 Output: 5x5x32 
        self.gap = nn.AvgPool2d(5)  
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition_layer1(x)
        x =self.conv2(x)
        
        x = self.conv_final(x)
        x=self.gap(x)
        x=x.view(-1,10)  
        return F.log_softmax(x)
