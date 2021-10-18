
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
        nn.Conv2d(1, 16, 3) ,     ## out=26x26x16
         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1),
         nn.Conv2d(16, 16, 3),    ## out=24x24x16,input=26x26x16,RF=5x5
         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1),     
         nn.Conv2d(16, 32, 3),    ## out=22x22x32,input=24x24x16,RF=7x7
                 nn.ReLU(),          
         nn.BatchNorm2d(32),  
         nn.Dropout2d(0.1)      
        )

        self.transition_layer1= nn.Sequential( 
            nn.Conv2d(32,16,1),              ## out= 22x22x16, input=22x22x32
            nn.ReLU(),                       
            nn.MaxPool2d(2, 2),               ## out= 11x11x16

            )      

        self.conv2 = nn.Sequential(       # input=11x11x16
        nn.Conv2d(16, 16, 3),              ## out=9 x9 x 16
                         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1) , 
        nn.Conv2d(16, 16, 3),             ## out= 7x7x16, input= 9x9x16
                         nn.ReLU(),          
         nn.BatchNorm2d(16),  
         nn.Dropout2d(0.1) , 
          nn.Conv2d(16, 32, 3,padding=1),             ## out= 5x5x32, input= 7x7x32
                         nn.ReLU(),          
         nn.BatchNorm2d(32),  
         nn.Dropout2d(0.1)  
          )

       
        self.conv_final=nn.Conv2d(32, 10, 1, bias=False) #input-5x5x32 Output: 5x5x10 
        self.gap = nn.AvgPool2d(5)  
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition_layer1(x)
        x =self.conv2(x)
        
        x = self.conv_final(x)
        x=self.gap(x)
        x=x.view(-1,10)  
        return F.log_softmax(x)
