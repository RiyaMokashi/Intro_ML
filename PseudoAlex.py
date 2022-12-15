#Pseudo Alex Architecture

class ALEX_CNN(nn.Module):
    def __init__(self, input_channels, conv_features, fc_features, output_size):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c_lay1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=0)
        self.c_lay2 = nn.Conv2d(64, 256, kernel_size=5, stride = 1, padding=2)
        self.c_lay3 = nn.Conv2d(256, 384, kernel_size=3, stride = 1, padding=1)
        self.c_lay4 = nn.Conv2d(384, 256, kernel_size=3, stride = 1, padding=1)
        self.c_lay5 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding=1)
        self.fc1 = nn.Linear(conv_feature * 6 * 6, fc_feature)
        self.fc2 = nn.Linear(fc_feature, fc_feature)
        self.fc3 = nn.Linear(fc_feature, output_size)
        self.Adaptive_AvgPool2d = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x):
        
        x = self.relu(self.c_lay1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.c_lay2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.c_lay3(x))
        x = self.max_pool2d(x)
        x = self.relu(self.c_lay4(x))
        x = self.max_pool2d(x)
        x = self.relu(self.c_lay5(x))
        x = self.max_pool2d(x)
        x = self.Adaptive_AvgPool2d(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x