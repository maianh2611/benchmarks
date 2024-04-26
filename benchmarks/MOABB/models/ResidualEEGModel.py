import torch
import speechbrain as sb

class ResidualEEGModel(torch.nn.Module):
    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        F1 = 8,
        D = 2,
        conv_module0_cnn0_kernelsizes = (63,1),
        conv_module1_cnn0_kernelsizes = (63,1),
        conv_module2_cnn0_kernelsizes = (63,1),
        conv_depthwise_kernelsizes = (1,15),
        cnn_deptwise_max_norm=1.0,
        classification_module_cnn_kernelsizes = (1,23),
        avg_pool_kernels = (4,1),
        conv_module4_avg_pool_kernels = (8,1),
        cnn_pool_type="avg",
        dropout=0.25,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        
        self.Conv2DType0_cnn = sb.nnet.CNN.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=conv_module0_cnn0_kernelsizes,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )

        self.Conv2DType0_Batchnorm = sb.nnet.normalization.BatchNorm2d(
            input_size=F1, momentum=0.01, affine=True,
        )
        self.Conv2DType1_cnn = sb.nnet.CNN.Conv2d(
            in_channels=F1,
            out_channels=F1,
            kernel_size=conv_module1_cnn0_kernelsizes,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )
        self.Conv2DType1_Batchnorm = sb.nnet.normalization.BatchNorm2d(
            input_size=F1, momentum=0.01, affine=True,
        )
        self.Conv2DType2_cnn = sb.nnet.CNN.Conv2d(
            in_channels=F1,
            out_channels=F1*D,
            kernel_size=conv_module2_cnn0_kernelsizes,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )
        self.Conv2DType2_Batchnorm = sb.nnet.normalization.BatchNorm2d(
            input_size=F1*D, momentum=0.01, affine=True,
        )
        self.Conv2DType2_activate = activation
        self.Conv2DType2_pool = sb.nnet.pooling.Pooling2d(
            pool_type=cnn_pool_type,
            kernel_size=avg_pool_kernels,
            pool_axis=[1, 2],
        )
        self.Conv2DType2_dropout = torch.nn.Dropout(p=dropout)

        # Separable Convolution Block 
        C = input_shape[2]
        # Depthwise Convolution
        self.Conv2DType3_depthwise = sb.nnet.CNN.Conv2d(
            in_channels=F1*D,
            out_channels=F1*D,
            kernel_size=(1, C),
            padding="valid",
            bias=False,
            max_norm=cnn_deptwise_max_norm,
            swap=True,
        )
        # Pointwise Convolution
        self.Conv2DType3_pointwise = sb.nnet.CNN.Conv2d(
            in_channels=F1*D,
            out_channels=F1*D,
            kernel_size=(1, 1),
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )
        self.Conv2DType3_Batchnorm = sb.nnet.normalization.BatchNorm2d(
            input_size=F1*D, momentum=0.01, affine=True,
        )

        # Separable Convolution Block 
        self.Conv2DType4_depthwise = sb.nnet.CNN.Conv2d(
            in_channels=F1*D,
            out_channels=F1*D,
            kernel_size=conv_depthwise_kernelsizes,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )

        self.Conv2DType4_pointwise = sb.nnet.CNN.Conv2d(
            in_channels=F1*D,
            out_channels=F1*D,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
            swap=True,
        )
        self.Conv2DType4_Batchnorm = sb.nnet.normalization.BatchNorm2d(
            input_size=F1*D, momentum=0.01, affine=True,
        )

        self.Conv2DType4_activate = activation

        self.Conv2DType4_pool = sb.nnet.pooling.Pooling2d(
            pool_type=cnn_pool_type,
            kernel_size=conv_module4_avg_pool_kernels,
            stride=conv_module4_avg_pool_kernels,
            pool_axis=[1, 2],
        )
        self.Conv2DType4_dropout = torch.nn.Dropout(p=dropout)

        # Classifier block

        self.classification_conv =sb.nnet.CNN.Conv2d(
            in_channels=F1*D,
            out_channels=4,
            kernel_size=classification_module_cnn_kernelsizes,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        )

        # Shape of intermediate feature maps
        out = self.getDenseShape(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
        

        self.classification_module = torch.nn.Sequential()
        self.classification_module.add_module(
            "flatten", torch.nn.Flatten(),
        )

        self.classification_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
         
        self.classification_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))
    
    def getDenseShape(self,input):
        x =  self.Conv2DType0_cnn(input)
        x = self.Conv2DType0_Batchnorm(x)
        res1 = x
        x = self.Conv2DType1_cnn(x)
        x = self.Conv2DType1_Batchnorm(x)
        x = x + res1
        x = self.Conv2DType2_cnn(x)
        x = self.Conv2DType2_Batchnorm(x)
        x = self.Conv2DType2_activate(x)
        
        x = self.Conv2DType2_pool(x)
        x = self.Conv2DType2_dropout(x)
        res2 = x
        x = self.Conv2DType3_depthwise(x)
        x = self.Conv2DType3_pointwise(x)
        x = self.Conv2DType3_Batchnorm(x)
        x = x + res2
        res3 = x
        x = self.Conv2DType4_depthwise(x)
        x = self.Conv2DType4_pointwise(x)
        x = self.Conv2DType4_Batchnorm(x)
        x = x + res3
        x = self.Conv2DType4_activate(x)
        x = self.Conv2DType4_pool(x)
        x = self.Conv2DType4_dropout(x)
        x = self.classification_conv(x)
        return x
    
    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x =  self.Conv2DType0_cnn(x)
        x = self.Conv2DType0_Batchnorm(x)
        res1 = x

        x = self.Conv2DType1_cnn(x)
        x = self.Conv2DType1_Batchnorm(x)
        # Adding the output of Conv2DType0
        x = x + res1
        x = self.Conv2DType2_cnn(x)
        x = self.Conv2DType2_Batchnorm(x)
        x = self.Conv2DType2_activate(x)
        x = self.Conv2DType2_pool(x)
        x = self.Conv2DType2_dropout(x)
        res2 = x
        x = self.Conv2DType3_depthwise(x)
        x = self.Conv2DType3_pointwise(x)
        x = self.Conv2DType3_Batchnorm(x)
        # Adding the output of Conv2DType2
        x = x + res2
        res3 = x
        x = self.Conv2DType4_depthwise(x)
        x = self.Conv2DType4_pointwise(x)
        x = self.Conv2DType4_Batchnorm(x)
        # Adding the output of Conv2DType3
        x = x + res3
        x = self.Conv2DType4_activate(x)
        x = self.Conv2DType4_pool(x)
        x = self.Conv2DType4_dropout(x)
        x = self.classification_conv(x)
        x = self.classification_module(x)
        return x