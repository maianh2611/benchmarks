import torch
import speechbrain as sb


class LSTMEEG(torch.nn.Module):
    """LSTMEEG.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        conv0_kernels = 8,
        conv0_size = 3,
        conv1_kernels = 32,
        conv1_size = 5,
        conv2_kernels = 64,
        conv2_size = 7,
        dropout=0.5,
        dense_max_norm=0.25,
        num_neurons = 32,
        activation_type="relu",
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
        self.default_sf = 128  # sampling rate of the original publication (Hz)


        self.conv_module_0 = torch.nn.Sequential()
        self.conv_module_0.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=conv0_kernels,
                kernel_size=conv0_size,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module_0.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv0_kernels, momentum=0.01, affine=True,
            ),
        ),
        self.conv_module_0.add_module(
            "activation",
            activation,
        )
        self.conv_module_0.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type='max',
                kernel_size=[2,2],
                pool_axis=[1, 2],
            ),
        )

        self.conv_module_1 = torch.nn.Sequential()
        self.conv_module_1.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=conv0_kernels,
                out_channels=conv1_kernels,
                kernel_size=conv1_size,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module_1.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv1_kernels, momentum=0.01, affine=True,
            ),
        ),
        self.conv_module_1.add_module(
            "activation",
            activation,
        )
        self.conv_module_1.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type='max',
                kernel_size=[2,2],
                pool_axis=[1, 2],
            ),
        )
        self.conv_module_2 = torch.nn.Sequential()
        self.conv_module_2.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=conv1_kernels,
                out_channels=conv2_kernels,
                kernel_size=conv2_size,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module_2.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv2_kernels, momentum=0.01, affine=True,
            ),
        ),
        self.conv_module_2.add_module(
            "activation",
            activation,
        )
        # self.conv_module_2.add_module(
        #     "pool_1",
        #     sb.nnet.pooling.Pooling2d(
        #         pool_type='max',
        #         kernel_size=[2,2],
        #         pool_axis=[1, 2],
        #     ),
        # )
        self.conv_module_2.add_module("dropout_1", torch.nn.Dropout(p=dropout))

       

        ones_tensor = torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        convOut = self.conv_module_0(ones_tensor)
        convOut = self.conv_module_1(convOut)
        convOut = self.conv_module_2(convOut)

        self.lstm = sb.nnet.RNN.LSTM(
            input_shape= convOut.shape,
            hidden_size=num_neurons,
            num_layers=1,
        )
        outLSTM,_ =  self.lstm(convOut)


        # Shape of intermediate feature maps

        dense_input_size = self._num_flat_features(outLSTM)
        
        self.lstm_dropout = torch.nn.Dropout(p=dropout)


        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(1),
        )
        self.dense_module.add_module(
            "fc_out_1",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=4,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def getDenseShape(self,input):
        x = input
        x = self.conv_module(x)
        return x
        
    def _num_flat_features(self, x, startDim = 1):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[startDim:]  # all dimensions except the batch dimension
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
        x = self.conv_module_0(x)
        x = self.conv_module_1(x)
        x = self.conv_module_2(x)
        out, _=  self.lstm(x)
        
        x = self.lstm_dropout(out)
        x = self.dense_module(x)
        return x
