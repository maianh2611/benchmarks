import torch
import speechbrain as sb


class LSTMEEG(torch.nn.Module):
    """LSTMEEG.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution in the convolutional module.
    cnn_spatial_kernels: int
        Number of kernels in the 2d spatial convolution in the convolutional module.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution in the convolutional module.
    cnn_poolsize: tuple
        Pool size in the convolutional module.
    cnn_poolstride: tuple
        Pool stride in the convolutional module.
    cnn_pool_type: string
        Pooling type in the convolutional module.
    cnn_dropout: float
        Dropout probability in the convolutional module.
    cnn_activation_type: str
        Activation function of hidden layers in the convolutional module.
    attn_depth: int
        Depth of the transformer module.
    attn_heads: int
        Number of heads in the transformer module.
    attn_dropout: float
        Dropout probability for the transformer module.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGConformer(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_temporal_kernels=40,
        cnn_spatial_kernels=40,
        cnn_temporal_kernelsize=(33, 1),
        cnn_poolsize=(38, 1),
        cnn_poolstride=(17, 1),
        cnn_pool_type="avg",
        cnn_dropout=0.5,
        cnn_activation_type="elu",
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")

        C = input_shape[-2]
        T = input_shape[-3]
    
         # EMBEDDING MODULE (CONVOLUTIONAL MODULE)
        self.emb_module = PatchEmbedding(
            cnn_temporal_kernels=cnn_temporal_kernels,
            cnn_spatial_kernels=cnn_spatial_kernels,
            cnn_temporal_kernelsize=cnn_temporal_kernelsize,
            cnn_spatial_kernelsize=(1, C),
            cnn_poolsize=cnn_poolsize,
            cnn_poolstride=cnn_poolstride,
            cnn_pool_type=cnn_pool_type,
            dropout=cnn_dropout,
            activation_type=cnn_activation_type,
        )

        out = self.emb_module(torch.ones((1, T, C, 1)))
        # LSTM module
        # self.EncoderDecoderLSTM = torch.nn.Sequential()
        # self.EncoderDecoderLSTM.add_module('lstm_0', EncoderDecoderLSTM(
        #     input_shape= out.shape,
        #     hidden_size=128,
        #     num_layers=1,
        # ))

        self.emb_module.add_module("act_1", torch.nn.ReLU())
        #self.EncoderDecoderLSTM.add_module('dropout', torch.nn.Dropout(0.2))

        

        self.VanillaLSTM = sb.nnet.RNN.LSTM(
            input_shape= out.shape,
            hidden_size=64,
            num_layers=1,
        )
        _, (h_final, _) = self.VanillaLSTM(out)
        h_final = torch.squeeze(h_final, 0)
        out = self.VanillaLSTM(out)


        dense_input_size = self._num_flat_features(h_final)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size, n_neurons=dense_n_neurons,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.emb_module(x)  # (batch, time_, EEG channel, channel)
        #print('real',x.shape)
        #out = self.EncoderDecoderLSTM(x)
        _, (h_final, _) = self.VanillaLSTM(x)
        h_final = torch.squeeze(h_final, 0)
        x = self.dense_module(h_final)
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


class PatchEmbedding(torch.nn.Module):
    """Class that defines the convolutional feature extractor based on a shallow CNN, to be used in EEGConformer.

    Arguments
    ---------
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution in the convolutional module.
    cnn_spatial_kernels: int
        Number of kernels in the 2d spatial convolution in the convolutional module.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution in the convolutional module.
    cnn_poolsize: tuple
        Pool size in the convolutional module.
    cnn_poolstride: tuple
        Pool stride in the convolutional module.
    cnn_pool_type: string
        Pooling type in the convolutional module.
    dropout: float
        Dropout probability in the convolutional module.
    activation_type: str
        Activation function of hidden layers in the convolutional module.
    """

    def __init__(
        self,
        cnn_temporal_kernels,
        cnn_spatial_kernels,
        cnn_temporal_kernelsize,
        cnn_spatial_kernelsize,
        cnn_poolsize,
        cnn_poolstride,
        cnn_pool_type,
        dropout,
        activation_type,
    ):
        super().__init__()
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

        self.emb_size = cnn_spatial_kernels

        self.shallownet = torch.nn.Sequential(
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,  # (1, kernel_temp_conv),
                padding="valid",
                bias=True,
                swap=True,
            ),
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=cnn_spatial_kernelsize,  # (C, 1),
                padding="valid",
                bias=True,
                swap=True,
            ),
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
            activation,
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_poolsize,  # (1, kernel_avg_pool),
                stride=cnn_poolstride,  # (1, stride_avg_pool),
                pool_axis=[1, 2],
            ),
            torch.nn.Dropout(dropout),
        )

        self.projection = sb.nnet.CNN.Conv2d(
            in_channels=cnn_spatial_kernels,
            out_channels=cnn_spatial_kernels,
            kernel_size=(1, 1),
            padding="valid",
            bias=True,
            swap=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the convolutional feature extractor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.shallownet(
            x
        )  # (batch, time_, 1, channel) was (batch, channel, EEG channel, time_)
        x = self.projection(x)  # (batch, time_, 1, channel)
        x = x.reshape(
            x.shape[0], x.shape[1] * x.shape[2], x.shape[-1]
        )  # (batch, time_, emb_size=channel*1) #ok
        return x


# class EncoderDecoderLSTM(torch.nn.Module):
#   def __init__(self, rnn_type=sb.nnet.RNN.LSTM, input_shape=None, hidden_size=8, num_layers=1):
#     super(EncoderDecoderLSTM, self).__init__()
#     # We here use rnn_type becuase we will also try an LSTM model.
#     # Encoder initialization
#     self.encoder = sb.nnet.RNN.LSTM(input_shape=input_shape, hidden_size=hidden_size, num_layers=num_layers)
    
#     reshaped_shape = input_shape[:-2] + (hidden_size,)
#     # Decoder initialization
#     self.decoder = sb.nnet.RNN.LSTM(input_shape=reshaped_shape, hidden_size=hidden_size, num_layers=num_layers)

#   def forward(self, X):
#     """Returns the output of the seq-to-seq LSTM.

#     Arguments
#     ---------
#     X : torch.Tensor

#     Returns
#     ---------
#     out: torch.Tensor
#     """
#     #print('before encode',X.shape)
#     _, (h_final, _) = self.encoder(X)
#     #print('before squeeze',h_final.shape)
#     h_concat = torch.squeeze(h_final, 0)
#     #print('after squeeze',h_concat.shape)
#     h_concat = h_concat.reshape(h_concat.shape[0], 1, h_concat.shape[1])
#     #print('before concat',h_concat.shape)
#     h_concat = h_concat.repeat(1, X.shape[1], 1)
#     #print('real_reshaped_shape',h_concat.shape)
#     out, _ = self.decoder(h_concat)

#     return out
  