from torch import nn

class ResidualLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn, batch_norm=False, dropout_rate=None, bias=False):
        super(ResidualLayer, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.bn(x)
        out = self.activation_fn(x)
        out = self.dropout(out)
        out = out + residual
        out = self.activation_fn(out)
        
        return out


class LinearLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn, batch_norm=False, dropout_rate=None, bias=False):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        out = self.dropout(x)
        
        return out


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dims, activation_fn, use_batch_norm, dropout_rate, bias):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        input_dim = num_features
        self.activation_fn = activation_fn

        
        for i, h_dim in enumerate(hidden_dims):
            # Determine if this is the last layer
            is_last_layer = (h_dim == hidden_dims[-1])
            
            # For the last layer, use different settings
            layer = LinearLayer(
                input_dim=input_dim,
                hidden_dim=h_dim,
                activation_fn=nn.Identity() if is_last_layer else activation_fn,#  
                batch_norm=False if is_last_layer else use_batch_norm,
                dropout_rate=None if is_last_layer else dropout_rate,
                bias=bias
            )

            self.layers.append(layer)
            input_dim = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dims, num_features, activation_fn, use_batch_norm, dropout_rate, bias):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()
        # Reverse the hidden dimensions to start decoding
        hidden_dims = list(reversed(hidden_dims))

        for i in range(len(hidden_dims) - 1):
            
            layer = LinearLayer(
                input_dim=hidden_dims[i],
                hidden_dim=hidden_dims[i + 1],
                activation_fn=activation_fn,
                batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                bias=bias
            )
            self.layers.append(layer)

        # Final output layer
        final_layer = LinearLayer(
            input_dim=hidden_dims[-1],
            hidden_dim=num_features,
            activation_fn=nn.Identity(),
            batch_norm=False,
            dropout_rate=None,
            bias=bias
        )
        self.layers.append(final_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x