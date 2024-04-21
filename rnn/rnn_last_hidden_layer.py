
import torch
import torch.nn as nn

def run_bidirectional_lstm(input_tensor, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True):
    """
    Function to create and run a bidirectional LSTM, extracting and concatenating the last hidden states.

    Args:
    input_tensor (Tensor): Input tensor of shape [batch, sequence_length, input_size]
    input_size (int): Size of input features
    hidden_size (int): Number of features in hidden state
    num_layers (int): Number of recurrent layers
    bidirectional (bool): If True, LSTM is bidirectional
    batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature)

    Returns:
    Tensor: The concatenated last hidden states of the forward and backward directions from the last layer, or just the forward state if unidirectional.
    """
    # Create the LSTM model
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    # Initialize hidden and cell states
    num_directions = 2 if bidirectional else 1
    batch_size = input_tensor.size(0) if batch_first else input_tensor.size(1)
    h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)

    # Forward pass
    outputs, (hn, cn) = lstm(input_tensor, (h0, c0))

    # Extract the last hidden states for the last layer
    hn_last = hn.view(num_layers, num_directions, batch_size, hidden_size)[-1]

    forward_hidden = hn_last[0]
    if bidirectional:
        backward_hidden = hn_last[1]
        # Concatenate the hidden states from both directions
        last_hidden = torch.cat((forward_hidden, backward_hidden), dim=-1)
    else:
        last_hidden = forward_hidden

    return last_hidden

# Example usage
# Example input (batch_size=3, sequence_length=5, input_size=10)
inputs = torch.randn(3, 5, 10)
hidden_size = 20
num_layers = 2
last_hidden_state = run_bidirectional_lstm(inputs, 10, hidden_size, num_layers, bidirectional=True, batch_first=True)

print("Last Hidden State:", last_hidden_state)
