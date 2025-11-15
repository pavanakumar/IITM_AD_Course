import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Model1(nn.Module):
    """Model for testing 1"""
    def __init__(self):
        super(Model1, self).__init__()
        # Define layers with tanh activation
        self.layer1 = nn.Linear(1, 10, bias=True)
        self.layer2 = nn.Linear(10, 20, bias=True)
        self.layer3 = nn.Linear(20, 10, bias=True)
        self.layer4 = nn.Linear(10, 1, bias=True)

        # Initialize weights with Xavier/Glorot uniform
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)  # No activation on final layer (identity)
        return x

class Model2(nn.Module):
    """Model for testing 2"""
    def __init__(self):
        super(Model2, self).__init__()
        # Define layers with tanh activation
        self.layer1 = nn.Linear(1, 10, bias=True)
        self.layer2 = nn.Linear(10, 20, bias=True)
        self.layer3 = nn.Linear(20, 20, bias=True)
        self.layer4 = nn.Linear(20, 20, bias=True)
        self.layer5 = nn.Linear(20, 20, bias=True)
        self.layer6 = nn.Linear(20, 10, bias=True)
        self.layer7 = nn.Linear(10, 1, bias=True)

        # Initialize with random normal (equivalent to randn32 with seed 1)
        self._initialize_weights()

    def _initialize_weights(self):
        # Save current random state
        current_state = torch.get_rng_state()

        # Set specific seed for this model (equivalent to MersenneTwister(1))
        torch.manual_seed(1)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4,
                     self.layer5, self.layer6, self.layer7]:
            # Julia randn32 creates normal distribution with std=1
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.zeros_(layer.bias)

        # Restore previous random state
        torch.set_rng_state(current_state)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        x = torch.tanh(self.layer6(x))
        x = self.layer7(x)  # No activation on final layer
        return x

def flatten_params(model, transpose=False):
    """
    Flatten model parameters into a single array.

    Args:
        model: PyTorch model
        transpose: If True, transpose weight matrices before flattening (C-style)
                  If False, use Fortran-style flattening

    Returns:
        Flattened parameter array
    """
    flat_params = []

    for param in model.parameters():
        if transpose and param.dim() == 2:
            # Transpose for C-style format (like Julia's transpose)
            param_data = param.data.t().flatten()
        else:
            # Fortran-style flattening (column-major, Julia default)
            param_data = param.data.flatten()

        flat_params.append(param_data.cpu().numpy())

    return np.concatenate(flat_params).astype(np.float32)

def train_model(model):
    """
    Run training to fit 1D curve
    """
    # Create exact solution data: y = 2x - x^3
    # Exact data = [([x], 2x-x^3) for x in -2:0.04f0:2]
    # This creates tuples of ([x], y) where x is in a vector
    x_vals = torch.arange(-2.0, 2.04, 0.04, dtype=torch.float32)
    data = []
    for x in x_vals:
        x_tensor = torch.tensor([[x]], dtype=torch.float32)  # [x] wrapped in a vector like Julia
        y_tensor = torch.tensor([[2*x - x**3]], dtype=torch.float32)  # scalar y
        data.append((x_tensor, y_tensor))

    # Setup optimizer
    optimizer = optim.Adam(model.parameters())

    # Training loop for 500 epochs
    model.train()
    for epoch in range(500):
        total_loss = 0.0

        for x, y in data:
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred - y).pow(2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss:.6f}")

    # Get predictions after training
    model.eval()
    with torch.no_grad():
        predictions = []
        for x, _ in data:
            pred = model(x)
            predictions.append(pred.item())

    # Flatten parameters and get total param size
    flat_params_f = flatten_params(model, transpose=False)
    print(f"Total number of params = {len(flat_params_f)}")

    # Store final parameters before changing model state
    final_state = {name: param.clone() for name, param in model.named_parameters()}

    # Restore final parameters correctly
    for name, param in model.named_parameters():
        param.data.copy_(final_state[name])

def plot_results(model, prefix):
    """Plot results comparing exact solution with model predictions"""
    # Create test data in same format as training
    x_vals = torch.arange(-2.0, 2.04, 0.04, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = []
        for x in x_vals:
            x_tensor = torch.tensor([[x]], dtype=torch.float32)
            predicted = model(x_tensor)
            predictions.append(predicted.item())

    # Create exact solution
    x_exact = np.linspace(-2, 2, 200)
    y_exact = 2 * x_exact - x_exact**3

    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, 'b-', label='Exact', linewidth=2)
    plt.scatter(x_vals.cpu().numpy(), predictions, c='red', s=20, label='PyTorch', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('1D Curve Fitting: y = 2x - x³')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(prefix, dpi=150, bbox_inches='tight')
    plt.close()

def main():

    print("Creating models...")
    model_1 = Model1()
    model_2 = Model2()

    print("Training and saving Model 1...")
    train_model(model_1)

    print("Training and saving Model 2...")
    train_model(model_2)

    print("Generating plots...")
    plot_results(model_1, "model_1.png")
    plot_results(model_2, "model_2.png")

if __name__ == "__main__":
    main()
