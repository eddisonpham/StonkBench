import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.parametrizations import weight_norm

from src.models.base.base_model import DeepLearningModel


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)


class Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x


class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, seq_len, conv_dropout=0.05):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()

class QuantGAN(DeepLearningModel):
    def __init__(
        self,
        seq_len: int,
        nz: int = 3,
        clip_value: float = 0.01,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.nz = nz
        self.clip_value = clip_value
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None

        self.training_data = None
        self.best_state_dict = None
        self.best_val_gen_loss = float('inf')
        self._best_model_loaded = False

    def _init_networks(self):
        if self.generator is None:
            self.generator = Generator().to(self.device)
            self.discriminator = Discriminator(self.seq_len).to(self.device)

    def _init_optimizers(self):
        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        x = batch.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def fit(self, data_loader, num_epochs: int = 15):
        all_batches = []
        for batch, _ in data_loader:
            batch = batch.to(self.device)
            if batch.dim() == 2:
                batch = batch.unsqueeze(-1).transpose(1, 2)

            all_batches.append(batch.cpu().numpy())
        
        all_data = np.concatenate(all_batches, axis=0)
        if self.seq_len is None:
            self.seq_len = all_data.shape[1]
            print(f"Inferred sequence length: {self.seq_len}")
        self.training_data = all_data.squeeze()

        all_data_tensor = torch.tensor(all_data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(all_data_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self._init_networks()
        self._init_optimizers()

        self.best_state_dict = {
            'generator': {k: v.cpu().clone() for k, v in self.generator.state_dict().items()},
            'discriminator': {k: v.cpu().clone() for k, v in self.discriminator.state_dict().items()}
        }

        self.generator.train()
        self.discriminator.train()

        for epoch in range(num_epochs):
            for idx, (data,) in enumerate(train_loader):
                real = self._prepare_batch(data)
                batch_size, _, seq_len = real.size()
                noise = torch.randn(batch_size, self.nz, seq_len, device=self.device)

                # Train discriminator
                self.discriminator.zero_grad()
                fake = self.generator(noise).detach()
                disc_loss = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                disc_loss.backward()
                self.optimizer_d.step()
                for dp in self.discriminator.parameters():
                    dp.data.clamp_(-self.clip_value, self.clip_value)

                # Train generator
                self.generator.zero_grad()
                gen_loss = -torch.mean(self.discriminator(self.generator(noise)))
                gen_loss.backward()
                self.optimizer_g.step()

            print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {disc_loss.item():.6f}, G Loss: {gen_loss.item():.6f}")

        # Save best model
        self.best_state_dict = {
            'generator': {k: v.cpu().clone() for k, v in self.generator.state_dict().items()},
            'discriminator': {k: v.cpu().clone() for k, v in self.discriminator.state_dict().items()}
        }
        self._best_model_loaded = True
        print("Training complete.")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.generator is None:
            raise RuntimeError("Model must be trained before generating samples.")
        if not self._best_model_loaded:
            self.generator.load_state_dict(self.best_state_dict['generator'])
            self.discriminator.load_state_dict(self.best_state_dict['discriminator'])
            self._best_model_loaded = True

        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, generation_length, device=self.device)
            y = self.generator(noise).cpu().squeeze()
            y = y - y.mean()
            return y
