import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer

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


class Gaussianize:
    """Gaussianize transformation using quantile transformation to make data more Gaussian."""
    def __init__(self):
        self.transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        self.fitted = False
    
    def fit_transform(self, X):
        """Fit and transform data to Gaussian distribution."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result = self.transformer.fit_transform(X)
        self.fitted = True
        return result
    
    def fit(self, X):
        """Fit the transformer."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.transformer.fit(X)
        self.fitted = True
    
    def transform(self, X):
        """Transform data to Gaussian distribution."""
        if not self.fitted:
            raise ValueError("Gaussianize must be fitted before transform")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.transformer.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform from Gaussian distribution back to original distribution."""
        if not self.fitted:
            raise ValueError("Gaussianize must be fitted before inverse_transform")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.transformer.inverse_transform(X)


class QuantGAN(DeepLearningModel):
    """
    QuantGAN model for generating synthetic time series.
    
    Input: DataLoader providing batches of shape (batch_size, seq_length) - assumes data is already log returns
    Output: Generated samples of shape (num_samples, generation_length) - outputs log returns
    """
    
    def __init__(
        self,
        seq_len: int = None,
        nz: int = 3,  # Noise dimension (embedding dimension)
        clip_value: float = 0.01,
        learning_rate: float = 0.0002,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.seq_len = seq_len  # Will be inferred from data if None
        self.nz = nz
        self.clip_value = clip_value
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Networks (will be initialized in fit if seq_len is None)
        self.generator = None
        self.discriminator = None
        
        # Optimizers
        self.optimizer_g = None
        self.optimizer_d = None
        
        # Preprocessing scalers (fitted during training)
        self.standard_scaler1 = StandardScaler()
        self.standard_scaler2 = StandardScaler()
        self.gaussianize = Gaussianize()
        self.scalers_fitted = False
        
        # Store training data statistics for filtering
        self.training_log_returns = None
    
    def _init_networks(self):
        """Initialize networks if not already initialized."""
        if self.generator is None:
            self.generator = Generator().to(self.device)
            self.discriminator = Discriminator(self.seq_len).to(self.device)
    
    def _init_optimizers(self):
        """Initialize optimizers."""
        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)
    
    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert batch from (batch_size, seq_len) to (batch_size, 1, seq_len) for Conv1d."""
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        x = batch.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len) for Conv1d
        return x
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess log returns: StandardScaler -> Gaussianize -> StandardScaler.
        
        Args:
            data: Log returns of shape (N, seq_len) or (N*seq_len,)
        
        Returns:
            Preprocessed data
        """
        # Flatten for preprocessing
        original_shape = data.shape
        data_flat = data.flatten().reshape(-1, 1)
        
        # Apply preprocessing pipeline
        data_scaled1 = self.standard_scaler1.fit_transform(data_flat)
        data_gaussianized = self.gaussianize.fit_transform(data_scaled1)
        data_scaled2 = self.standard_scaler2.fit_transform(data_gaussianized)
        
        # Reshape back to original shape
        return data_scaled2.reshape(original_shape)
    
    def _inverse_preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse preprocess: StandardScaler2 -> Gaussianize -> StandardScaler1.
        
        Args:
            data: Preprocessed data of shape (N, seq_len) or (N*seq_len,)
        
        Returns:
            Log returns
        """
        # Flatten for inverse preprocessing
        original_shape = data.shape
        data_flat = data.flatten().reshape(-1, 1)
        
        # Apply inverse preprocessing pipeline
        data_scaled2_inv = self.standard_scaler2.inverse_transform(data_flat)
        data_gaussianized_inv = self.gaussianize.inverse_transform(data_scaled2_inv)
        data_scaled1_inv = self.standard_scaler1.inverse_transform(data_gaussianized_inv)
        
        # Reshape back to original shape
        return data_scaled1_inv.reshape(original_shape)
    
    def fit(self, data_loader, num_epochs: int = 50, *args, **kwargs):
        """
        Train QuantGAN model using WGAN-style training.
        
        Args:
            data_loader: DataLoader providing batches of shape (batch_size, seq_length)
                        Assumes data is already log returns.
            num_epochs: Number of training epochs
        """
        all_batches = []
        for batch, _ in data_loader:
            batch = batch.to(self.device)
            if batch.dim() == 2:
                batch = batch.unsqueeze(-1)
            all_batches.append(batch.cpu().numpy())
        
        all_data = np.concatenate(all_batches, axis=0)
        if self.seq_len is None:
            self.seq_len = all_data.shape[1]
            print(f"Inferred sequence length: {self.seq_len}")
        self.training_log_returns = all_data.squeeze()

        print("Preprocessing data...")
        all_data_preprocessed = self._preprocess_data(all_data.squeeze()) 
        self.scalers_fitted = True
        all_data_tensor = torch.tensor(all_data_preprocessed, dtype=torch.float32)
        
        class PreprocessedDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __len__(self):
                return len(self.data)
        
        dataset = PreprocessedDataset(all_data_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self._init_networks()
        self._init_optimizers()
        
        print(f"Training QuantGAN for {num_epochs} epochs...")
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(num_epochs):
            for idx, data in enumerate(train_loader):
                real = self._prepare_batch(data)
                batch_size, _, seq_len = real.size()
                
                noise = torch.randn(batch_size, self.nz, seq_len, device=self.device)
                
                # Train discriminator
                self.discriminator.zero_grad()
                fake = self.generator(noise).detach()
                disc_loss = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                disc_loss.backward()
                self.optimizer_d.step()

                # Clip discriminator weights
                for dp in self.discriminator.parameters():
                    dp.data.clamp_(-self.clip_value, self.clip_value)

                self.generator.zero_grad()
                gen_loss = -torch.mean(self.discriminator(self.generator(noise)))
                gen_loss.backward()
                self.optimizer_g.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Discriminator Loss: {disc_loss.item():.8f}, Generator Loss: {gen_loss.item():.8f}')
        
        print('QuantGAN training complete!')
    
    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        """
        Generate synthetic log returns.
        
        Args:
            num_samples: Number of samples to generate
            generation_length: Length of each generated sample
            seed: Random seed
        
        Returns:
            Generated log returns of shape (num_samples, generation_length)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if self.generator is None:
            raise RuntimeError("Model must be trained before generating samples.")
        
        if not self.scalers_fitted:
            raise RuntimeError("Preprocessing scalers must be fitted before generating samples.")
        
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise: (num_samples, nz, generation_length)
            noise = torch.randn(num_samples, self.nz, generation_length, device=self.device)
            y_preprocessed = self.generator(noise).cpu().detach().squeeze()  # (num_samples, generation_length)
            y_preprocessed = (y_preprocessed - y_preprocessed.mean(axis=0)) / (y_preprocessed.std(axis=0) + 1e-8)
            y_preprocessed_np = y_preprocessed.numpy()
            y_log_returns = self._inverse_preprocess(y_preprocessed_np)  # (num_samples, generation_length)
            if self.training_log_returns is not None:
                max_threshold = 2 * self.training_log_returns.max()
                min_threshold = 2 * self.training_log_returns.min()
                mask = (y_log_returns.max(axis=1) <= max_threshold) & (y_log_returns.min(axis=1) >= min_threshold)
                y_log_returns = y_log_returns[mask]
                
                if len(y_log_returns) < num_samples:
                    # Generate additional samples
                    additional_needed = num_samples - len(y_log_returns)
                    additional_samples = []
                    attempts = 0
                    max_attempts = 100
                    
                    while len(additional_samples) < additional_needed and attempts < max_attempts:
                        noise_add = torch.randn(additional_needed * 2, self.nz, generation_length, device=self.device)
                        y_add_preprocessed = self.generator(noise_add).cpu().detach().squeeze()
                        y_add_preprocessed = (y_add_preprocessed - y_add_preprocessed.mean(axis=0)) / (y_add_preprocessed.std(axis=0) + 1e-8)
                        y_add_log_returns = self._inverse_preprocess(y_add_preprocessed.numpy())
                        
                        mask_add = (y_add_log_returns.max(axis=1) <= max_threshold) & (y_add_log_returns.min(axis=1) >= min_threshold)
                        additional_samples.extend(y_add_log_returns[mask_add].tolist())
                        attempts += 1
                    
                    if additional_samples:
                        y_log_returns = np.vstack([y_log_returns, np.array(additional_samples[:additional_needed])])
                y_log_returns = y_log_returns[:num_samples]
            y_log_returns = y_log_returns - y_log_returns.mean()
            
            return torch.tensor(y_log_returns, dtype=torch.float32)
