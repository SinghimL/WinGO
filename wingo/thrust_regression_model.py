import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from rich.progress import Progress
from typing import Tuple

class DatasetPrepocessing:
    """
    Class DatasetPrepocessing: Normalise the dataset for the machine learning model training.

    :method minmax_scaler: Min-max normalise the dataset in to a range of (-1, 1)
    """
    # Dataset and boundary unpacking
    def __init__(self, dataset: np.ndarray, x_bounds: np.ndarray):
        self.x_raw = dataset[:, :22]
        self.y_raw = dataset[:, 22]
        self.x_bounds = x_bounds
        self.lower_bounds = x_bounds.T[0]
        self.upper_bounds = x_bounds.T[1]

    # Min-max normalisation with sklearn MinMaxScaler
    def minmax_scaler(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise dataset in to a range of (-1, 1).

        :return: Normalised x and y values
        """
        from sklearn.preprocessing import MinMaxScaler

        minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
        minmax_scaler.fit(self.x_bounds.T)
        x_normalised = minmax_scaler.transform(self.x_raw)

        # Logarithmetic scaling for the thrust values (y) for better model performance
        y_logarithmic = np.log10(self.y_raw)

        return x_normalised, y_logarithmic

class ThrustPredictor:
    """
    Class ThrustPredictor: Train and predict the thrust values for the given dataset using resnet regression model.

    :method train: Train the regression model with the given dataset
    :method save:  Save the trained model to the given path
    :method load:  Load the trained model for further usage
    """
    # Model configuration and dataset preprocessing
    def __init__(self, model_config: dict, dataset: np.ndarray, x_bounds: np.ndarray, training_device: str = 'cpu'):
        """
        Initialise the ThrustPredictor class with the given model configuration and dataset.
        
        :param model_config:    Configuration of the regression model
        :param dataset:         Dataset for the regression model training
        :param x_bounds:        Boundary values for the dataset normalisation
        :param training_device: Device for the model training, default as 'cpu'
        """
        # Unpack the model configuration
        self.model_config = model_config
        self.model_label  = model_config['model_label']
        self.epochs_num   = model_config['epochs_num']
        self.batch_size   = model_config['batch_size']
        learning_rate     = model_config['learning_rate']
        weight_decay      = model_config['weight_decay']

        # Preprocess the dataset (normalise)
        self.dataset = dataset
        self.x_normalised, self.y_target = DatasetPrepocessing(dataset, x_bounds).minmax_scaler()

        # Initialise the regression model
        self.model = _RegressionResNet(model_config)
        self.device = torch.device(training_device)
        self.model.to(self.device)
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    # Method of building batch for the machine learning model
    def _build_batch(self, datapoint_indexes: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the training batch with datapoint indexes for the model training.
        
        :param datapoint_indexes: Indexes of the datapoints should be contained in the batch
        :return:                  Normalised x and y values in the batch
        """
        x_register = np.zeros((self.batch_size, 22))
        y_register = np.zeros((self.batch_size, 1))

        # Filter the datapoints with the given indexes
        for i in range(0, self.batch_size):
            x_register[i] = self.x_normalised[datapoint_indexes[i]]
            y_register[i] = self.y_target[datapoint_indexes[i]]

        # Convert the numpy arrays to torch tensors
        x_to_torch = torch.tensor(x_register).float().to(self.device)
        y_to_torch = torch.tensor(y_register).float().to(self.device)

        return x_to_torch, y_to_torch

    # Training step description
    def _regressor_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> torch.Tensor:
        """
        Training step of the regression model with the given batch of x and y values.

        :param x_batch: Normalised x values in the batch
        :param y_batch: Normalised y values in the batch
        :return:        Loss value of the training step
        """
        self.optimiser.zero_grad()
        predicted_y = self.model(x_batch)     
        loss = F.mse_loss(predicted_y, y_batch)
        loss.backward()
        self.optimiser.step()

        return loss

    # Method to train the regression model
    def train(self, batches_per_epoch: int = 16, subsample_per_batch: int = 16):
        """
        Train the regression model with the given dataset and execute the validation process.
        The training process takes reference from C-ShipGen: https://github.com/noahbagz/C_ShipGen.
        
        :param batches_per_epoch:   Number of batches per epoch, default as 16
        :param subsample_per_batch: Number of subsamples per batch, default as 16
        """
        # Import r2 score evaluation method for model validation
        from sklearn.metrics import r2_score

        # Preset the training configuration
        batch_num = len(self.dataset) // self.batch_size
        batches_per_epoch = min(batches_per_epoch, batch_num)
        loss_collect = []

        print(":hugging_face: [b orange1]Training started ...[/b orange1]\n")

        # Setup the progress bar
        with Progress() as progress:
            epoch_task = progress.add_task("[green]Epochs", total=self.epochs_num)
            # Training loop
            for i in range(0, self.epochs_num):
                epoch_loss = 0.0
                # Execute the training process muiltiple times
                for _ in range(0, batches_per_epoch):
                    indexes_randomised = np.random.choice(len(self.dataset), self.batch_size, replace = False)
                    for _ in range(0, subsample_per_batch):
                        x_batch, y_batch = self._build_batch(indexes_randomised)
                        loss = self._regressor_step(x_batch, y_batch)
                        epoch_loss += loss.item()
                        # progress.advance(batch_task)

                # Calculate the average loss per epoch
                avg_loss = epoch_loss / (batches_per_epoch * subsample_per_batch)
                loss_collect.append(avg_loss)
                if i % (self.epochs_num / 10) == 0:
                    print(f"\nEpoch: {str(i)} | Loss: {str(loss)}")
                
                progress.advance(epoch_task)
        
        print("\n:hugging_face: [b light_green] Regression Model Training Complete! [/b light_green]")
        np.savetxt(f'{self.model_label}_loss_per_epoch.csv', loss_collect, delimiter = ', ', fmt = '%.8f')

        # Model evaluation with R2 score
        self.model.eval()
        eval_size = len(self.dataset) // 25
        indexes_eval = np.random.choice(len(self.dataset), eval_size, replace = False)
        x_eval, y_calc = self._build_batch(indexes_eval)

        y_calc = y_calc.to(torch.device('cpu')).detach().numpy()
        y_pred = self.model(x_eval).to(torch.device('cpu')).detach().numpy()
        print(f"R2 score of Y: {str(r2_score(y_calc, y_pred))}")

    # Method to save the regression model
    def save(self, save_path: str):
        import json
        JSON = json.dumps(self.model_config)

        torch.save(self.model.state_dict(), f'{save_path}{self.model_label}.pth')
        with open(f'{save_path}{self.model_label}_config.json', 'w') as f:
            f.write(JSON)

    # Method to load the regression model
    def load(self, model_path: str):
        state_dict = torch.load(f'{model_path}{self.model_label}.pth', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

class _RegressionResNet(torch.nn.Module):
    """
    Class _RegressionResNet: Define the regression model structure with ResNet theory.
    The structure design takes reference from C-ShipGen: https://github.com/noahbagz/C_ShipGen
    """
    def __init__(self, network_config: dict):
        nn.Module.__init__(self)

        self.x_dim      = network_config['input_dim']
        self.y_dim      = network_config['output_dim']
        self.latent_dim = network_config['latent_dim']
        self.network    = network_config['network_structure']

        self.fc = nn.ModuleList()
        self.fc.append(self._link(self.latent_dim, self.network[0]))
        for i in range(1, len(self.network)):
            self.fc.append(self._link(self.network[i - 1], self.network[i]))
        self.fc.append(self._link(self.network[-1], self.latent_dim))

        self.x_embed = nn.Linear(self.x_dim, self.latent_dim)
        self.final_layer = nn.Sequential(nn.Linear(self.latent_dim, self.y_dim))
    
    def forward(self, x) -> torch.Tensor:
        x = self.x_embed(x)
        res_x = x
        for i in range(0, len(self.fc)):
            x = self.fc[i](x)
        x = torch.add(x, res_x)
        x = self.final_layer(x)
        
        return x
    
    def _link(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.SiLU(),
                             nn.LayerNorm(output_dim),
                             # nn.Dropout(p = 0.1)
                             nn.Dropout(p=0.1) if self.training else nn.Identity())

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_dataset = np.load('./datasets/ds2train.npy')
    testing_dataset  = np.load('./datasets/ds2test.npy')

    X_BOUNDS = np.array([[20  ,  100],  # Overall span range
                         [2   ,   30],  # Clearance over water range
                         [4   ,   25],  # Chord of bottom section range
                         [10  ,   25],  # Bottom section NACA profile range
                         [0.2 , 0.65],  # Distance to bottom from lower section range
                         [0.8 ,  1.1],  # Chord of lower section range
                         [8   ,   25],  # Lower section NACA profile range
                         [-0.2,  0.2],  # Lengthways offset of lower section range
                         [0.35, 0.85],  # Distance to bottom from middle section range
                         [0.6 ,  1.2],  # Chord of middle section range
                         [6   ,   25],  # Middle section NACA profile range
                         [-0.2,  0.4],  # Lengthways offset of middle section range
                         [0.8 , 0.95],  # Distance to bottom from upper section range
                         [0.3 ,    1],  # Chord of upper section range
                         [6   ,   25],  # Upper section NACA profile range
                         [0   ,  0.5],  # Lengthways offset of upper section range
                         [0.2 ,    1],  # Chord of tip section range
                         [6   ,   25],  # Tip section NACA profile range
                         [0   ,  0.7],  # Lengthways offset of tip section range
                         [6   ,   10],  # Vessel speed range
                         [6   ,   10],  # True wind speed range
                         [30  ,  150]]) # True wind angle range
    
    nodes = 512
    predictor_config  = {'model_label'      : 'Thrust_Regressor',
                         'input_dim'        : len(training_dataset[0]) - 1,
                         'output_dim'       : 1,
                         'latent_dim'       : nodes,
                         'network_structure': [nodes,nodes,nodes,nodes],
                         'epochs_num'       : 4000, # number of training epochs
                         'batch_size'       : 8192,
                         'learning_rate'    : 0.0005,
                         'weight_decay'     : 0.01,
                         'vs_range'         : [6, 10, 1],
                         'tws_range'        : [6, 10, 1],
                         'twa_range'        : [30, 150, 10]}


    thrust_reg = ThrustPredictor(predictor_config, training_dataset, X_BOUNDS, 'cuda:0')

    model_size = 0
    for p in thrust_reg.model.parameters():
        model_size += p.numel()
    print('Model size: ', model_size)

    thrust_reg.train(16, 8)
    thrust_reg.save('saved_model/')
