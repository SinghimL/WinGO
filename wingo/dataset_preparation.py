import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
from typing import Tuple
from rich import print
from rich.progress import Progress
from .physical_calculation import PreCalculation

class RawParameters:
    """
    Class ParametersGenerator: Generate a sets of the 19 wingsail parameters for dataset preparation.

    :method generate: Generate the dataset with thrustworthiness requirement, and save to .npy file (and .csv file optionally)
    """
    
    # Class constructor
    def __init__(self, dataset_size: int, save_dir: str, data_trustworthiness: str = 'positive', save_csv: bool = False):
        """
        Initialise the raw parameters generator configurations.

        :param dataset_size:         The length of the dataset to be generated
        :param data_trustworthiness: Positive trustworthiness means a set of wingsail parameters can be converted to a realistic wingsail model
        :param save_csv:             Save the dataset to .csv file additionally, default as False
        """
        # Check attribute validation
        if data_trustworthiness == 'positive':
            self.data_trustworthiness = 'feasible'
            self.negative_data_gen = False
        elif data_trustworthiness == 'negative':
            self.data_trustworthiness = 'unfeasible'
            self.negative_data_gen = True
        else:
            raise ValueError("'data_trustworthiness' should be either 'positive' or 'negative'")

        self.dataset_size  = dataset_size
        self.save_dir     = save_dir
        self.save_csv_bool = save_csv
        self.num_processes = mp.cpu_count()
    
    # Static method to randomise wingsail datapoint
    @staticmethod
    def _randomise_wingsail_datapoint() -> np.ndarray:
        """
        Randomly generate a single set of wingsail parameters.

        :return: A numpy array of 19 wingsail parameters
        """
        wingsail_datapoints = np.zeros(19)
        wingsail_datapoints[0]  = np.round(np.random.uniform(20, 100), 2)    # Overall span
        wingsail_datapoints[1]  = np.round(np.random.uniform(2, 30), 2)      # Clearance over water
        wingsail_datapoints[2]  = np.round(np.random.uniform(4, 25), 2)      # Chord of bottom section
        wingsail_datapoints[3]  = np.random.randint(10, 25)                  # Bottom section NACA profile
        wingsail_datapoints[4]  = np.round(np.random.uniform(0.2, 0.65), 2)  # Distance to bottom from lower section
        wingsail_datapoints[5]  = np.round(np.random.uniform(0.8, 1.1), 2)   # Chord of lower section
        wingsail_datapoints[6]  = np.random.randint(8, 25)                   # Lower section NACA profile
        wingsail_datapoints[7]  = np.round(np.random.uniform(-0.2, 0.2), 2)  # Lengthways offset of lower section
        wingsail_datapoints[8]  = np.round(np.random.uniform(0.35, 0.85), 2) # Distance to bottom from middle section
        wingsail_datapoints[9]  = np.round(np.random.uniform(0.6, 1.2), 2)   # Chord of middle section
        wingsail_datapoints[10] = np.random.randint(6, 25)                   # Middle section NACA profile
        wingsail_datapoints[11] = np.round(np.random.uniform(-0.2, 0.4), 2)  # Lengthways offset of middle section
        wingsail_datapoints[12] = np.round(np.random.uniform(0.8, 0.95), 2)  # Distance to bottom from upper section
        wingsail_datapoints[13] = np.round(np.random.uniform(0.3, 1), 2)     # Chord of upper section
        wingsail_datapoints[14] = np.random.randint(6, 25)                   # Upper section NACA profile
        wingsail_datapoints[15] = np.round(np.random.uniform(0, 0.5), 2)     # Lengthways offset of upper section
        wingsail_datapoints[16] = np.round(np.random.uniform(0.2, 1), 2)     # Chord of tip section
        wingsail_datapoints[17] = np.random.randint(6, 25)                   # Tip section NACA profile
        wingsail_datapoints[18] = np.round(np.random.uniform(0, 0.7), 2)     # Lengthways offset of tip section

        return wingsail_datapoints

    # Static method to detect wingsail constraints and trustworthiness
    @staticmethod
    def _constraint_detection(wingsail_datapoints, negative_data_gen):
        """
        Adjust the constraints and evaluate the wingsail geometry trustworthiness.

        :param wingsail_datapoints: The randomly generated wingsail parameters to be evaluated
        :param negative_data_gen:   Boolean indicating whether to generate negative datapoint
        """
        # Adjust the redline constraints
        if wingsail_datapoints[8] <= wingsail_datapoints[4]:
            return False
        elif wingsail_datapoints[12] <= wingsail_datapoints[8]:
            return False
        
        # Analyse the geometry characeristics
        span_clearance_ratio = wingsail_datapoints[0] / wingsail_datapoints[1]
        span_chord_ratio = wingsail_datapoints[0] / wingsail_datapoints[2]
        if span_clearance_ratio < 2 or span_clearance_ratio > 10:
            return False
        elif span_chord_ratio < 2 or span_chord_ratio > 6:
            return False
        elif wingsail_datapoints[6] - 2 > wingsail_datapoints[3]:
            return False
        elif wingsail_datapoints[10] - 1 > wingsail_datapoints[6]:
            return False
        elif wingsail_datapoints[13] > wingsail_datapoints[9]:
            return False
        elif wingsail_datapoints[14] > wingsail_datapoints[10]:
            return False
        elif wingsail_datapoints[16] > wingsail_datapoints[13]:
            return False
        elif wingsail_datapoints[17] > wingsail_datapoints[14]:
            return False
        
        # Calculate the wingsail geometry edges smoothness features
        leading_points_coords = np.array([[0, 0],
                                        [(wingsail_datapoints[2] * wingsail_datapoints[7]), (wingsail_datapoints[0]*wingsail_datapoints[4])],
                                        [(wingsail_datapoints[2] * wingsail_datapoints[11]), (wingsail_datapoints[0]*wingsail_datapoints[8])],
                                        [(wingsail_datapoints[2] * wingsail_datapoints[15]), (wingsail_datapoints[0]*wingsail_datapoints[12])],
                                        [(wingsail_datapoints[2] * wingsail_datapoints[18]), wingsail_datapoints[0]]])
        leading_points_vector = np.diff(leading_points_coords, axis=0)
        leading_edge_gradient = leading_points_vector[:, 0] / leading_points_vector[:, 1]
        leading_edge_gradient_diff = np.diff(leading_edge_gradient)
        trailing_points_coords = np.array([[wingsail_datapoints[2], 0],
                                        [(wingsail_datapoints[2] * (wingsail_datapoints[5] + wingsail_datapoints[7])), (wingsail_datapoints[0]*wingsail_datapoints[4])],
                                        [(wingsail_datapoints[2] * (wingsail_datapoints[9] + wingsail_datapoints[11])), (wingsail_datapoints[0]*wingsail_datapoints[8])],
                                        [(wingsail_datapoints[2] * (wingsail_datapoints[13] + wingsail_datapoints[15])), (wingsail_datapoints[0]*wingsail_datapoints[12])],
                                        [(wingsail_datapoints[2] * (wingsail_datapoints[16] + wingsail_datapoints[18])), wingsail_datapoints[0]]])
        trailing_points_vector = np.diff(trailing_points_coords, axis=0)
        trailing_edge_gradient = trailing_points_vector[:, 0] / trailing_points_vector[:, 1]
        trailing_edge_gradient_diff = np.diff(trailing_edge_gradient)
        variance_to_gradient = (sum(trailing_edge_gradient ** 2) + sum(leading_edge_gradient ** 2))/ 8
        wingsail_gradient = 5e-6 ** variance_to_gradient
        variance_to_smoothness = (sum(trailing_edge_gradient_diff ** 2) + sum(leading_edge_gradient_diff ** 2))/ 6
        wingsail_smoothness = 5e-6 ** variance_to_smoothness

        # Adjust the wingsail trustworthiness status
        if negative_data_gen:
            if wingsail_gradient > 0.85:
                return False
            elif wingsail_smoothness > 0.85:
                return False
            else:
                return True
        else:
            if wingsail_gradient <= 0.85:
                return False
            elif wingsail_smoothness <= 0.85:
                return False
            else:
                return True

    # Static method to generate unique datapoint
    @staticmethod
    def _generate_unique_datapoint(generated_datapoints, lock, negative_data_flag) -> np.ndarray:
        """
        Insure the datapoints generated by different processes are unique and satisfy the constraints.

        :param generated_datapoints: The list of generated datapoints
        :param lock:                 A lock to prevent the datapoint uniqueness
        :param negative_data_flag:   Boolean indicating whether to generate negative datapoint
        :return:                     A unique and constraint-satisfied wingsail datapoint
        """
        while True:
            randomised_datapoint = RawParameters._randomise_wingsail_datapoint()
            if RawParameters._constraint_detection(randomised_datapoint, negative_data_flag):
                datapoint = tuple(randomised_datapoint)
                with lock:
                    if datapoint not in generated_datapoints:
                        generated_datapoints.append(datapoint)

                        return randomised_datapoint

    # Dataset generation task setup with multiprocessing
    def _generate_dataset_mp(self, progress, task) -> list:
        """
        :param progress and param task: For the rich progress bar display
        :return:                        The generated wingsail dataset
        """
        negative_data_flag = self.negative_data_gen

        # Setup the multiprocessing manager and lock
        with Manager() as manager:
            generated_datapoints = manager.list()
            lock = manager.Lock()

            # Setup the multiprocessing pool
            with mp.Pool(self.num_processes) as pool:
                results = []
                while len(results) < self.dataset_size:
                    result = pool.apply_async(RawParameters._generate_unique_datapoint,
                                              args = (generated_datapoints, lock, negative_data_flag))
                    results.append(result)
                # Save the generated datapoints to the dataset
                wingsail_dataset = []
                for result in results:
                    datapoint = result.get()
                    if datapoint is not None:
                        wingsail_dataset.append(datapoint)
                        progress.update(task, advance=1)

        return wingsail_dataset

    # Dataset generation function to be called
    def generate(self) -> str:
        """
        :return: The generated dataset path
        """
        t = time.perf_counter()

        # Generate the dataset with generation config and multiprocessing
        with Progress() as progress:
            task = progress.add_task("[green]Generating dataset...", total=self.dataset_size)
            wingsail_dataset = self._generate_dataset_mp(progress, task)
        
        # Save the dataset to .npy or a .csv file additionally
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f'ws_dataset_{self.dataset_size}_{self.data_trustworthiness}.npy')
        np.save(save_path, wingsail_dataset)

        # Save the dataset to .csv file additionally
        if self.save_csv_bool:
            save_path_csv = os.path.join(self.save_dir, f'ws_dataset_{self.dataset_size}_{self.data_trustworthiness}.csv')
            np.savetxt(save_path_csv, wingsail_dataset, fmt='%.3f', delimiter=', ')

        print(f"Time elapsed for dataset generation: {time.perf_counter() - t:.6f} seconds.\nDataset saved to [b light_green]{save_path}[/b light_green]")

        return save_path
    
class DatasetBuild:
    """
    Class DatasetBuild: Calculate the maximum thrust for each wingsail parameters datapoint, and build the training and testing dataset.

    :method save_dataset:
    """
    # Configuration initialisation
    def __init__(self, parameter_dataset_path: str, calculation_config: dict):
        """
        Initialise the dataset and its calculation configurations.

        :param parameter_dataset_path: The path to the original wingsail parameters dataset, in .npy format
        :param calculation_config:     The calculation configurations for thrust calculation
        """
        # Import wingsail parameters dataset
        self.parameter_dataset = np.load(parameter_dataset_path)
        self.dataset_dir = os.path.dirname(os.path.abspath(parameter_dataset_path))
        self.original_filename, _ = os.path.splitext(os.path.abspath(parameter_dataset_path))

        self.dataset_filename = f'{self.original_filename}_with_thrust'
        self.dataset_save_path = os.path.join(self.dataset_dir, self.dataset_filename)

        # Initialise calculation configurations
        self.vessel_speed_range    = calculation_config['vessel_speed_range']
        self.true_wind_speed_range = calculation_config['true_wind_speed_range']
        self.true_wind_angle_range = calculation_config['true_wind_angle_range']
        self.physical_constants    = calculation_config['physical_constants']
        self.aoa_range             = calculation_config['aoa_range']
        self.refinement_level      = calculation_config['refinement_level']
        self.neuralfoil_network    = calculation_config['neuralfoil_network']
        self.input_vs_range  = [self.vessel_speed_range[0], self.vessel_speed_range[1] + self.vessel_speed_range[2], self.vessel_speed_range[2]]
        self.input_tws_range = [self.true_wind_speed_range[0], self.true_wind_speed_range[1] + self.true_wind_speed_range[2], self.true_wind_speed_range[2]]
        self.input_twa_range = [self.true_wind_angle_range[0], self.true_wind_angle_range[1] + self.true_wind_angle_range[2], self.true_wind_angle_range[2]]

    # Thrust calculation for wingsail parameters in the dataset
    def thrust_calculation(self, save_csv: bool = False) -> str:
        """
        Calculate the maximum thrust for each wingsail parameters datapoint in the dataset.

        :param save_csv: Save the dataset to .csv file additionally, default as False
        :return:         The path to the .npy dataset with thrust calculation results
        """
        # Progress bar setup
        datapoint_num = len(self.parameter_dataset)
        with Progress() as progress:
            task = progress.add_task("[indian_red1]Processing wingsail performance ...", total = datapoint_num)
            # Calculate the thrust for each wingsail parameters datapoint in various sailing conditions
            dataset_with_result = []
            for datapoint_index in range(len(self.parameter_dataset)):
                wincal = PreCalculation(self.parameter_dataset[datapoint_index], self.physical_constants, self.refinement_level)
                for vessel_speed in range(*self.input_vs_range):
                    for true_wind_speed in range(*self.input_tws_range):
                        for true_wind_angle in range(*self.input_twa_range):
                            sailing_condition = [vessel_speed, true_wind_speed, true_wind_angle]
                            max_thrust, _ = wincal.max_thrust_with_neuralfoil(sailing_condition, self.aoa_range, self.neuralfoil_network)
                            data_to_add = np.array([vessel_speed, true_wind_speed, true_wind_angle, max_thrust])
                            data_to_save = np.append(self.parameter_dataset[datapoint_index], data_to_add)
                            dataset_with_result.append(data_to_save)
                progress.update(task, advance=1)

        # Save the dataset with thrust calculation results to .npy or a .csv file additionally
        np.save(f'{self.dataset_save_path}.npy', dataset_with_result)
        if save_csv:
            np.savetxt(f'{self.dataset_save_path}.csv', dataset_with_result, fmt='%.3f', delimiter=', ')

        return f'{self.dataset_save_path}.npy'

    # Build the training and testing dataset for machine learning
    def build_training_and_testing_dataset(self, train_test_ratio: float = 0.8, training_filename: str = 'ds2train.npy',
                                           testing_filename: str = 'ds2test.npy') -> Tuple[str, str]:
        """
        Reconstruct the dataset with the thrust calculation results into training and testing dataset for machine learning.

        :param train_test_ratio:  The ratio of training dataset to the whole dataset, default as 0.8
        :param training_filename: The filename for the training dataset, default as 'ds2train.npy'
        :param testing_filename:  The filename for the testing dataset, default as 'ds2test.npy'
        """
        # Load and shuffle the dataset with thrust calculation results
        dataset_with_result = np.load(f'{self.dataset_save_path}.npy')
        np.random.shuffle(dataset_with_result)

        # Split the dataset into training and testing dataset
        train_size = int(len(dataset_with_result) * train_test_ratio)
        training_dataset = dataset_with_result[:train_size]
        testing_dataset = dataset_with_result[train_size:]

        # Save the training and testing dataset to .npy file
        training_dataset_path = os.path.join(self.dataset_dir, training_filename)
        testing_dataset_path = os.path.join(self.dataset_dir, testing_filename)
        np.save(training_dataset_path, training_dataset)
        np.save(testing_dataset_path, testing_dataset)

        return training_dataset_path, testing_dataset_path

if __name__ == "__main__":
    # Generate the raw parameter datapoints
    DATASET_DIR = './test_dataset'
    DATASET_SIZE = 30

    dataset_generator = RawParameters(dataset_size = DATASET_SIZE,
                            save_dir = DATASET_DIR,
                            data_trustworthiness = 'positive',
                            save_csv = True)
    raw_dataset_path = dataset_generator.generate()

    print(f"Parameters dataset shape: {np.load(raw_dataset_path).shape}")

    #################################################################################################
    # Calculate the thrust for each wingsail parameters datapoint in the dataset

    PARAMETER_DATASET_PATH = raw_dataset_path
    DATASET_CALCULATION_CONFIG = {'vessel_speed_range'   : [6, 10, 1],
                                  'true_wind_speed_range': [6, 10, 1],
                                  'true_wind_angle_range': [30, 150, 10],
                                  'physical_constants'   : {'sea_surface_roughness'  : 2e-4,
                                                            'air_density'            : 1.205,
                                                            'air_kinematic_viscosity': 15.06e-6},
                                  'aoa_range'            : [1, 25, 1],
                                  'refinement_level'     : 1,
                                  'neuralfoil_network'   : 'xlarge'}
    
    dataset_reconstructor = DatasetBuild(parameter_dataset_path = PARAMETER_DATASET_PATH,
                                         calculation_config = DATASET_CALCULATION_CONFIG)

    calculated_dataset_path = dataset_reconstructor.thrust_calculation(save_csv = True)
    print(f"Parameters dataset shape: {np.load(calculated_dataset_path).shape}")