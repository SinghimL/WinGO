import os
import numpy as np
from .geometry_build import Geometrisation
from typing import Tuple
from rich import print

class PreCalculation:
    """
    Class PreCalculation: Calculate the wingsail performance with given parameters and sailing conditions.

    :method max_thrust_with_xfoil:          Calculate the maximum thrust of the wingsail with given sailing condition using Xfoil
    :method max_thrust_with_neuralfoil:     Calculate the maximum thrust of the wingsail with given sailing condition using Neuralfoil
    :method solve_wingsail_trustworthiness: Calculate the smoothness level of the wingsail edges with the tangent variance formula
    """
    # Data cleansing and preprocessing
    def __init__(self, wingsail_parameters: list, physical_constants: dict = {}, refinement_level: int = 1, xfoil_root: str = "xfoil"):
        """
        Wingsail performance calculation with given conditions.

        :param wingsail_parameters: Wingsail parameters list
        :param physical_constants:  Physical constants dictionary
        :param refinement_level:    Refinement level of the geometry, default as 1
        :param xfoil_root:          The path to the xfoil executable program
        """
        # Verify input parameters
        if not isinstance(refinement_level, int) or refinement_level < 1 or refinement_level > 9:
            raise TypeError("refinement_level: Have to be an integer and in the range of 1 to 9")
        if len(physical_constants) != 3:
            print(f":construction: [bright_yellow]Notification:[/bright_yellow] Input physical constants incomplete, reset to default values.")
            physical_constants = {'sea_surface_roughness'  : 2e-4,
                                  'air_density'            : 1.205,
                                  'air_kinematic_viscosity': 15.06e-6} # Setup default physical constants
        
        # Define initial physical constants
        self.sea_surface_roughness   = physical_constants['sea_surface_roughness']
        self.air_density             = physical_constants['air_density']
        self.air_kinematic_viscosity = physical_constants['air_kinematic_viscosity']

        # Map class attributes
        self.xfoil_root = xfoil_root
        self.wingsail_id = f"Wingsail Parameters: {str(wingsail_parameters)}\nRefinement Level: {str(refinement_level)}"

        # Unpackage and globalise key variables
        self.refinement_level = refinement_level
        self.class_geometrisation = Geometrisation(wingsail_parameters, refinement_level, True)
        self.clearance_ow = self.class_geometrisation.clearance_ow
        self.section_heights = self.class_geometrisation.section_z_coords[0] + self.clearance_ow
        self.section_num = self.class_geometrisation.panel_num + 1
        self.section_2d_coords = self.class_geometrisation.section_2d_coords[:, :-5, :]
        self.span = self.class_geometrisation.span

        # Globalise geometry specification and physical model result
        self.section_chords, self.section_reorganised_coords, self.panel_areas, self.aspect_ratio, self.span_efficiency = self._solve_wing_spec()

    # Wingsail geometry sepcification calculation
    def _solve_wing_spec(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Calculate the chord length of each section, total wingsail project area and aspect ratio with 2D coordinates.

        :return: The chord length and project area of each section, wingsail total area, aspect ratio and span efficiency
        """
        # Calculate chord length and reorganise 2D coordinates for Xfoil/Neuralfoil analysis of each section
        section_chords, section_reorganised_coords, panel_areas = [], [], []
        for i in range(self.section_num):
            # Chord length calculation
            section_x_coords = (self.section_2d_coords[i])[:, 0]
            section_y_coords = (self.section_2d_coords[i])[:, 1]
            leading_x_coord = np.min(section_x_coords)
            offsetted_x_coords = section_x_coords - leading_x_coord
            section_scale = offsetted_x_coords[0]
            # Reorganise 2D coordinates
            reorganised_x_coords = offsetted_x_coords / section_scale
            reorganised_y_coords = section_y_coords / section_scale
            reorganised_2d_coords = np.column_stack((reorganised_x_coords, reorganised_y_coords))
            section_chords.append(section_scale)
            section_reorganised_coords.append(reorganised_2d_coords)

        # Calculate wing area
        panel_height = np.round(np.diff(self.section_heights)[0], 5)
        for i in range(self.section_num - 1):
            panel_area = 0.5 * (section_chords[i] + section_chords[i + 1]) * panel_height
            panel_areas.append(panel_area)
        project_area = np.sum(panel_areas)

        # Calculate aspect ratio and span efficiency
        aspect_ratio = self.span ** 2 / project_area
        tan_max_sweep_angle = ((section_chords[0] - section_chords[1]) / 2) / panel_height
        span_efficiency = 2 / (2 - aspect_ratio + np.sqrt(4 + aspect_ratio ** 2 * (1 + tan_max_sweep_angle ** 2)))
        
        return np.array(section_chords), np.array(section_reorganised_coords), np.array(panel_areas), aspect_ratio, span_efficiency
    
    # Leading edge smoothness level calculation
    def solve_wingsail_trustworthiness(self) -> float:
        """
        Calculate the smoothness level of the wingsail leading edge with the tangent variance formula.

        :return: The smoothness level of the wingsail leading edge, range from 0 to 1 (the higher the smoother)
        """
        def __rectifier(x: float) -> float:
            return (np.arctan(1 / (np.pi * np.abs(x)))/ np.pi) * (np.abs(12 * x) ** 1.8 + x ** 6) ** 0.6
        #    return x ** 2
        
        # Calculate leading edge trailing edge gradient
        x_z_coords_leading = []
        x_z_coords_trailing = []
        x_z_coords_spine = []
        for i in range(self.section_num):
            y_zero_index = np.where(self.section_2d_coords[i, :, 1] == 0)[0][0]
            x_coord_leading = self.section_2d_coords[i, y_zero_index, 0]
            x_z_coord_leading = [x_coord_leading , self.section_heights[i]]
            x_z_coords_leading.append(x_z_coord_leading)
            x_coord_trailing = self.section_2d_coords[i, 0, 0]
            x_z_coord_trailing = [x_coord_trailing , self.section_heights[i]]
            x_z_coords_trailing.append(x_z_coord_trailing)
            x_coords_spine = (x_coord_trailing - x_coord_leading) / 2 + x_coord_leading
            x_z_coord_spine = [x_coords_spine, self.section_heights[i]]
            x_z_coords_spine.append(x_z_coord_spine)

        leading_points_vector = np.diff(x_z_coords_leading, axis=0)
        leading_edge_gradient = leading_points_vector[:, 0] / leading_points_vector[:, 1]
        leading_edge_gradient_diff = np.diff(leading_edge_gradient)
        trailing_points_vector = np.diff(x_z_coords_trailing, axis=0)
        trailing_edge_gradient = trailing_points_vector[:, 0] / trailing_points_vector[:, 1]
        trailing_edge_gradient_diff = np.diff(trailing_edge_gradient)
        spine_points_vector = np.diff(x_z_coords_spine, axis=0)
        spine_gradient = spine_points_vector[:, 0] / spine_points_vector[:, 1]
        spine_gradient_diff = np.diff(spine_gradient)

        # Calculate smoothness level
        variance_to_gradient = (sum(leading_edge_gradient ** 2) + sum(trailing_edge_gradient ** 2) + sum(spine_gradient ** 2))\
                                / (len(leading_edge_gradient) + len(trailing_edge_gradient) + len(spine_gradient))
        variance_to_smoothness = (sum(leading_edge_gradient_diff ** 2) + sum(trailing_edge_gradient_diff ** 2) + sum(spine_gradient_diff ** 2)) \
                                / (len(leading_edge_gradient_diff) + len(trailing_edge_gradient_diff) + len(spine_gradient_diff))
        wingsail_trustworthiness = 0.2 ** ((__rectifier(variance_to_smoothness) + __rectifier(variance_to_gradient)) / 2)
        wingsail_trustworthiness = (np.arctan(40 * (wingsail_trustworthiness -0.8)) / np.pi) + 0.5

        return wingsail_trustworthiness

    # Reynolds number and wind speed distribution calculation
    def _renv_distribution(self, vessel_speed: float, true_wind_speed: float, true_wind_angle: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """        
        Calculate the Reynolds number and the wind speed distribution along the wingsail.

        :vessel_speed:      The input speed of the vessel in m/s
        :true_wind_speed:   The input true wind speed in m/s
        :true_wind_angle:   The input true wind angle in degrees
        :return:            The Reynolds number distribution along sailwing, the apparent wind speed distribution and the apparent wind angle
        """
        # Calculate apparent wind speed at 10m above sea level and apparent wind angle
        apperent_wind_speed_10 = np.sqrt(vessel_speed ** 2 + true_wind_speed ** 2
                                    + 2 * vessel_speed * true_wind_speed * np.cos(np.deg2rad(true_wind_angle)))
        apperent_wind_angle = np.rad2deg(np.arcsin((true_wind_speed * np.sin(np.deg2rad(true_wind_angle))) / apperent_wind_speed_10))

        # Replace zero values to avoid division error
        sea_surface_roughness = np.where(self.sea_surface_roughness <= 0, 1e-3, self.sea_surface_roughness)
        section_heights = np.where(self.section_heights <= 0, 1e-3, self.section_heights)

        # Calculate Reynolds number and wind speed distribution
        apperent_wind_speeds = apperent_wind_speed_10 * np.log(section_heights / sea_surface_roughness) / np.log(10 / sea_surface_roughness)
        reynolds = apperent_wind_speeds * self.section_chords / self.air_kinematic_viscosity

        return np.round(reynolds), apperent_wind_speeds, apperent_wind_angle

    def max_thrust_with_neuralfoil(self, sailing_condition: list, alpha_range: list = [1, 25, 1], network_model: str = 'xlarge') -> Tuple[float, float]:
        # Import dependence library
        import neuralfoil

        # Unpackage sailing condition and calculate Reynolds number distribution
        vessel_speed, true_wind_speed, true_wind_angle = sailing_condition
        reynolds, apperent_wind_speeds, apperent_wind_angle = self._renv_distribution(vessel_speed, true_wind_speed, true_wind_angle)
        full_aoa_range = np.arange(alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2])

        # Execute xfoil calculation for each section
        calculation_gap = int(self.refinement_level * 4)
        involved_section_num = int(self.class_geometrisation.panel_num / calculation_gap + 1) # 26 sections are involved in cl, cd calculation
        aoa_all_sections, cl_all_sections, cd_all_sections = [], [], []
        for i in range(involved_section_num):
            section_index = int(i * calculation_gap)
            neuralfoil_output = neuralfoil.get_aero_from_coordinates(coordinates = self.section_reorganised_coords[section_index],
                                                                     alpha = full_aoa_range,
                                                                     Re = reynolds[section_index],
                                                                     model_size = network_model)
            aoa_all_sections.append(full_aoa_range)
            cl_all_sections.append(neuralfoil_output['CL'])
            cd_all_sections.append(neuralfoil_output['CD'])
        clcd_all_in_one = np.dstack((np.array(aoa_all_sections).T, np.array(cl_all_sections).T, np.array(cd_all_sections).T))

        # Calculate thrust for each angle of attack
        wingsail_thrusts = []
        for i in range(len(clcd_all_in_one)):
            panel_lifts, panel_drags = [], []
            for j in range(len(clcd_all_in_one[i])):
                cl_2d, cd_2d = clcd_all_in_one[i][j][1], clcd_all_in_one[i][j][2]
                dynamic_pressure = 0.5 * self.air_density * apperent_wind_speeds[int(j * calculation_gap)] ** 2
                if j == 0:
                    panel_area = np.sum(self.panel_areas[:int(2 * self.refinement_level)])
                elif j == len(clcd_all_in_one[i]) - 1:
                    panel_area = np.sum(self.panel_areas[int(-2 * self.refinement_level):])
                else:
                    panel_area = np.sum(self.panel_areas[int(self.refinement_level * (4 * j - 2)): int(self.refinement_level * (4 * j + 2))])
                cl_3d = cl_2d / (1 + cl_2d / (np.pi * self.aspect_ratio))
                panel_lift = dynamic_pressure * panel_area * cl_3d
                cd_induced = cl_3d ** 2 / (np.pi * self.span_efficiency * self.aspect_ratio)
                panel_drag = dynamic_pressure * panel_area * (cd_2d + cd_induced)
                panel_lifts.append(panel_lift)
                panel_drags.append(panel_drag)
            wingsail_lift = np.sum(panel_lifts)
            wingsail_drag = np.sum(panel_drags)
            wingsail_thrust = wingsail_lift * np.sin(np.deg2rad(apperent_wind_angle)) - wingsail_drag * np.cos(np.deg2rad(apperent_wind_angle))
            wingsail_thrusts.append(wingsail_thrust)
            # print(f"[b dark_orange]Debug -[/b dark_orange] Wingsail thrust at AoA {i + 1} deg: {np.round(wingsail_thrust, 3)} N")

        # Find the maximum thrust and corresponding angle of attack
        max_thrust = np.max(wingsail_thrusts)
        max_thrust_index = np.argmax(wingsail_thrusts)
        max_angle_of_attack = full_aoa_range[max_thrust_index]

        return max_thrust, max_angle_of_attack

    # Wingsail thrust calculation with given sailing condition
    def max_thrust_with_xfoil(self, sailing_condition: list, alpha_range: list = [1, 25, 1], xfoil_iteration: int = 200) -> Tuple[float, float]:
        """
        Calculate the maximum thrust of the wingsail with given sailing condition.

        :param sailing_condition: The sailing condition list: [vessel speed, true wind speed, true wind angle]
        :param alpha_range:       The angle of attack range for xfoil calculation
        :param xfoil_iteration:   The iteration number for xfoil calculation
        :return:                  The maximum thrust and the corresponding angle of attack
        """
        # Import libraries
        import shutil
        from datetime import datetime

        # Unpackage sailing condition and calculate Reynolds number distribution
        vessel_speed, true_wind_speed, true_wind_angle = sailing_condition
        reynolds, apperent_wind_speeds, apperent_wind_angle = self._renv_distribution(vessel_speed, true_wind_speed, true_wind_angle)
        sailing_info = f"\nVessel Speed: {vessel_speed} m/s\nTrue Wind Speed: {true_wind_speed} m/s" \
                     + f"\nTrue Wind Angle: {true_wind_angle} deg\nApperent Wind Angle: {apperent_wind_angle} deg"
        
        # Get xfoil root path
        xfoil_dir = ConfigXfoil(self.xfoil_root).deploy()

        # Create temporary directory for xfoil data storage
        sail_name = "case_" + datetime.now().strftime("%H%M%S") + "_" + str(np.random.randint(10,99))
        sail_path = os.path.join(xfoil_dir, f".cache/{sail_name}")
        if os.path.exists(sail_path):
            shutil.rmtree(sail_path)
        else:
            os.makedirs(sail_path)

        # Record sail parameters and sailing condition to directory
        with open(os.path.join(sail_path, f"{sail_name}_parameters.txt"), 'w') as f:
            f.write(self.wingsail_id)
            f.write(sailing_info)

        # Execute xfoil calculation for each section
        calculation_gap = int(self.refinement_level * 4)
        involved_section_num = int(self.class_geometrisation.panel_num / calculation_gap + 1) # 26 sections are involved in cl, cd calculation
        aoa_all_sections, cl_all_sections, cd_all_sections = [], [], []
        for i in range(involved_section_num):
            section_index = int(i * calculation_gap)
            section_name = f"sec_{(section_index + 1):03}"
            section_dir = os.path.join(sail_path, section_name)
            os.makedirs(section_dir)
            np.savetxt(os.path.join(section_dir, f"{section_name}.dat"), self.section_reorganised_coords[section_index], fmt='%1.7f')
            reynold = reynolds[section_index]
            aoas, cls, cds = self._call_xfoil(section_name, section_dir, xfoil_dir, reynold, alpha_range, xfoil_iteration, pane=True)
            aoa_all_sections.append(aoas)
            cl_all_sections.append(cls)
            cd_all_sections.append(cds)

        # Post-process xfoil output data
        clcd_all_in_one, full_aoa_range = self._xfoil_postprocess(aoa_all_sections, cl_all_sections, cd_all_sections, alpha_range, involved_section_num)

        # Calculate thrust for each angle of attack
        wingsail_thrusts = []
        for i in range(len(clcd_all_in_one)):
            panel_lifts, panel_drags = [], []
            for j in range(len(clcd_all_in_one[i])):
                cl_2d, cd_2d = clcd_all_in_one[i][j][1], clcd_all_in_one[i][j][2]
                dynamic_pressure = 0.5 * self.air_density * apperent_wind_speeds[int(j * calculation_gap)] ** 2
                if j == 0:
                    panel_area = np.sum(self.panel_areas[:int(2 * self.refinement_level)])
                elif j == len(clcd_all_in_one[i]) - 1:
                    panel_area = np.sum(self.panel_areas[int(-2 * self.refinement_level):])
                else:
                    panel_area = np.sum(self.panel_areas[int(self.refinement_level * (4 * j - 2)): int(self.refinement_level * (4 * j + 2))])
                cl_3d =  cl_2d / (1 + cl_2d / (np.pi * self.aspect_ratio))
                panel_lift = dynamic_pressure * panel_area * cl_3d
                cd_induced = cl_3d ** 2 / (np.pi * self.span_efficiency * self.aspect_ratio)
                panel_drag = dynamic_pressure * panel_area * (cd_2d + cd_induced)
                panel_lifts.append(panel_lift)
                panel_drags.append(panel_drag)
            wingsail_lift = np.sum(panel_lifts)
            wingsail_drag = np.sum(panel_drags)
            wingsail_thrust = wingsail_lift * np.sin(np.deg2rad(apperent_wind_angle)) - wingsail_drag * np.cos(np.deg2rad(apperent_wind_angle))
            wingsail_thrusts.append(wingsail_thrust)
            # print(f"[b dark_orange]Debug -[/b dark_orange] Wingsail thrust at AoA {i + 1} deg: {np.round(wingsail_thrust, 3)} N")

        # Find the maximum thrust and corresponding angle of attack
        max_thrust = np.max(wingsail_thrusts)
        max_thrust_index = np.argmax(wingsail_thrusts)
        max_angle_of_attack = full_aoa_range[max_thrust_index]

        return max_thrust, max_angle_of_attack

    # Xfoil result post-process and data reorganisation
    def _xfoil_postprocess(self, aoa_all_sections: list, cl_all_sections: list, cd_all_sections: float,
                          alpha_range: list, involved_section_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refill xfoil unconverged data with cubic interpolation method.
        
        :param aoa_all_sections:        The xfoil converged angle of attack range of each section
        :param cl_all_sections:         The xfoil converged lift coefficient result of each section
        :param cd_all_sections:         The xfoil converged drag coefficient result of each section
        :param alpha_range:             The alpha range need to be reorganised
        :param involved_section_num:    The number of sections involved in the calculation
        :return:                        The reorganised aoa, cl and cd data array, default shape: (25, 26, 3)
        """
        # Import libraries
        from scipy import interpolate
        from scipy.signal import savgol_filter

        # Verify list alignment and initialise output array
        assert len(aoa_all_sections) == len(cl_all_sections) == len(cd_all_sections), "Xofil output cl and cd lists not aligned"
        full_aoa_range = np.arange(alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2])
        aoa_num = len(full_aoa_range)
        reorganised_result = np.zeros((aoa_num, involved_section_num, 3))

        # Interpolate and smooth the data
        for i in range(involved_section_num):
            aoa_one_section = aoa_all_sections[i]
            cl_one_section = cl_all_sections[i]
            cd_one_section = cd_all_sections[i]
            # Consign converged cl and cd data
            converged_flag = np.isin(full_aoa_range, aoa_one_section)
            cl_register = np.zeros(25)
            cd_register = np.zeros(25)
            cl_register[converged_flag] = cl_one_section
            cd_register[converged_flag] = cd_one_section
            # Interpolate missing cl and cd data
            if len(aoa_one_section) != aoa_num:
                f_cl = interpolate.interp1d(aoa_one_section, cl_one_section, kind='cubic', fill_value='extrapolate')
                f_cd = interpolate.interp1d(aoa_one_section, cd_one_section, kind='cubic', fill_value='extrapolate')
                cl_register[~converged_flag] = f_cl(full_aoa_range[~converged_flag])
                cl_register[~converged_flag] = f_cd(full_aoa_range[~converged_flag])
                # Smooth the interpolation
                smoothed_cl = savgol_filter(cl_register, window_length = 5, polyorder = 2)
                smoothed_cd = savgol_filter(cd_register, window_length = 5, polyorder = 2)
                # Refill original data
                smoothed_cl[converged_flag] = cl_one_section
                smoothed_cd[converged_flag] = cd_one_section
                reorganised_result[:, i, 1] = np.round(smoothed_cl, 4)
                reorganised_result[:, i, 2] = np.round(smoothed_cd, 5)
            else:
                reorganised_result[:, i, 1] = cl_one_section
                reorganised_result[:, i, 2] = cd_one_section
            reorganised_result[:, i, 0] = full_aoa_range
                
        return reorganised_result, full_aoa_range

    # Xfoil call and execution
    def _call_xfoil(self, airfoil_section: str, section_dir: str, xfoil_dir: str, reynold: float = 8e6,
            alpha_range: list = [1, 25, 1], iter: str = 100, pane = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Call and execute Xfoil program with given parameters and return the lift and drag coefficient.

        :param airfoil_section: The section name of the airfoil
        :param section_dir:    The path to the section data file
        :param xfoil_dir:       The path to the xfoil executable program
        :param reynold:         The Reynolds number of the airfoil
        :param alpha_range:     The alpha range for the calculation
        :param iter:            The number of iteration for the calculation
        :param pane:            True to calculate the pane effect
        :return:                The angle of attack, lift coefficient and drag coefficient
        """
        # Import libraries
        import subprocess
        import platform

        # Unpackage alpha range
        alpha_init, alpha_fini, alpha_step = alpha_range

        # Construct full paths for Python operations
        full_airfoil_path = os.path.join(section_dir, f"{airfoil_section}.dat")
        full_result_path = os.path.join(section_dir, f"result_{airfoil_section}.dat")
        full_inject_path = os.path.join(xfoil_dir, "inject_command.in")

        # Construct relative paths for Xfoil
        rel_airfoil_path = os.path.relpath(full_airfoil_path, xfoil_dir)
        rel_result_path = os.path.relpath(full_result_path, xfoil_dir)
        if platform.system() == 'Windows':
            xfoil_exe = os.path.join(xfoil_dir, "xfoil.exe")
        elif platform.system() == 'Linux':
            xfoil_exe = os.path.join(xfoil_dir, "xfoil")
        else:
            raise NotImplementedError("Xfoil execution only supported on Windows and Linux")
        
        # Program xfoil injection file
        with open(full_inject_path, 'w') as inject_file:
            inject_file.write("LOAD\n")
            inject_file.write(f"{rel_airfoil_path}\n")
            inject_file.write(f"{airfoil_section}\n")
            if pane:
                inject_file.write("PANE\n")
            inject_file.write("OPER\n")
            inject_file.write(f"V {reynold}\n")
            inject_file.write("PACC\n")
            inject_file.write(f"{rel_result_path}\n\n")
            inject_file.write(f"ITER {iter}\n")
            inject_file.write(f"AS {alpha_init} {alpha_fini} {alpha_step}\n")
            inject_file.write("\n\n")
            inject_file.write("QUIT\n")

        # Run xfoil without GUI
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        # Change working directory to Xfoil directory before running
        original_dir = os.getcwd()
        os.chdir(xfoil_dir)
        try:
            with open("inject_command.in", "r") as infile:
                process = subprocess.Popen(
                    xfoil_exe,
                    stdin=infile,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    startupinfo=startupinfo,
                    text=True
                )
                _, _ = process.communicate()
            # Activate the following code for debugging
            # print(f"Debug - Xfoil stdout:\n{stdout}")
            # print(f"Debug - Xfoil stderr:\n{stderr}")
            # Extract data
            if os.path.exists(full_result_path):
                xfoil_result = np.loadtxt(full_result_path, skiprows=12)
                angle_of_attack = xfoil_result[:, 0]
                lift_coefficient = xfoil_result[:, 1]
                drag_coefficient = xfoil_result[:, 2]
                return angle_of_attack, lift_coefficient, drag_coefficient
            else:
                print(f":warning: Result file not found: {full_result_path}")
                return None, None, None
        except Exception as e:
            print(f":warning: Exception occurred: {str(e)}")
            return None, None, None
        finally:
            # Change back to original directory
            os.chdir(original_dir)

    def debug(self):
        print(f"\n[b dark_orange]Debug -[/b dark_orange] This function is reserved for calculation class debugging\n")
        pass

class ConfigXfoil:
    """
    Class XfoilConfig: Configurate the Xfoil execution.

    :method deploy: Deploy the xfoil executable program to the working directory (only for Windows)
    :method remove: Remove the xfoil executable program from the working directory
    """
    # Xfoil configuration
    def __init__(self, install_root: str = "./xfoil"):
        import platform
        if install_root == "./xfoil" or install_root == "/xfoil" or install_root == "xfoil":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.deploy_dir = os.path.join(current_dir, "xfoil")
        else:
            self.deploy_dir = install_root
        if platform.system() == 'Windows':
            self.xfoil_url = 'https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip'
        else:
            self.xfoil_url = 'https://web.mit.edu/drela/Public/web/xfoil/xfoil6.97.tar.gz'

    # Xfoil download and deployment
    def deploy(self):
        import requests
        import zipfile
        from io import BytesIO
        import platform

        # Create xfoil deployment directory
        if os.path.exists(self.deploy_dir):
            if os.path.exists(os.path.join(self.deploy_dir, "xfoil.exe")):
                xfoil_path = os.path.join(self.deploy_dir, "xfoil.exe")
                print(f"\n:white_check_mark: Xfoil program existed: [b light_green]{self.deploy_dir}[/b light_green]")
            elif os.path.exists(os.path.join(self.deploy_dir, "xfoil")):
                xfoil_path = os.path.join(self.deploy_dir, "xfoil")
                print(f"\n:white_check_mark: Xfoil program existed: [b light_green]{self.deploy_dir}[/b light_green]")
        else:
            os.makedirs(self.deploy_dir, exist_ok=True)
            print(f"\n:file_folder: Deploying xfoil program to path: [b light_green]{self.deploy_dir}[/b light_green]")

            # Download xfoil
            response = requests.get(self.xfoil_url)
            response.raise_for_status()
            if self.xfoil_url.endswith(".zip"):
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(self.deploy_dir)
            elif self.xfoil_url.endswith(".tar.gz"):
                import tarfile
                with tarfile.open(fileobj=BytesIO(response.content)) as tar_ref:
                    tar_ref.extractall(self.deploy_dir)
            else:
                raise NotImplementedError("Xfoil download failed!")
            
            # Check file availability and compile for Linux
            if platform.system() == 'Windows':
                xfoil_path = os.path.join(self.deploy_dir, "xfoil.exe")
                print(f":inbox_tray: Xfoil has been deployed at: [b light_green]{xfoil_path}[/b light_green]\n")
            elif platform.system() == 'Linux':
                self._compile_xfoil(self.deploy_dir)
            else:
                xfoil_path = None
                raise NotImplementedError("Xfoil call funtion only supported on Windows and Linux")

        return self.deploy_dir

    # Xfoil compilation for Linux
    def _compile_xfoil(source_dir):
        raise NotImplementedError(f"Please deploy Xfoil manually on Linux to: [b light_green]{source_dir}[/b light_green]\n")

    # Xfoil uninstallation
    def remove(self):
        import shutil
        if os.path.exists(self.deploy_dir):
            shutil.rmtree(self.deploy_dir)
            print(f":boom: Xfoil has been removed from: [b light_green]{self.deploy_dir}[/b light_green]\n")
        else:
            print(f":warning: [b red3]Xfoil folder not found, please check manually[/b red3]\n")

if __name__ == '__main__':
    # Import libraries
    import time
    import numpy as np

    t = time.perf_counter()

    # Sample wingsail parameters
    sample_wingsail_0 = [[40, 10, 8, 15, 0.3, 1, 15, 0, 0.5, 1, 15, 0, 0.8, 1, 15, 0, 1, 15, 0]]                        # Rectangular wingsail for calculation
    sample_wingsail_1 = [[40, 10, 8, 15, 0.3, 1, 15, 0, 0.6, 1, 15, 0, 0.8, 0.9, 12, 0, 0.7, 10, 0]]                    # Simplified tapered wingsail for calculation
    sample_wingsail_2 = [[20, 2, 4, 21, 0.55, 1.05, 19, 0, 0.8, 0.9, 18, 0, 0.92, 0.7, 15, 0.10, 0.4, 10, 0.4]]         # Wingsail with minimum allowance size
    sample_wingsail_3 = [[80, 30, 25, 24, 0.45, 1.1, 22, -0.05, 0.75, 0.95, 20, 0, 0.9, 0.72, 18, 0.05, 0.5, 17, 0.18]] # Wingsail with maximum allowance size
    oracle_ac72 = [[40, 3, 9, 20, 0.26, 1.03, 17, -0.15, 0.52, 0.96, 15, -0.2, 0.78, 0.76, 15, -0.15, 0.38, 12, 0.06]]  # Oracle AC72 wingsail

    # Nagative wingsail parameters
    # nagative_dataset = np.loadtxt("nagative_dataset.csv", delimiter=",")

    # # Import wingsail parameters from file
    # dataset = np.loadtxt("wingsail_dataset_3k.csv", delimiter=",")
    # dataset = dataset[0:20]
    # debugging_datapoint = dataset[10]
    # print(f"Debugging parameters: {debugging_datapoint}\n")

    APPLIED_WINGSAIL_DATAPOINT = sample_wingsail_1[0]

    ######################################################################
    ####### Release to evaluate wingsail geometry trustworthiness ########
    ######################################################################
    # trustworthiness_array = []
    # for i in range(len(APPLIED_WINGSAIL_DATAPOINT)):
    #     wincal = PreCalculation(APPLIED_WINGSAIL_DATAPOINT[i])
    #     trustworthiness = wincal.solve_wingsail_trustworthiness()
    #     trustworthiness_array.append(trustworthiness)
    # print(f"\nWingsail trustworthiness: {trustworthiness_array}\n")
    ######################################################################

    #####################################################################
    ########### Release to find maximum thrust and best AOA #############
    #####################################################################
    # Air kinematic viscosity, air density, sea surface roughness, vessel speed range, TWS range, TWA range
    PHYSICAL_CONSTANTS = {'sea_surface_roughness'  : 2e-4,
                          'air_density'            : 1.205,
                          'air_kinematic_viscosity': 15.06e-6}
    SAILING_CONDITION = [7, 8, 50]
    AOA_RANGE = [1, 25, 1]
    XFOIL_ITER = 200
    REFINEMENT_LEVEL = 1
    # Calculate the maximum thrust of the wingsail
    wincal = PreCalculation(APPLIED_WINGSAIL_DATAPOINT, PHYSICAL_CONSTANTS, REFINEMENT_LEVEL) # For PreCalculation class test
    # Choose the calculation method (either neuralfoil or xfoil)
    max_thrust, max_angle_of_attack = wincal.max_thrust_with_neuralfoil(SAILING_CONDITION, AOA_RANGE, "xlarge")
    # max_thrust, max_angle_of_attack = wincal.max_thrust_with_xfoil(SAILING_CONDITION, AOA_RANGE, XFOIL_ITER)
    print(f"\n:sailboat: Vessel speed: {SAILING_CONDITION[0]}")
    print(f":triangular_flag_on_post: Wind condition: {SAILING_CONDITION[1]} m/s at {SAILING_CONDITION[2]} deg")
    print(f":smiley: Max thrust of {np.round(max_thrust, 3)} N found at angle of attack: {max_angle_of_attack} deg\n")
    #####################################################################

    print(f":clock1: Cost time: {time.perf_counter() - t:8f} seconds!\n")