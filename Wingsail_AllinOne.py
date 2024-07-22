from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import os

class Geometrisation:
    # Data cleansing and preprocessing
    def __init__(self, wingsail_parameters: list, refinement_level: int = 1, uniform_mesh: bool = True):
        """
        Wingsail parameters geometrisation, model generation and visualization.

        :param wingsail_parameters: Wingsail parameters list
        :param refinement_level:    Refinement level of the geometry, default as 1
        :param uniform_mesh:        Bool for uniform the mesh to 100 panels * refinement level, default as True
        """
        # Verify input parameters
        if not len(wingsail_parameters) == 19:
            raise TypeError("wingsail_parameters: Should have 19 parameters as input")
        if not isinstance(refinement_level, int) or refinement_level < 1 or refinement_level > 9:
            raise TypeError("refinement_level: Have to be an integer and in the range of 1 to 9")
        
        # Unpackage wingsail parameters
        self.span         = wingsail_parameters[0]
        self.clearance_ow = wingsail_parameters[1]
        self.chord_bot    = wingsail_parameters[2]
        self.naca_bot     = wingsail_parameters[3]
        self.height_low   = wingsail_parameters[4] * self.span
        self.chord_low    = wingsail_parameters[5] * self.chord_bot
        self.naca_low     = wingsail_parameters[6]
        self.offset_low   = wingsail_parameters[7]
        self.height_mid   = wingsail_parameters[8] * self.span
        self.chord_mid    = wingsail_parameters[9] * self.chord_bot
        self.naca_mid     = wingsail_parameters[10]
        self.offset_mid   = wingsail_parameters[11]
        self.height_upp   = wingsail_parameters[12] * self.span
        self.chord_upp    = wingsail_parameters[13] * self.chord_bot
        self.naca_upp     = wingsail_parameters[14]
        self.offset_upp   = wingsail_parameters[15]
        self.chord_tip    = wingsail_parameters[16] * self.chord_bot
        self.naca_tip     = wingsail_parameters[17]
        self.offset_tip   = wingsail_parameters[18]

        # Pre-process parameter optional
        self.uniform_bool = uniform_mesh
        if self.uniform_bool:
            self.height_low = round(wingsail_parameters[4], 2) * self.span
            self.height_mid = round(wingsail_parameters[8], 2) * self.span
            self.height_upp = round(wingsail_parameters[12], 2) * self.span
            print(f":wrench: [deep_sky_blue1]Adjustment:[/deep_sky_blue1] Number of geometry layer has uniformed to {str(100 * refinement_level)}.")
            print(":wrench: [deep_sky_blue1]Adjustment:[/deep_sky_blue1] All sections height are rounded to 2 decimal places.")

        # Prepare characteristic parameters
        self.key_section_heights  = np.array([0, self.height_low, self.height_mid, self.height_upp, self.span])
        self.key_section_chords   = np.array([self.chord_bot, self.chord_low, self.chord_mid, self.chord_upp, self.chord_tip])
        self.refinement_level     = refinement_level
        self.panel_num            = self._solve_section()
        self.profile_bottom       = self._naca_sketch(self.naca_bot)
        self.profile_low          = self._naca_sketch(self.naca_low, self.offset_low)
        self.profile_mid          = self._naca_sketch(self.naca_mid, self.offset_mid)
        self.profile_upp          = self._naca_sketch(self.naca_upp, self.offset_upp)
        self.profile_tip          = self._naca_sketch(self.naca_tip, self.offset_tip)
        self.key_section_profiles = np.array([self.profile_bottom, self.profile_low, self.profile_mid, self.profile_upp, self.profile_tip]) \
                                    * self.key_section_chords[:, np.newaxis, np.newaxis]
        
        # Arrange geometry axes coordinates
        self.section_x_coords, self.section_y_coords, self.section_z_coords, self.section_2d_coords = self._geometry_coordinates()

    # NACA airfoil coordinates preparation
    def _naca_sketch(self, naca_code: int, x_offset: float = 0.0) -> np.ndarray:
        """
        Coordinates builder for symmetrical 4-digit NACA airfoil profile.

        :param naca_code: The last two digits of symmetrical 4-digit NACA airfoile
        :param x_offset:  The section offset scale versus chord on x coordinates
        :return:          The x, y coordinates of the airfoil profile
        """
        # Generate non-linear x distribution for better resolution on leading and trialing edge
        if self.uniform_bool:
            datapoint_density = 50
        else:
            datapoint_density = int(self.panel_num / 5)
            if datapoint_density < 10:
                datapoint_density = 10
        beta = np.linspace(0, np.pi, self.refinement_level * datapoint_density)
        x = (1 - np.cos(beta)) / 2
        x = np.round((1 - np.power(1 - np.power(x, 1.5), 1.2)), 7)

        # Calculate y coordinates based on: https://ntrs.nasa.gov/api/citations/19930091108/downloads/19930091108.pdf Page 4
        yt = 0.05 * naca_code * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        # Accumulate x, y coordinates
        x += x_offset
        y_upper = np.column_stack((x, yt))
        y_lower = np.column_stack((x[::-1], -yt[::-1]))
        trailing_radius = yt[-1]

        # Polish trailing edge coordinates
        x_trailing = np.array([(1 + trailing_radius * 0.5 + x_offset), (1 + trailing_radius * 0.866 + x_offset),
                        (1 + trailing_radius + x_offset), (1 + trailing_radius * 0.866 + x_offset),
                        (1 + trailing_radius * 0.5 + x_offset)])
        y_trailing = np.array([(-0.866 * trailing_radius), (-0.5 * trailing_radius), 0.0, (0.5 * trailing_radius), (0.866 * trailing_radius)])
        trailing_polisher = np.column_stack((x_trailing, y_trailing))

        # Package airfoil coordinates
        airfoil_coordinates =  np.vstack((np.flip(y_upper, axis=0)[:-1], np.array([x_offset, 0.0]), np.flip(y_lower, axis=0)[1:], trailing_polisher))

        return airfoil_coordinates

    # Model required sections calculation
    def _solve_section(self)-> int:
        """
        Calculate geometry panels number and section spacing ratio based on heights and refinement level.

        :return: The number of subdivided panels
        """
        if self.uniform_bool:
            panel_num = 100 * self.refinement_level
        else:
            spacing_deltas = np.round(np.diff(self.key_section_heights), 15)
            shaping_factor = 10 ** max(len(str(delta).split('.')[-1]) for delta in spacing_deltas)
            unprocessed_spacings = (spacing_deltas * shaping_factor).astype(int)                              # Section spacings in integer
            spacing_ratio = [ratio // np.gcd.reduce(unprocessed_spacings) for ratio in unprocessed_spacings]  # Section spacing ratio with GCD = 1
            panel_num = np.sum(spacing_ratio) * self.refinement_level
        
        return panel_num

    # Section x, y coordinates interpolation
    def _coordinates_pchip(self, x_keys: np.ndarray, y_keys: np.ndarray,
                        z_keys: np.ndarray, z_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        A parametric PCHIP interpolation function, returns the interpolated x and y coordinates for the given dense z.

        :param x_keys: The user defined key x coordinates of the datapoints
        :param y_keys: The user defined key y coordinates of the datapoints
        :param z_keys: The user defined key z coordinates of the datapoints
        :param z_all:  The z coordinates (sections height) of the complete datapoints to interpolate
        :return:       The interpolated x and y coordinates
        """
        from scipy import interpolate
        x_interp = interpolate.PchipInterpolator(z_keys, x_keys)
        y_interp = interpolate.PchipInterpolator(z_keys, y_keys)
        x_interpolated = x_interp(z_all)
        y_interpolated = y_interp(z_all)

        return x_interpolated, y_interpolated

    # Geometry 3D coordinates generation and reorganisation
    def _geometry_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Arrange the x, y, z coordinates tuple based on the input parameters and repack into 3D array.

        :return: The x, y, z coordinates of the geometry and repacked 3D array
        """
        # Prepare section coordinates
        section_x_coords = []
        section_y_coords = []
        section_z_heights = np.linspace(0, 1, self.panel_num + 1) * self.span
        key_section_x_coords = self.key_section_profiles[:, :, 0].T
        key_section_y_coords = self.key_section_profiles[:, :, 1].T

        # Interpolate x, y coordinates for each section
        for i in range(key_section_x_coords.shape[0]):
            x_interpolated, y_interpolated = self._coordinates_pchip(key_section_x_coords[i], key_section_y_coords[i],
                                                                    self.key_section_heights, section_z_heights)
            section_x_coords.append(x_interpolated)
            section_y_coords.append(y_interpolated)
        section_x_coords = np.array(section_x_coords)
        section_y_coords = np.array(section_y_coords)
        section_z_coords = np.array([section_z_heights] * section_x_coords.shape[0])

        # Reorganise coordinates       
        section_2d_coords = np.zeros((section_x_coords.shape[1], section_x_coords.shape[0], 2))
        for layer in range(section_x_coords.shape[1]):
            for point in range(section_x_coords.shape[0]):
                section_2d_coords[layer, point, 0] = section_x_coords[point, layer]
                section_2d_coords[layer, point, 1] = section_y_coords[point, layer]

        return section_x_coords, section_y_coords, section_z_coords, section_2d_coords

    # STL file generation
    def generate_stl(self, path: str = 'Untitled_Geometry.stl', add_lid: bool = True):
        """
        Generate a closed STL file from the given coordinates with optional top and bottom caps.
        
        :param path:    The path to save the generated 3D model
        :param add_lid: True to add cap and bottom to the 3D model
        """
        from stl import mesh
        # Perpare triangular faces property
        x_coords, y_coords, z_coords = self.section_x_coords, self.section_y_coords, self.section_z_coords
        datapoint_num, layer_num = x_coords.shape
        side_faces = datapoint_num * (layer_num - 1) * 2
        lid_faces = datapoint_num * 2 if add_lid else 0
        total_faces = side_faces + lid_faces

        # Instantiation geometry mesh function from numpy-stl
        geometry = mesh.Mesh(np.zeros(total_faces, dtype=mesh.Mesh.dtype))

        # Assign side face vectors
        k = 0
        for i in range(datapoint_num):
            for j in range(layer_num - 1):
                i_new = (i + 1) % datapoint_num # Determine point index, return to 0 at the last point
                # First triangle
                geometry.vectors[k][0] = [x_coords[i, j], y_coords[i, j], z_coords[i, j]]
                geometry.vectors[k][1] = [x_coords[i_new, j], y_coords[i_new, j], z_coords[i_new, j]]
                geometry.vectors[k][2] = [x_coords[i, j + 1], y_coords[i, j + 1], z_coords[i, j + 1]]
                k += 1
                # Second triangle
                geometry.vectors[k][0] = [x_coords[i_new, j], y_coords[i_new, j], z_coords[i_new, j]]
                geometry.vectors[k][1] = [x_coords[i_new, j + 1], y_coords[i_new, j + 1], z_coords[i_new, j + 1]]
                geometry.vectors[k][2] = [x_coords[i, j + 1], y_coords[i, j + 1], z_coords[i, j + 1]]
                k += 1

        # Assign tip and bottom face vectors
        if add_lid:
            for i in range(datapoint_num):
                i_new = (i + 1) % datapoint_num
                geometry.vectors[k][0] = [x_coords[0, 0], y_coords[0, 0], z_coords[0, 0]]    # Bottom center point
                geometry.vectors[k][1] = [x_coords[i, 0], y_coords[i, 0], z_coords[i, 0]]
                geometry.vectors[k][2] = [x_coords[i_new, 0], y_coords[i_new, 0], z_coords[i_new, 0]]
                k += 1
            for i in range(datapoint_num):
                i_new = (i + 1) % datapoint_num
                geometry.vectors[k][0] = [x_coords[0, -1], y_coords[0, -1], z_coords[0, -1]] # Tip center point
                geometry.vectors[k][1] = [x_coords[i_new, -1], y_coords[i_new, -1], z_coords[i_new, -1]]
                geometry.vectors[k][2] = [x_coords[i, -1], y_coords[i, -1], z_coords[i, -1]]
                k += 1

        # Save the geometry mesh to STL file
        print(f"Datapoints per section: {x_coords.shape[0]}\nNumber of sections    : {x_coords.shape[1]}\nNumber of triangles   : {total_faces}")
        geometry.save(path)
        print(f"\n:floppy_disk: Geometry STL file saved to: [b light_green]{path}[/b light_green]")

    # Geometry 3D visualization with matplotlib
    def plot3d(self):
        """
        Plot point cloud to 3D axes.
        """
        # Initialize 3D plot function
        x_coords, y_coords, z_coords = self.section_x_coords, self.section_y_coords, self.section_z_coords
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Paint closed sections
        for i in range(z_coords.shape[1]):
            x = np.append(x_coords[:, i], x_coords[0, i])
            y = np.append(y_coords[:, i], y_coords[0, i])
            z = np.append(z_coords[:, i], z_coords[0, i])
            ax.plot(x, y, z, color='gray', alpha=0.5)

        # Paint vertical edge
        for i in range(x_coords.shape[0]):
            ax.plot(x_coords[i, :], y_coords[i, :], z_coords[i, :], color='blue', alpha=0.5)

        # Plot 3D scatter
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', s=20)

        # Plot embellishments
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Visualization of Interpolated Points with Closed Polygons')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Z coordinate')
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

class Calculation:
    # Data cleansing and preprocessing
    def __init__(self, wingsail_parameters: list, physical_constants: tuple = (), refinement_level: int = 1, xfoil_root: str = 'xfoil'):
        """
        Wingsail performance calculation with given conditions.

        :param wingsail_parameters: Wingsail parameters list
        :param physical_constants:  Physical constants list
        :param refinement_level:    Refinement level of the geometry, default as 1
        :param xfoil_root:          The path to the xfoil executable program
        """
        # Verify input parameters
        if not isinstance(refinement_level, int) or refinement_level < 1 or refinement_level > 9:
            raise TypeError("refinement_level: Have to be an integer and in the range of 1 to 9")
        if len(physical_constants) != 19:
            print(f":construction: [bright_yellow]Notification:[/bright_yellow] Input physical constants incomplete, reset to default values.")
            physical_constants = (2e-4, 1.205, 15.06e-6, 0.85, [5, 10, 1], [5, 10, 1], [30, 150, 10]) # Setup default physical constants
        
        # Define initial physical constants
        self.sea_surface_roughness   = physical_constants[0]
        self.air_density             = physical_constants[1]
        self.air_kinematic_viscosity = physical_constants[2]
        self.span_effciency          = physical_constants[3]
        self.vessel_speed_range      = physical_constants[4]
        self.true_wind_speed_range   = physical_constants[5]
        self.true_wing_angle_range   = physical_constants[6]

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
        self.section_chords, self.section_reorganised_coords, self.panel_areas, self.aspect_ratio = self.solve_wing_spec()

    # Wingsail geometry sepcification calculation
    def solve_wing_spec(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        '''
        Calculate the chord length of each section, total wingsail project area and aspect ratio with 2D coordinates.

        :return: The chord length and project area of each section, wingsail total area and aspect ratio
        '''
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

        # Calculate aspect ratio
        aspect_ratio = self.span ** 2 / project_area

        return np.array(section_chords), np.array(section_reorganised_coords), np.array(panel_areas), aspect_ratio

    # Reynolds number and wind speed distribution calculation
    def renv_distribution(self, vessel_speed: float, true_wind_speed: float, true_wind_angle: float) -> Tuple[np.ndarray, np.ndarray, float]:
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
        reynolds, apperent_wind_speeds, apperent_wind_angle = self.renv_distribution(vessel_speed, true_wind_speed, true_wind_angle)
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
            cl_all_sections.append(neuralfoil_output["CL"])
            cd_all_sections.append(neuralfoil_output["CD"])
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
                cl_3d = cl_2d / (1 + 2 / (self.span_effciency * self.aspect_ratio))
                panel_lift = dynamic_pressure * panel_area * cl_3d
                cd_induced = cl_3d ** 2 / (np.pi * self.span_effciency * self.aspect_ratio)
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
        reynolds, apperent_wind_speeds, apperent_wind_angle = self.renv_distribution(vessel_speed, true_wind_speed, true_wind_angle)
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
            aoas, cls, cds = self.call_xfoil(section_name, section_dir, xfoil_dir, reynold, alpha_range, xfoil_iteration, pane=True)
            aoa_all_sections.append(aoas)
            cl_all_sections.append(cls)
            cd_all_sections.append(cds)

        # Post-process xfoil output data
        clcd_all_in_one, full_aoa_range = self.xfoil_postprocess(aoa_all_sections, cl_all_sections, cd_all_sections, alpha_range, involved_section_num)

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
                cl_3d = cl_2d / (1 + 2 / (self.span_effciency * self.aspect_ratio))
                panel_lift = dynamic_pressure * panel_area * cl_3d
                cd_induced = cl_3d ** 2 / (np.pi * self.span_effciency * self.aspect_ratio)
                panel_drag = dynamic_pressure * panel_area * (cd_2d + cd_induced)
                panel_lifts.append(panel_lift)
                panel_drags.append(panel_drag)
            wingsail_lift = np.sum(panel_lifts)
            wingsail_drag = np.sum(panel_drags)
            wingsail_thrust = wingsail_lift * np.sin(np.deg2rad(apperent_wind_angle)) - wingsail_drag * np.cos(np.deg2rad(apperent_wind_angle))
            wingsail_thrusts.append(wingsail_thrust)
            print(f"[b dark_orange]Debug -[/b dark_orange] Wingsail thrust at AoA {i + 1} deg: {np.round(wingsail_thrust, 3)} N")

        # Find the maximum thrust and corresponding angle of attack
        max_thrust = np.max(wingsail_thrusts)
        max_thrust_index = np.argmax(wingsail_thrusts)
        max_angle_of_attack = full_aoa_range[max_thrust_index]

        return max_thrust, max_angle_of_attack

    # Xfoil result post-process and data reorganisation
    def xfoil_postprocess(self, aoa_all_sections: list, cl_all_sections: list, cd_all_sections: float,
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
    def call_xfoil(self, airfoil_section: str, section_dir: str, xfoil_dir: str, reynold: float = 8e6,
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
            xfoil_exe = os.path.join(xfoil_dir, 'xfoil.exe')
        elif platform.system() == 'Linux':
            xfoil_exe = os.path.join(xfoil_dir, 'xfoil')
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
        print(f"\n[b dark_orange]Debug -[/b dark_orange] This function is reserved for debugging.")
        return 

# Further development plan
# class Prediction:
# class Verification:

class ConfigXfoil:
    # Xfoil configuration
    def __init__(self, install_root: str = './xfoil'):
        import platform
        if install_root == './xfoil' or install_root == '/xfoil' or install_root == 'xfoil':
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.deploy_dir = os.path.join(current_dir, 'xfoil')
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
            if os.path.exists(os.path.join(self.deploy_dir, 'xfoil.exe')):
                xfoil_path = os.path.join(self.deploy_dir, 'xfoil.exe')
                print(f"\n:white_check_mark: Xfoil program existed: [b light_green]{self.deploy_dir}[/b light_green]")
            elif os.path.exists(os.path.join(self.deploy_dir, 'xfoil')):
                xfoil_path = os.path.join(self.deploy_dir, 'xfoil')
                print(f"\n:white_check_mark: Xfoil program existed: [b light_green]{self.deploy_dir}[/b light_green]")
        else:
            os.makedirs(self.deploy_dir, exist_ok=True)
            print(f"\n:file_folder: Deploying xfoil program to path: [b light_green]{self.deploy_dir}[/b light_green]")

            # Download xfoil
            response = requests.get(self.xfoil_url)
            response.raise_for_status()
            if self.xfoil_url.endswith('.zip'):
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(self.deploy_dir)
            elif self.xfoil_url.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(fileobj=BytesIO(response.content)) as tar_ref:
                    tar_ref.extractall(self.deploy_dir)
            else:
                raise NotImplementedError("Xfoil download failed!")
            
            # Check file availability and compile for Linux
            if platform.system() == 'Windows':
                xfoil_path = os.path.join(self.deploy_dir, 'xfoil.exe')
                print(f":inbox_tray: Xfoil has been deployed at: [b light_green]{xfoil_path}[/b light_green]\n")
            elif platform.system() == 'Linux':
                self.compile_xfoil(self.deploy_dir)
            else:
                xfoil_path = None
                raise NotImplementedError("Xfoil call funtion only supported on Windows and Linux")

        return self.deploy_dir

    # Xfoil compilation for Linux
    def compile_xfoil(source_dir):
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
    from datetime import datetime
    import time
    t = time.perf_counter()

    # Export a sample wingsail geometry
    sample_wingsail_0 = [4, 0, 1, 15, 0.2, 1, 15, 0, 0.5, 1, 15, 0, 0.8, 1, 15, 0, 1, 15, 0]
    sample_wingsail_1 = [4, 0, 1, 15, 0.2, 1, 15, 0, 0.5, 1, 15, 0, 0.8, 1, 12, 0, 0.5, 10, 0]
    sample_wingsail_2 = [3.5, 1, 1, 21, 0.5, 1.05, 21, 0, 0.8, 0.8, 18, 0.05, 0.93, 0.6, 15, 0.2, 0.35, 10, 0.4]
    sample_wingsail_3 = [60, 30, 15, 22, 0.45, 0.95, 20, 0.02, 0.75, 0.8, 20, 0.05, 0.95, 0.66, 17, 0.09, 0.4, 12, 0.16]
    oracle_ac72 = [40, 3, 9, 20, 0.26, 1.03, 17, -0.15, 0.52, 0.96, 15, -0.21, 0.78, 0.76, 15, -0.15, 0.38, 12, 0.06]

    # Define physical constants
    SAILING_CONDITION = [7, 8, 50]
    AOA_RANGE = [1, 25, 1]
    XFOIL_ITER = 200
    REFINE_LEVEL = 1

    # Calculate the maximum thrust of the wingsail
    wincal = Calculation(oracle_ac72, [], REFINE_LEVEL) # For Calculation class test

    max_thrust, max_angle_of_attack = wincal.max_thrust_with_neuralfoil(SAILING_CONDITION, AOA_RANGE, "xlarge")
    # max_thrust, max_angle_of_attack = wincal.max_thrust_with_xfoil(SAILING_CONDITION, AOA_RANGE, XFOIL_ITER)

    print(f"\n:sailboat: Vessel speed: {SAILING_CONDITION[0]}")
    print(f":triangular_flag_on_post: Wind condition: {SAILING_CONDITION[1]} m/s at {SAILING_CONDITION[2]} deg")
    print(f":smiley: Max thrust of {np.round(max_thrust, 3)} N found at angle of attack: {max_angle_of_attack} deg\n")
    print(f":clock1: Cost time: {time.perf_counter() - t:8f} seconds!\n")

    """
    # wingsail = Geometrisation(sample_wingsail_3, 1, True) # For Geometirsation class test
    # # Generate the 3D model
    # refinement_level = wingsail.refinement_level
    # if wingsail.uniform_bool:
    #     output_uniform_str = "_Uniformed_"
    # else:
    #     output_uniform_str = "_"
    # output_filename = "Wingsail" + output_uniform_str + datetime.now().strftime("%y%m%d%H%M%S") + "_RL" + str(refinement_level) + ".stl"
    # save_path = fr"./{output_filename}"
    # wingsail.generate_stl(save_path)
    # # Plot the 3D model
    # wingsail.plot3d()
    """
