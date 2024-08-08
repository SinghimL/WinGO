import numpy as np
from rich import print
from typing import Tuple

class Geometrisation:
    """
    Class Geometrisation: Wingsail geometry cloud points buildup, model generation and visualization with the 19 parameters.

    :method generate_stl: Generate a closed STL file from the given parameters with optional top and bottom caps
    :method plot3d:       Plot point clouds to 3D axes with matlibplot
    """

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
            # print(f":wrench: [deep_sky_blue1]Adjustment:[/deep_sky_blue1] Number of geometry layer has uniformed to {str(100 * refinement_level)}, all sections height are rounded to 2 decimal places.")

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
    def generate_stl(self, path: str = "Untitled_Geometry.stl", add_lid: bool = True):
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
                geometry.vectors[k][0] = [x_coords[0, -1], y_coords[0, -1], z_coords[0, -1]]  # Tip center point
                geometry.vectors[k][1] = [x_coords[i_new, -1], y_coords[i_new, -1], z_coords[i_new, -1]]
                geometry.vectors[k][2] = [x_coords[i, -1], y_coords[i, -1], z_coords[i, -1]]
                k += 1

        # Save the geometry mesh to STL file
        # print(f"Datapoints per section: {x_coords.shape[0]}\nNumber of sections    : {x_coords.shape[1]}\nNumber of triangles   : {total_faces}")
        geometry.save(path)
        print(f"\n:floppy_disk: Geometry STL file saved to: [b light_green]{path}[/b light_green]")

    # Geometry 3D visualization with matplotlib
    def plot3d(self):
        """
        Plot point cloud to 3D axes.
        """
        import matplotlib.pyplot as plt

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
        plt.title("3D Visualization of Interpolated Points with Closed Polygons")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Z coordinate")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Import libraries
    import numpy as np
    from datetime import datetime

    # Sample wingsail parameters
    sample_wingsail_0 = [[40, 10, 8, 15, 0.3, 1, 15, 0, 0.5, 1, 15, 0, 0.8, 1, 15, 0, 1, 15, 0]]                        # Rectangular wingsail for calculation
    sample_wingsail_1 = [[40, 10, 8, 15, 0.3, 1, 15, 0, 0.6, 1, 15, 0, 0.8, 0.9, 12, 0, 0.7, 10, 0]]                    # Simplified tapered wingsail for calculation
    sample_wingsail_2 = [[20, 2, 4, 21, 0.55, 1.05, 19, 0, 0.8, 0.9, 18, 0, 0.92, 0.7, 15, 0.10, 0.4, 10, 0.4]]         # Wingsail with minimum allowance size
    sample_wingsail_3 = [[80, 30, 25, 24, 0.45, 1.1, 22, -0.05, 0.75, 0.95, 20, 0, 0.9, 0.72, 18, 0.05, 0.5, 17, 0.18]] # Wingsail with maximum allowance size
    oracle_ac72 = [[40, 3, 9, 20, 0.26, 1.03, 17, -0.15, 0.52, 0.96, 15, -0.2, 0.78, 0.76, 15, -0.15, 0.38, 12, 0.06]]  # Oracle AC72 wingsail

    APPLIED_WINGSAIL_DATAPOINT = sample_wingsail_1
    REFINEMENT_LEVEL = 1
    UNIFORM_BOOL = True

    for i in range(len(APPLIED_WINGSAIL_DATAPOINT)):
        wingsail = Geometrisation(APPLIED_WINGSAIL_DATAPOINT[i], REFINEMENT_LEVEL, UNIFORM_BOOL) # For Geometirsation class test

        # Generate the 3D model
        output_filename = "Debugging_" + datetime.now().strftime("%m%d%H%M%S") + str(np.random.randint(10,99)) + "_RL" + str(REFINEMENT_LEVEL) + ".stl"
        save_path = fr"./{output_filename}"
        wingsail.generate_stl(save_path)

        # # Plot the 3D model
        # wingsail.plot3d()