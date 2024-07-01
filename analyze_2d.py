import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pickle
import os
from scipy.fft import fft2, ifft2, fftfreq

# Find maximum time step in the run directory
def find_max_time(run_dir):
    # List all the VTK files contained in the run directory
    vtk_files = [vtk_file for vtk_file in os.listdir(run_dir) if vtk_file.endswith('.vtr')]

    # Check if any files were found
    if not vtk_files:
        raise FileNotFoundError(f'No VTK files found in directory {run_dir}')
    
    # Ignore files that begin with '._' (these are temporary files created by macOS)
    vtk_files = [vtk_file for vtk_file in vtk_files if not vtk_file.startswith('._')]

    # Extract the time step from each file
    #  name by splitting based on '.' and 't'
    time_steps = [int(vtk_file.split('.')[0].split('t')[-1]) for vtk_file in vtk_files]

    # Return the maximum time step
    return max(time_steps)

def combine_tiles(time, run_dir):

    # List all the VTK files contained in the run directory
    vtk_files = [vtk_file for vtk_file in os.listdir(run_dir) if vtk_file.endswith('.vtr')]

    # Check if any files were found
    if not vtk_files:
        raise FileNotFoundError(f'No VTK files found in directory {run_dir}')
    
    # Ignore files that begin with '._' (these are temporary files created by macOS)
    vtk_files = [vtk_file for vtk_file in vtk_files if not vtk_file.startswith('._')]

    # Specify time as a string with leading zeros
    time_str = f'{time:04d}'

    # List all the VTK files contained in the run directory
    # vtk_files = [os.path.join(run_dir, vtk_file) for vtk_file in os.listdir(run_dir) if vtk_file.endswith(time_str+'.vtr')]
    vtk_files = [os.path.join(run_dir, vtk_file) for vtk_file in vtk_files if vtk_file.endswith(time_str+'.vtr')]

    # Check if any files were found
    if not vtk_files:
        raise FileNotFoundError(f'No VTK files found for time {time_str} in directory {run_dir}')
    
    # Load each tile
    tiles = [pv.read(vtk_file) for vtk_file in vtk_files]

    # Combine the tiles into a single grid
    combined_grid = tiles[0].copy()  # Start with a copy of the first tile
    for tile in tiles[1:]:  # Iterate over the rest of the tiles
        combined_grid = combined_grid.merge(tile)

    return combined_grid

# Convert unstructured grid to structured grid
def unstructured_to_structured(grid, variable_name='Electron Density'):
    # Assuming the grid is orthogonal and points can be mapped directly
    # Identify unique x, y, z coordinates (assuming sorted)
    x_coords = np.unique(grid.points[:, 0])
    y_coords = np.unique(grid.points[:, 1])
    z_coords = np.unique(grid.points[:, 2])

    # Check if the product of unique counts matches the total points (a necessary condition)
    if len(x_coords) * len(y_coords) * len(z_coords) == len(grid.points):
        # Proceed with conversion
        
        # Create meshgrid (assuming points are ordered and grid-like)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Create StructuredGrid
        structured_grid = pv.StructuredGrid(X, Y, Z)
        if np.allclose(structured_grid.points, grid.points):
            structured_grid.point_data[variable_name] = grid.point_data[variable_name]
        else:
            structured_grid_with_data = structured_grid.sample(grid)
    else:
        print("Cannot directly convert to StructuredGrid: point arrangement does not form a regular grid.")
    return structured_grid_with_data

def get_mesh_grid(grid):
    # Assuming the grid is orthogonal and points can be mapped directly
    # Identify unique x, y, z coordinates (assuming sorted)
    x_coords = np.unique(grid.points[:, 0])
    y_coords = np.unique(grid.points[:, 1])
    z_coords = np.unique(grid.points[:, 2])

    X, Y, Z = None, None, None
    # Check if the product of unique counts matches the total points (a necessary condition)
    if len(x_coords) * len(y_coords) * len(z_coords) == len(grid.points):
        # Proceed with conversion
        
        # Create meshgrid (assuming points are ordered and grid-like)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    else:
        print("Cannot directly convert to StructuredGrid: point arrangement does not form a regular grid.")

    return X, Y, Z

def get_mesh_subset(mesh, nxr, nzr):

    nxrange  = np.arange(nxr[0], nxr[1])
    nzrange  = np.arange(nzr[0], nzr[1])

    # Get the cell data for the electron density from the structured grid
    data = mesh.point_data['Electron Density']

    # Grid dimensions (assuming you know these or retrieve them from the grid)
    nx, nz, nu = mesh.dimensions

    Fy = np.zeros((nzr[1]-nzr[0], nxr[1]-nxr[0]))
    for zi in nzrange:
        for xi in nxrange:
            # Extract values along x for constant y and z
            value = data[(zi * nx) + xi]
            Fy[zi-nzr[0], xi-nxr[0]] = value
    return Fy

def plot_mesh_with_time_slider(mesh, scalar_field, cmap='terrain', clim=None, to_plot=True):
    # Get an array of x, y, z coordinates from the grid
    r = mesh.points[:, 0]
    z = mesh.points[:, 1]
    dr = r[1]-r[0]
    dz = dr
    time_max = find_max_time(run_path)
    print(f"Max Time: {time_max}")

    # Function to update the plot based on the slider's value (time step)
    def update_plot(value):
        time = int(value)
        print(f"Time Step: {time}" )
        # Combine the tiles into a single grid
        mesh2 = combine_tiles(time, run_path)

        # Get an array of x, y, z coordinates from the grid
        r = mesh2.points[:, 0]
        z = mesh2.points[:, 1]
        dr = r[1]-r[0]
        dz = dr

        # Update the scalar field based on the selected time step
        plotter.add_mesh(mesh2, scalars=scalar_field, cmap=cmap, clim=clim)
        plotter.render()

    # Create the plotter
    plotter = pv.Plotter()

    # Add the mesh with the scalar field
    plotter.add_mesh(mesh, scalars=scalar_field, cmap=cmap, clim=clim)

    # Show the axes
    plotter.show_axes()

    # Show the bounds
    plotter.show_bounds(grid='back', location='outer', ticks='both')

    # Add the slider to the plotter
    plotter.add_slider_widget(update_plot, rng=[0, time_max], value=0, title='Time Step')

    # Show the plotter (this will also render the plot)
    if to_plot:
        plotter.show()
    return r, z, dr, dz


# # Plot the mesh with the scalar field
def plot_mesh_with_scalar(mesh, scalar_field, cmap='terrain', clim=None, to_plot=True, plotterext=None, plotter_loc=[0, 0], columns=1, rows=1):
    if plotterext is None:
        plotter = pv.Plotter(shape=(rows, columns ))
        plotter.subplot(plotter_loc[0], plotter_loc[1])
    else:
        plotter = plotterext
        plotter.subplot(plotter_loc[0], plotter_loc[1])
    plotter.add_mesh(mesh, scalars=scalar_field, cmap=cmap, clim=clim)
    plotter.show_axes()
    plotter.show_bounds(grid='back', location='outer', ticks='both')
    if to_plot:
        plotter.show()
    return plotter

def fft_poisson_solver(charge_density, dx, dy, epsilon_0=8.854187817e-12):
    """
    Solve Poisson's equation for the electric potential using the FFT method.
    
    Parameters:
    charge_density (numpy.ndarray): 2D array of charge density values
    dx (float): Spacing between points in the x-direction
    dy (float): Spacing between points in the y-direction
    epsilon_0 (float): Permittivity of free space (default is 8.854187817e-12 F/m)
    
    Returns:
    numpy.ndarray: 2D array of electric potential values
    """
    ny, nx = charge_density.shape
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2

    # Avoid division by zero at the zero frequency component
    k2[0, 0] = 1.0

    # Compute the Fourier transform of the charge density
    rho_hat = fft2(charge_density)
    
    # Solve Poisson's equation in Fourier space
    phi_hat = -rho_hat / (epsilon_0 * k2)
    
    # Set the zero frequency component to zero to ensure zero mean for potential
    phi_hat[0, 0] = 0.0
    
    # Inverse Fourier transform to get back to real space
    phi = np.real(ifft2(phi_hat))

     # Plot the results
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13.385, 13.385)
    plt.imshow(phi, cmap='RdBu', aspect='equal')
    plt.colorbar()
    plt.tight_layout()
    plt.show()   

    return phi

def jacobi_solver(charge_density, dx, dy, epsilon_0=8.854187817e-12, tol=1e-6, max_iterations=10):
    """
    Solve Poisson's equation for the electric potential using the Jacobi method.
    
    Parameters:
    charge_density (numpy.ndarray): 2D array of charge density values
    dx (float): Spacing between points in the x-direction
    dy (float): Spacing between points in the y-direction
    epsilon_0 (float): Permittivity of free space (default is 8.854187817e-12 F/m)
    tol (float): Tolerance for convergence (default is 1e-6)
    max_iterations (int): Maximum number of iterations (default is 10000)
    
    Returns:
    numpy.ndarray: 2D array of electric potential values
    """
    ny, nx = charge_density.shape
    phi = np.zeros_like(charge_density)
    # Check if potential.pkl exists and load it
    if os.path.exists(savedir+'potential.pkl'):
        print("Loading potential from file")
        with open(savedir+'potential.pkl', 'rb') as f:
            phi = pickle.load(f)

    rho = charge_density / epsilon_0
    rho = charge_density 
    dx2 = dx**2
    dy2 = dy**2

    oneoverdx2dy2  = 1/(dx2 + dy2)

    for iteration in range(max_iterations):
        phi_new = np.copy(phi)
        
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                phi_new[i, j] = (0.5)*(oneoverdx2dy2)*(dy2*(phi[i+1, j] + phi[i-1, j]) + dx2*(phi[i, j+1] + phi[i, j-1]) - dx2*dy2*rho[i, j])
        
        if iteration == (max_iterations-1):
            print(f"Iteration {iteration}: Residual = {np.linalg.norm(phi_new - phi)}")

            # Save the potential in a pickle file
            with open(savedir+'potential.pkl', 'wb') as f:
                pickle.dump(phi_new, f)
                print(f"Saved {iteration}")


        # Check for convergence
        if np.linalg.norm(phi_new - phi) < tol:
            print(f"Converged after {iteration} iterations")
            
            # Save the potential in a pickle file
            with open(savedir+'potential.pkl', 'wb') as f:
                pickle.dump(phi_new, f)
            break
        else:
            print(f"Iteration {iteration}: Residual = {np.linalg.norm(phi_new - phi)}") 


        phi = phi_new
    
    # Plot the results
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13.385, 13.385)
    plt.imshow(phi, cmap='RdBu', aspect='equal')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    
    return phi/epsilon_0

def compute_electric_field(charge_density, dx, dy, epsilon_0=8.854187817e-12):
    """
    Compute the electric field from a 2D array of charge density.
    
    Parameters:
    charge_density (numpy.ndarray): 2D array of charge density values
    dx (float): Spacing between points in the x-direction
    dy (float): Spacing between points in the y-direction
    epsilon_0 (float): Permittivity of free space (default is 8.854187817e-12 F/m)
    
    Returns:
    (numpy.ndarray, numpy.ndarray): Tuple of 2D arrays representing the electric field components (Ex, Ey)
    """
    if True:
        # Solve for the electric potential using the FFT method
        phi = fft_poisson_solver(charge_density, dx, dy, epsilon_0)
    else:
        # Solve for the electric potential using the Jacobi method
        phi = jacobi_solver(charge_density, dx, dy, epsilon_0)

    # Compute the gradient of the potential
    Ex, Ey = np.gradient(phi, dx, dy)
    
    # The electric field is the negative gradient of the potential
    Ex = -Ex
    Ey = -Ey
    
    return Ex, Ey

def compute_divergence(Ex, Ey, dx, dy):
    """
    Compute the divergence of the electric field components.
    
    Parameters:
    Ex (numpy.ndarray): 2D array of the x-component of the electric field
    Ey (numpy.ndarray): 2D array of the y-component of the electric field
    dx (float): Spacing between points in the x-direction
    dy (float): Spacing between points in the y-direction
    
    Returns:
    numpy.ndarray: 2D array of the divergence of the electric field
    """
    dEx_dx = np.gradient(Ex, dx, axis=1)
    dEy_dy = np.gradient(Ey, dy, axis=0)
    divergence = dEx_dx + dEy_dy
    return divergence

# Function to convert the pyvista data to a numpy array
def pyvista_to_numpy(mesh, variable_name):
    # Get the cell data for the electron density from the structured grid
    data = mesh.point_data[variable_name]

    # Grid dimensions (assuming you know these or retrieve them from the grid)
    nx, nz, nu = mesh.dimensions
    shape = data.shape
    nD = shape[1] if len(shape) > 1 else 1

    Fy = np.zeros((nz, nx, nD))

    for n in range(nD):

        # Fy = np.zeros((nz, nx))
        for zi in range(nz):

            # Constant y and z indices
            constant_z_index = 0  # Example: constant z index
            constant_nu_index = 0  # Example: constant z index

            # Calculate the start and end indices in the 1D array for the slice
            start_index = (constant_nu_index * nu * nx) + (zi * nx)
            end_index = start_index + nx

            # Extract values along x for constant y and z
            values_along_x = data[start_index:end_index]
            if nD == 1:
                Fy[zi, :, n] = values_along_x
            else:
                Fy[zi, :, n] = values_along_x[:, n]

        # Reflect Fy about the horizontal axis
        Fy = np.flip(Fy, axis=0)
    if nD == 1:
        return Fy[:, :, 0]  
    elif nD == 2:
        return Fy[:, :, 0], Fy[:, :, 1]
    elif nD == 3:     
        return Fy[:, :, 0], Fy[:, :, 1], Fy[:, :, 2]

# Run Directory
# Check which file system is being used
osx = False
if os.name == 'posix':
    osx = True


# If mac osx #
if osx:
    datadir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Data/'
    savedir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Plots/'
else:
    drive_letter = 'D:'

    data_path_on_external_drive = 'XSPL/Lasers/Simulations/Bluehive/PERSEUS/EM_Sims/' 
    plot_path_on_external_drive = 'XSPL/Lasers/Outputs/Plots/'  

    datadir = drive_letter + '\\' + data_path_on_external_drive       
    savedir = drive_letter+'\\'+plot_path_on_external_drive

run = 'S-Polarized'
run = 'P-polarized'

run_path = datadir+run+'/data/'

time_analyze = 10
time = find_max_time(run_path)

# Combine the tiles into a single grid
mesh = combine_tiles(time_analyze, run_path)

# Get an array of x, y, z coordinates from the grid
test = mesh.points
r = mesh.points[:, 0]
z = mesh.points[:, 1]
minR = np.min(r)
maxR = np.max(r)
minZ = np.min(z)
maxZ = np.max(z)

# Find the minimum spacing in the z direction that is greater than zero
dr = np.abs(r[1]-r[0])
diffR = np.diff(r)
dr = np.min(diffR[diffR > 0])

# Find the minimum spacing in the z direction that is greater than zero
diffZ = np.diff(z)
dz = np.min(diffZ[diffZ > 0])

# Convert to StructuredGrid
smesh = unstructured_to_structured(mesh, variable_name='Log Ion Density')

# Grid dimensions (assuming you know these or retrieve them from the grid)
nx, nz, nu = smesh.dimensions

# Get the cell data for the electron density from the structured grid
Exp, Eyp, Ezp = pyvista_to_numpy(smesh, 'Electric Field')
cd = pyvista_to_numpy(smesh, 'charge density')
divE = pyvista_to_numpy(smesh, 'div E')
ne = pyvista_to_numpy(smesh, 'Electron Density')
ni = pyvista_to_numpy(smesh, 'Density')

# Plot the results
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(Exp, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('Ex')
#Save the figure
fig.savefig(savedir+run+"_"+str(time_analyze)+'_Ex.png', dpi=300)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(Eyp, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('Ey')
#Save the figure
fig.savefig(savedir+run+"_"+str(time_analyze)+'_Ey.png', dpi=300)


elc = 1.60217662e-19
Ex, Ey = compute_electric_field(elc*cd, dr, dz)

# Compute gradient of E #
eps0 = 8.854187817e-12
rho2 = eps0*compute_divergence(Ex, Ey, dr, dz)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(ni, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('Ni Perseus')
fig.savefig(savedir+run+"_"+str(time_analyze)+'_nipers.png', dpi=300)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(ne, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('Ne Perseus')
fig.savefig(savedir+run+"_"+str(time_analyze)+'_nepers.png', dpi=300)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(elc*cd, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('CD Perseus')
fig.savefig(savedir+run+"_"+str(time_analyze)+'_cdpers.png', dpi=300)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(rho2, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('CD Computed')
fig.savefig(savedir+run+"_"+str(time_analyze)+'_cdpyth.png', dpi=300)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 13.385)
plt.imshow(elc*divE, cmap='RdBu', extent=[minR,maxR, minZ, maxZ], aspect='equal')
plt.colorbar()
plt.tight_layout()
plt.title('div E')
fig.savefig(savedir+run+"_"+str(time_analyze)+'_divEpers.png', dpi=300)

plt.show()

# # Plotting
r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True)
r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Magnetic Field', cmap='terrain', clim=[0,250], to_plot=True)
print(f"dr: {dr}, dz: {dz}")

# Plot the structured grid and show the axis and labels    
# plotter1 = plot_mesh_with_scalar(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True, plotter_loc=[0, 0], columns=1, rows=1)
# plotter2 = plot_mesh_with_scalar(smesh, 'Magnetic Field', cmap='terrain', clim=[0, 100], to_plot=True, plotter_loc=[0, 0])

# Plot the results
# # Attenuation for 10 keV in aluminum
# mu = 5.033E+01 # cm^2/g for 8 keV in aluminum
# # mu = 2.623E+01 # cm^2/g for 10 keV in aluminum
# # mu = 7.955E+00 # cm^2/g for 15 keV in aluminum
# # mu = 3.441E+00 # cm^2/g for 20 keV in aluminum
# # mu = 1.128E+00 # cm^2/g for 30 keV in aluminum

# # Convert number density to mass density
# M=26.98 #g/mol for aluminum,
# NA=6.022e23 # atoms/mole.

# arg = Fy*1e-4 * mu * M/NA
# Ip = np.exp(-arg)

# # Add this Scalar value to the grid
# smesh.point_data['stopping'] = Ip.flatten()

# num_photons_per_pulse = 1e12
# V=60e-6
# H=60e-6
# focal_spot_area = V*H #(VxH) in m^2

# # Get the bounds of the grid
# bounds = mesh.bounds

# # The bounds are in the order: [xmin, xmax, ymin, ymax, zmin, zmax]
# rmin, rmax, zmin, zmax, _, _ = bounds

# # Print min and max values
# print(f"X Min: {rmin}, X Max: {rmax}")
# print(f"Y Min: {zmin}, Y Max: {zmax}")

# # Define the region of interest
# scl = 5
# rblow = 0
# rbhigh = H/2
# zblow = -V/2
# zbhigh = V/2

# rblow = 0
# rbhigh = 200e-6
# zblow = -1.5e-3
# zbhigh = 1.5e3

# # Get Ip for a region of interest
# # Assuming the region of interest is a box defined by the following limits

# # Find the 1D indices for the region of interest in the mesh
# r_min_index = int(rblow/ dr)
# r_max_index = int(rbhigh / dr)
# z_min_index = int((zblow-(zmax-zmin)/2) / dz)
# z_max_index = int((zbhigh-(zmax-zmin)/2) / dz)

# # Plot and make the aspect ratio consistent with the extent
# # Reflect Ip about the vertical axis and plot both
# Ipleft = np.flip(Ip, axis=1)
# Ipfull = np.concatenate((Ipleft, Ip), axis=1)

# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(13.385, 6.0)
# plt.imshow(Ipfull, cmap='RdBu', extent=[-rmax, rmax, zmin, zmax], aspect='equal', vmin=0, vmax=1)

# sbmesh = get_mesh_subset(smesh, [r_min_index, r_max_index], [z_min_index, z_max_index])
# plt.imshow(sbmesh, cmap='terrain', aspect='equal', extent=[10**6*rblow, 10**6*rbhigh, 10**6*zblow, 10**6*zbhigh])
# cbar = fig.colorbar(lc, ax=ax)
# cbar.set_label('Color mapping value')
# show colorbar
# plt.colorbar()
# plt.tight_layout()
# fig.savefig(savedir+run+"_"+str(time_analyze)+'_'+str(mu)+'_absorption.png', dpi=300)
# plt.show()
