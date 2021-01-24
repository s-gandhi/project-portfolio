import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()


# all the output should are saved in a 'results' directory

def test_1():
    """This test plots the phantom image next to the reconstructed CT data, allowing for a direct visual comparison
    between the original and returned image. This high level comparison demonstrates the CT simulator is generally
    working as expected if features in the phantom appear in the appropriate positions on the reconstructed image"""

    def run_trial(case, material_name=None):
        """run_trial plots a phantom alongside a CT reconstructed image
        x = run_trial(case, material) creates a CT phantom of a given type and material, of size 256 x 256 and
        scale 0.1, from the ct_phantom function. It is then scanned and reconstructed using 100kVp source with a
        1mm Al filter, and the reconstructed image plot beside the phantom."""

        # work out what the initial conditions should be
        # construct the phantom and X-ray source
        p = ct_phantom(material.name, 256, case, metal=material_name)
        s = source.photon('100kVp, 1mm Al')
        # obtain the reconstructed CT image
        y = scan_and_reconstruct(s, material, p, 0.1, 256)

        # plot the phantom and CT image side-by-side
        map = 'gray'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Phantom (' + material_name + ')')
        im = ax1.imshow(p, cmap=map)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, orientation='vertical')

        ax2.set_title('Reconstruction')
        im2 = ax2.imshow(y, cmap=map)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, orientation='vertical')

        plt.axis('off')

        # save the plots
        full_path = get_full_path('results', 'test_1_case_' + str(case) + '_' + material_name)
        plt.savefig(full_path)
        plt.close()

    # perform the test for a single hip replacement (type 3) with various materials
    run_trial(3, 'Titanium')
    run_trial(3, 'Bone')
    run_trial(3, 'Iron')
    run_trial(3, 'Aluminium')

    # perform the test for the remaining type cases
    run_trial(4, 'Titanium')
    run_trial(5, 'Titanium')
    run_trial(6, 'Titanium')
    run_trial(7, 'Titanium')
    run_trial(8, 'Titanium')


def test_2():
    """This test quantitatively assesses the geometrical accuracy of a reconstructed CT image, by comparing the
    location of features with the original phantom"""

    def run_trial(case, accuracy, energy=0.1, material_name='Titanium'):
        """run_trial returns the percentage error in the number of pixels corresponding to a metal implant feature
         between the reconstructed image and the phantom.
         run_trial(case, accuracy, energy=0.1, material_name='Titanium') constructs a phantom of given type
         and material, scans it with an ideal source energy distribution, and reconstructs. The pixel locations
         corresponding to the metal feature are found in the phantom. The feature location in the reconstruction is
         found by identifying pixels which lie to within a specified accuracy of the maximum data value. The error
         is taken to be the number of differing positions."""
        # construct a phantom of size 256x256 and given material
        p = ct_phantom(material.name, 256, case, metal=material_name)
        # use an ideal source to negate the effects of a polyenergetic beam on the measured attenuation coefficients
        s = fake_source(source.mev, energy, method='ideal')
        # scan the phantom, assuming a pixel scale of 0.1 cm per pixel, and reconstruct the image
        y = scan_and_reconstruct(s, material, p, 0.1, 256)

        # obtain the location of and the number of pixels corresponding to the metal features in the phantom
        # assign these locations a flag bit 1, and all other locations a 0
        phantom_indices = np.where(p == material.name.index(material_name), 1, 0)
        phantom_indices_len = len(np.where(phantom_indices == 1)[0])

        # the maximum data value in the reconstructed array will
        # correspond to the metal feature in the reconstructed image
        max_value = np.max(y)

        # locate all the data values which lie to within a given tolerance of this maximum
        # these locations are assumed to correspond to the metal feature
        # assign these locations a flag bit 1, and all other locations a 0
        samples = np.where(y > max_value * (1 - accuracy), 1, 0)

        # find the number of differing bits between the two feature arrays
        error = np.sum(np.abs(phantom_indices - samples).flatten())

        # save the array index results for both the phantom and reconstruction, as well as a % error
        f.write('Case ' + str(case) + ', accuracy = ' + str(accuracy) + ', energy = ' + str(energy) + '\n')
        f.write('Number of incorrect indices: ' + str(error) + '\n')
        f.write('Number of pixels in phantom with index ' + str(material.name.index(material_name))
                + ' (' + material_name + ') : ' + str(phantom_indices_len) + '\n')
        f.write('Error percentage: ' + str(np.round((error / phantom_indices_len)*100, 2)) + '%\n\n')

    f = open('results/test_2_output.txt', mode='w')
    run_trial(3, 0.3)
    run_trial(3, 0.2)
    run_trial(3, 0.1)
    run_trial(3, 0.05)
    run_trial(3, 0.01)
    run_trial(4, 0.2)
    run_trial(5, 0.2)
    run_trial(6, 0.2)
    run_trial(7, 0.2)
    run_trial(8, 0.2)
    f.close()


def test_3():
    """ This test quantitatively assesses the accuracy of the measured linear attenuation coefficients u"""

    def run_trial(material_name, energy, mean=True, accuracy=0.05):
        """run_trial returns the mean value of the linear attenuation coefficient from a region of the measured
        distribution, and compares it to the actual value by means of a % error.
        The function takes in a material (string) and an energy (float). It creates a type 1 image phantom
        (circular disk) of size 256x256, scale 0.1 cm per pixel and specified material, which is scanned using an ideal
        fake source of specified MeVp energy. The mean coeff is found for a 128x128 square in the centre of the image,
        and compared to the actual coeff data from an xlsx spreadsheet. The results are stored in test_3_output.txt with
        an error range (percentage difference between experimental and exact attenuation values)
        """
        # construct a phantom of size 256x256 and given material
        p = ct_phantom(material.name, 256, 1, metal=material_name)
        # use an ideal source to negate the effects of a polyenergetic beam on the measured attenuation coefficients
        s = fake_source(source.mev, energy, method='ideal')
        # scan the phantom, assuming a pixel scale of 0.1 cm per pixel, and reconstruct the image
        y = scan_and_reconstruct(s, material, p, 0.1, 256)

        if mean:
            # calculate the mean of the centre 128x128 pixels
            experimental_value = np.mean(y[64:192, 64:192])
        else:
            # find the max if material is a metal or with low energies
            max_value = np.max(y)
            # average the coeff values which lie within a given accuracy of this max
            samples = np.where(y > max_value * (1 - accuracy), y, 0)  # Select values accuracy% from max value
            samples = np.ma.masked_equal(samples, 0)  # Remove zeros for mean calculation
            experimental_value = np.mean(samples)

        # obtain the peak energy in the ideal source
        shifted_energy = np.round(energy * 0.7, 3)
        # obtain actual coeff value from excel spreadsheet
        actual_value = material.coeff(material_name)[np.where(material.mev == shifted_energy)[0][0]]

        # save the mean, actual and % difference results in a txt file
        f.write(material_name + ' (' + str(energy) + 'kVp): \n')
        f.write('Mean experimental value: ' + str(experimental_value) + ' cm^-1\n')
        f.write('Actual linear attenuation coefficient: ' + str(actual_value) + ' cm^-1\n')
        f.write('Difference between the values: ' +
                str(np.round((abs(actual_value - experimental_value) / actual_value) * 100, 2)) + '%\n\n')

    # perform the 128x128 mean for various materials
    f = open('results/test_3_output.txt', mode='w')
    f.write('Taking the average over values that lie in the middle of the image\n')
    run_trial('Soft Tissue', 0.05)
    run_trial('Soft Tissue', 0.1)
    run_trial('Soft Tissue', 0.2)
    run_trial('Adipose', 0.05)
    run_trial('Adipose', 0.1)
    run_trial('Adipose', 0.2)
    run_trial('Bone', 0.1)
    run_trial('Bone', 0.2)
    run_trial('Bone', 0.25)
    run_trial('Titanium', 0.25)

    # average the coeff for data close to the max value
    f.write('\n')
    f.write('Taking the average over values that are 5% away from the maximum\n')
    run_trial('Bone', 0.05, False)
    run_trial('Soft Tissue', 0.02, False)
    run_trial('Titanium', 0.1, False)
    run_trial('Iron', 0.2, False)
    run_trial('Soft Tissue', 0.02, False, 0.1)
    f.close()


# Run the various tests
#print('Test 1')
#test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()

