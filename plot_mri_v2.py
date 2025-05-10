import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import os
import numpy as np
import nibabel as nib

class MRIImageViewer:
    def __init__(self, mri_volume):
        """
        Initialize the MRI Image Viewer.

        Parameters:
        mri_volume (numpy.ndarray): 3D MRI data (e.g., shape: (240, 240, 155))
        """
        if len(mri_volume.shape) != 3:
            raise ValueError("Input MRI volume must be a 3D array.")

        self.mri_volume = mri_volume
        self.num_slices = mri_volume.shape[2]
        self.current_slice = 0

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.mri_volume[:, :, self.current_slice], cmap='gray')
        self.ax.set_title(f"Slice {self.current_slice + 1}/{self.num_slices}")
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        # Add buttons
        self.prev_button_ax = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.next_button_ax = plt.axes([0.81, 0.02, 0.1, 0.04])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')
        self.prev_button.on_clicked(self.show_previous_slice)
        self.next_button.on_clicked(self.show_next_slice)

    def show_previous_slice(self, event=None):
        """Display the previous slice."""
        if self.current_slice > 0:
            self.current_slice -= 1
            self.update_display()

    def show_next_slice(self, event=None):
        """Display the next slice."""
        if self.current_slice < self.num_slices - 1:
            self.current_slice += 1
            self.update_display()

    def update_display(self):
        """Update the displayed slice."""
        self.img.set_data(self.mri_volume[:, :, self.current_slice])
        self.ax.set_title(f"Slice {self.current_slice + 1}/{self.num_slices}")
        self.fig.canvas.draw()

    def key_press(self, event):
        """Handle key press events."""
        if event.key == 'left':
            self.show_previous_slice()
        elif event.key == 'right':
            self.show_next_slice()

    def show(self):
        """Display the viewer."""
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your MRI data (replace with actual data loading)
    file_name = 'BraTS20_Training_308'
    img = nib.load('/Users/sgutta/Desktop/Brain_Tumor_Data/MICCAI_BraTS2020_TrainingData/'+file_name + '/' + file_name + '_weighted1233.nii.gz')
    mri_volume = img.get_fdata()

    # Create and display the viewer
    viewer = MRIImageViewer(mri_volume[40:200, 20:230, 27:127])
    #viewer = MRIImageViewer(mri_volume[50:190, 30:220, 1:155])
    viewer.show()
