import numpy as np
import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ContrastCurveGenerator:
    def __init__(self, Fmax=0.65):
        """
        Initialize the ContrastCurveGenerator with a maximum contrast value.

        Parameters:
            Fmax (float): Maximum contrast value for scaling (default is 0.65).
        """
        self.Fmax = Fmax

    def generate_contrast_curves(self, option="exp", custom_file=None):
        """
        Generate contrast curves for Band, Pole, and Ambient based on the specified option.

        Parameters:
            option (str): The type of function to use for generating contrast curves. 
                          Options are "exp", "lin", or "custom".
            custom_file (str): Path to a file containing custom contrast values (used for "custom" option).

        Returns:
            pd.DataFrame: A DataFrame containing the contrast curves for Band, Pole, and Ambient.
        """
        # Generate a grid of log10 pressures from 0.1 bar to 10 bar
        pressures = np.flip(np.logspace(-1, 1, 100))  # Reverse the array using np.flip

        if option == "exp":
            # Generate gentle exponential functions
            band = (np.exp(-1*pressures) + 1.5) / 3.0 * self.Fmax  # Scale to Fmax
            pole = (np.exp(-0.55*pressures) + 1.2) / 3.0 * self.Fmax
            ambient = (np.exp(-1*pressures) + 1.0) / 3.0 * self.Fmax
        elif option == "lin":
            # Generate linear functions
            band = (-pressures + 1.5) / 3.0 * self.Fmax  # Scale to Fmax
            pole = (-pressures + 1.2) / 3.0 * self.Fmax
            ambient = (-pressures + 1.0) / 3.0 * self.Fmax
        elif option == "custom":
            if custom_file is None:
                raise ValueError("Custom file path must be provided for 'custom' option.")
            # Read custom contrast values from the file
            custom_data = pd.read_csv(custom_file)
            if not all(col in custom_data.columns for col in ["Band", "Pole", "Ambient"]):
                raise ValueError("Custom file must contain 'Band', 'Pole', and 'Ambient' columns.")
            band = custom_data["Band"].values * self.Fmax / max(custom_data["Band"].max(), 1)
            pole = custom_data["Pole"].values * self.Fmax / max(custom_data["Pole"].max(), 1)
            ambient = custom_data["Ambient"].values * self.Fmax / max(custom_data["Ambient"].max(), 1)
        else:
            raise ValueError("Invalid option. Choose from 'exp', 'lin', or 'custom'.")

        # Create a DataFrame for the results
        contrast_df = pd.DataFrame({
            "Pressure (bar)": pressures[::-1],  # Reverse the pressures to match the order of the curves
            "Band": band,
            "Pole": pole,
            "Ambient": ambient
        })

        return contrast_df

    def plot_contrast_curves(self, contrast_df):
        """
        Plot the contrast curves from the given DataFrame.

        Parameters:
            contrast_df (pd.DataFrame): DataFrame containing the contrast curves to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(contrast_df["Pressure (bar)"], contrast_df["Band"], label="Band", color='blue') 
        plt.plot(contrast_df["Pressure (bar)"], contrast_df["Pole"], label="Pole", color='orange')
        plt.plot(contrast_df["Pressure (bar)"], contrast_df["Ambient"], label="Ambient", color='green') 
        plt.xscale('log')
        plt.xlabel("Pressure (bar)")
        plt.ylabel("Contrast")
        plt.title(f"Contrast Curves (Fmax = {self.Fmax})")
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Flip the x-axis to ensure higher pressure is on the left
        plt.gca().invert_xaxis()

        plt.show()

# Example usage:
# Create an instance of the class
generator = ContrastCurveGenerator(Fmax=0.75)

# Generate contrast curves
df = generator.generate_contrast_curves(option="exp")

# Plot the contrast curves
generator.plot_contrast_curves(df)
