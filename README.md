# ADVERSARIAL_SENSITIVITY

Welcome to the official repository for the research paper titled "On the Interpretable Adversarial Sensitivity of Iterative Optimizers" authored by Elad Sofer and Nir Shlezinger, published in the 33rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP 2023)


The authors are with the School of ECE, Ben-Gurion University of the Negev, Israel (e-mail: elad.g.sofer@gmail.com; nirshl@bgu.ac.il). 


## Installation
To get started, please follow the instructions below to install the necessary requirements:

1. Ensure that Python version 3 is installed on your system.
2. Download or clone this repository to your local machine.
3. Navigate to the project directory using **'cd \<folder>'** command.
4. Run the command **'pip install -r requirements.txt'** to install all the necessary packages.

## Usage

To replicate the experiments and perform the analysis outlined in the paper, please follow the instructions below:
1. Navigate to the project directory using **'cd \<folder>'** command.
2. Run the following commnad: **'python main.py'** using the appropriate flag to execute each attack and generate the associated graphs. The available flags are as follows:

**--ista**: Executes the attack using the ISTA (Iterative Soft Thresholding Algorithm) case-study and generates the corresponding graphs.

**--admm**: Executes the attack using the ADMM (Alternating Direction Method of Multipliers) case-study and generates the corresponding graphs.

**--beamforming**: Executes the attack using the Hybrid Beamforming case-study and generates the related graphs.

Each flag corresponds to a distinct case study discussed in the paper.
Feel free to explore the provided code and experiment with different configurations. If you have any questions or require further assistance, please don't hesitate to reach out to us.
