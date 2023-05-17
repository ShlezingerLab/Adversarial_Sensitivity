# ADVERSARIAL_SENSITIVTY

Welcome to the official repository for the research paper titled "On the Interpretable Adversarial Sensitivity of Iterative Optimizers" authored by an anonymous researcher.

## Installation
To get started, please follow the instructions below to install the necessary requirements:

1. Ensure that Python version 3 is installed on your system.
2. Download or clone this repository to your local machine.
3. Navigate the the project directory using **'cd \<folder>'** command.
4. Run the command **'pip install -r requirements.txt'** to install all the necessary packages.

## Usage

To replicate the experiments and perform the analysis outlined in the paper, please follow the instructions below:

Run the following commnad: **'python main.py'** using the appropriate flag to execute each attack and generate the associated graphs. The available flags are as follows:

**--ista**: Executes the attack using the ISTA (Iterative Soft Thresholding Algorithm) case-study and generates the corresponding graphs.

**--admm**: Executes the attack using the ADMM (Alternating Direction Method of Multipliers) case-study and generates the corresponding graphs.

**--beamforming**: Executes the attack using the Hybrid Beamforming case-study and generates the related graphs.

Each flag corresponds to a distinct case study discussed in the paper.
Feel free to explore the provided code and experiment with different configurations. If you have any questions or require further assistance, please don't hesitate to reach out to us.
