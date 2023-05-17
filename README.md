# ADVERSARIAL_SENSITIVTY

Welcome to the official repository for the research paper titled "On the Interpretable Adversarial Sensitivity of Iterative Optimizers" authored by an anonymous researcher.

## Installation
To get started, please follow the instructions below to install the necessary requirements:

1. Download/clone this repository.
2. Navigate the the project directory using **cd \<folder>**
3. Run the command **'pip install -r requirements.txt'** to make sure all necessary packages is installed

## Usage

To replicate the experiments and perform the analysis outlined in the paper, please follow the instructions below:

Run the following commnad: **'python main.py'** using the appropriate flag to execute each attack and generate the associated graphs. The available flags are as follows:

**--ista**: Executes the attack using the ISTA (Iterative Soft Thresholding Algorithm) method and generates the corresponding graphs.

**--admm**: Executes the attack using the ADMM (Alternating Direction Method of Multipliers) method and generates the corresponding graphs.

**--beamforming**: Executes the attack using the Hybrid Beamforming technique and generates the related graphs.

Each flag corresponds to a distinct case study discussed in the paper.
Feel free to explore the provided code and experiment with different configurations. If you have any questions or require further assistance, please don't hesitate to reach out to us.
