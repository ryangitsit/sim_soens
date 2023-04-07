# Superconducting Optoelectronic Network Simulator
![plot](./img/wafer_tilted.png)
Superconducting Optoelectronic Networks (SOENs) are an in-development neuromorphic hardware that offer *lightspeed communication* and *brain-level scalability*.\
This repo is for the coders, neuroscientists, and machine-learners that want to play with the computational properties of SOENs through simulations, based on the phenomological model linked here:
 - https://arxiv.org/abs/2210.09976

For the interested hardware afficianados and device physicists, explanatory materical can be found:
 - https://aip.scitation.org/doi/full/10.1063/1.5096403

Enjoy!

## Getting Started
 - Clone this repo
   - Type the following in the command line of your desired local directory
   - `git clone https://github.com/ryangitsit/sim_soens.git` 
 - Be sure to have the necessary python packages with the following commands
   - `pip install -r requirements.txt` 
 - Open `NICE_tutorial` for a simulator walkthrough 
   - How to use jupyter notebooks: https://www.dataquest.io/blog/jupyter-notebook-tutorial/
   - Or just use vscode jupyter extension to use in the vscode IDE

## Features
 - Custom neuron generation
   - Any possible SOEN neuron morphology can be called through the 'SuperNode` class
 - Networking
   - Hand craft networks with specified connectivity or call on pre-made nets
 - Input options
   - Canonical and custom datasets can be called natively
     - Random
     - Defined
     - Neuromorphic MNIST
     - Saccade MNIST
 - Visualization tools
   - Neuron morphologies
   - Dendrite behavior
   - Network activity
