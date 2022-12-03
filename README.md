# Superconducting Optoelectronic Network Simulator (`sim_soens`)
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
   - `pip3 install numpy`
   - `pip3 install matplotlib`
   - `pip3 install brian2`
   - `pip3 install scipy` 
 - Open `soen_tutorial` for a simulator walkthrough 
   - How to use jupyter notebooks: https://www.dataquest.io/blog/jupyter-notebook-tutorial/
   - Or just use vscode jupyter extension to use in your IDE

## Features
 - *Custom neuron generation
   - Any possible SOEN neuron morphology can be called with a single line 
 - *Networking
   - Simply define network size, connectivity, and neuron types
 - Input options
   - Canonical and custom datasets can be called natively
     - Neuromorphic MNIST
     - **Heidelberg spoken digit dataset
     - **IBM DVS gesture dataset
 - Visualization tools
   - See neuron morphologies, dendrite behavior and network activity with built-\
     in plotting functions

## Algorithms
 - *Liquid State Machines
 - **Eprop
 - **Dendritic credit assignment


## Benchmarks
 - *Nine pixel classifier with a single neuron
 - **Image classifcation
 - **Speech recognition
 - **Gesture recognition

\
\
*in development\
**planned