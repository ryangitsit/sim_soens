Introduction
============

.. _intro:

Overview
--------
*Superconducting Optoelectronic Networks (SOENs)* combine photonics 
and superconductors to instantiate computing systems that 
approach the fundamental limits of information processing in terms
of speed and scalability. Overcoming the engineering challenges
of integrating these technologies into one system has consistently
been aided by using neuro-inspired architectures. SOENs are natively 
Spiking Neural Networks (SNNs) of loop neurons, which
themselves are comprised of many subsequent dendrites organized
into intricate morphologies.

**sim_soens** uses the forward euler method to simulate the dynamics of 
SOENs one time-step at a time (with the expection of a few speed-motivated tricks).
The simulator has an easy-to-use python front end.  A user defines a network of 
specified neurons and simulations instrucitons.  sim_soens runs this network over
these instructions with either a detail-rich python backend, or more constrained 
(but much faster) julia version.  There are number of run-options described in greater
detail in the system architecture section of this documentation.  

The user may wish to define custom neurons with `SuperNode` class or generic neurons 
from `neuron_library.py`.  Likewise, there is a custom network class `SuperNet` and a
collection of pre-defined networks in `network_library.py`.  All of these options come
equipped with in-built plotting and analytic functions. 

Basics
--------


Examples
---------
