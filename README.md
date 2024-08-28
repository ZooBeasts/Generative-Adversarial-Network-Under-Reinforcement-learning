# Enhanced Disordered Waveguide Inverse Design Using a Generative Adversarial Network with Agent-Guided Integration for Nonlinear Conditioned Geometries Generation
## Ziheng Guo<sup>1</sup>, Andrea Di Falco<sup>1</sup>


### This is the follow-up project of a generative model for multi-purpose inverse design disordered waveguides, we reconstruct the whole project. The RL is a supervisor in the generator along with Critic to double guide the generation processes.

### The aim of the project is to find a consistent generation of disordered, spare type of waveguide geometries that understand the nonlinear chaotic dynamics in light propagation in nonlinear regimes 

![image](https://github.com/user-attachments/assets/51666bea-27d3-47ce-b778-c0dfe15de0cc)

The idea has been proven to be correct. The generator's accuracy in correctly generating geometries matching optical conditions has risen from 0.1 % to 1% to 10% to 30%. In other words, 100 images used to be 1 image correctly matched to now 30 images. But this is still statistically speaking.

DQN takes action after warm-up epochs. These actions include changing the learning rate, adjusting the latent vectors, reducing the drop-out rate, and perturbing the parameters.
Mode collapse prevention is controlled by DQN with parameters rerolls and hard reset. Conditional Batch Normalizations along with Batch Normalization mixing control have been considered, but beta in CBN changes the brightness of the fake images, causing hole probability values to shift.  

Measurement of how good the generator is now determined by 50 images per data per batch, such as 64x50 = 3200 images after post-normalization and extract positions feeding into the pre-trained model, by MSE threshold matching. During the training, the generator creates more matching condition geometries.  
The pre-trained model training loss is 1.9e-5 and val loss is 1.87e-5 and test loss is 2.4e-5 in disordered dataset.
Therefore, the MSE threshold set is [0.09, 0.008, 0.001,0.0005]




--------------------------------------------------------------------------------
Actions are taken sequentially in training as follows but in the author's training situations. Action choices might change during to different configurations.

### Action 1 Changing Z2 latent, stable for 30 epochs, MSE checked 6e-2, lower than 0.09 MSE threshold, training continues.
### Action 2 Changing Dropout to 0.45, stable for 30 epochs, MSE checked 6e-2, lower than 0.09 but higher than 0.008, training continues
### Action 3 Parameter Perturbation, stable for 30 epochs, MSE higher than 0.008, training continues
### Action 4 Changing Z2 latent, stable for 30 epochs, MSE checked 3.213e-2. training continues. 
