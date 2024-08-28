# Enhanced Disordered Waveguide Inverse Design Using a Generative Adversarial Network with Agent-Guided Integration for Nonlinear Conditioned Geometries Generation
## Ziheng Guo<sup>1</sup>, Andrea Di Falco<sup>1</sup>


### This project is a continuation of work on a generative model designed for the inverse design of disordered waveguides, particularly focused on achieving multi-purpose functionalities. The project involves a complete reconstruction of the previous model, integrating Reinforcement Learning (RL) as a supervisory mechanism in the generator, alongside a Critic, to enhance the guidance of the generation process.

The primary objective is to develop a model that can consistently generate disordered, sparse waveguide geometries that accurately capture the nonlinear chaotic dynamics associated with light propagation in nonlinear regimes.

The effectiveness of this approach has been validated, as demonstrated by a significant improvement in the generator’s accuracy. The proportion of generated geometries that match the desired optical conditions has increased from 0.1% to 30%, meaning that now 30 out of 100 generated images meet the criteria, compared to just one image previously. However, this improvement is still under statistical consideration.

During the training process, the DQN (Deep Q-Network) takes action after a warm-up period, adjusting parameters such as the learning rate, latent vectors, dropout rate, and perturbation parameters. It also plays a crucial role in preventing mode collapse through parameter rerolls and hard resets.

However, Conditional Batch Normalization (CBN) has introduced challenges by altering the brightness of generated images through the beta parameter. This change in brightness affects pixel values post-normalization, sometimes causing certain pixel values to become zero or very small, effectively making them disappear or become less significant.

The generator’s performance is now evaluated using 50 images per data batch (e.g., 64x50 = 3200 images) after post-normalization, with positions extracted and fed into a pre-trained model. The model’s performance is assessed using MSE (Mean Squared Error) threshold matching. The pre-trained model achieves a training loss of 1.9e-5, validation loss of 1.87e-5, and test loss of 2.4e-5 on the disordered dataset. Consequently, the MSE thresholds are set at [0.09, 0.008, 0.001, 0.0005].

![image](https://github.com/user-attachments/assets/51666bea-27d3-47ce-b778-c0dfe15de0cc)






--------------------------------------------------------------------------------
Actions are taken sequentially in training as follows but in the author's training situations. Action choices might change during to different configurations.

### Action 1 Changing Z2 latent, stable for 30 epochs, MSE checked 6e-2, lower than 0.09 MSE threshold, training continues.
### Action 2 Changing Dropout to 0.45, stable for 30 epochs, MSE checked 6e-2, lower than 0.09 but higher than 0.008, training continues
### Action 3 Parameter Perturbation, stable for 30 epochs, MSE higher than 0.008, training continues
### Action 4 Changing Z2 latent, stable for 30 epochs, MSE checked 3.213e-2. training continues. 
