import gymnasium
import torch
import torch as nn




class ENV:
    def __init__(self, generator, critic, pre_trained_cnn, device):
        self.generator = generator
        self.critic = critic
        self.pre_trained_cnn = pre_trained_cnn
        self.device = device

    def reset(self):
        return torch.zeros(21)


    def step(self,action, points,points21,noise):
        if action == 0:
            # Adjust learning rate
            for g in opt_gen.param_groups:
                g['lr'] *= 0.9
        elif action == 1:
            # Modify latent vector
            noise += torch.randn_like(noise) * 0.01
        elif action == 2:
            # Perturb target points
            points += torch.randn_like(points) * 0.01

        fake = self.generator(points)

        predicted_points = self.pre_trained_cnn(fake)

        mse = nn.MSELoss()(predicted_points, points21).item()
        reward = -mse

        new_state = (predicted_points - points21).cpu().numpy().flatten()

        critic_real, _ = self.critic(img_real, points21).reshape(-1)
        critic_fake, _ = self.critic(fake, points21).reshape(-1)

        done = torch.mean(critic_fake)>torch.mean(critic_real)

        return new_state, reward, done

    def render(self):
        pass












