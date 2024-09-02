
import os
from DQN import DQNAgent
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataholder import get_loader
from gradientpenalty import gradient_penalty
from model_v2 import Critic, Generator, initialize_weights
from PreTrainCNN import PretrainedCNN
from ENV import ENV
from utilis import save_logging, plot, set_seed
# torch.autograd.set_detect_anomaly(True)
# tensorboard --logdir=dqn3 --port 8123

save_dir = ('E:/GANDQN/dqn5/')
os.path.exists(save_dir) or os.makedirs(save_dir)

set_seed(42, save_dir=save_dir)
dataindex = 151
Batch_size = 64
nc = 1
image_size = 64
features_d = 64
features_g = 64
Z_dim = 150
num_epochs = 1000
lr = 1e-4
lr2 = 1e-4
beta1 = 0.5
Samplesindex = 10
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
z_dim_1 = 50
z_dim_2 = 50
num_conditions = 150  # equal to len(data) how many points per row

warmup_epochs = 120
img_list = []
G_loss = []
C_loss = []
dqn_no_improvement_counter = 0
dqn_best_performance = float('inf')
dqn_patience = 15
best_gen_loss = float('inf')
best_critic_loss = float('inf')
action_interval = 30
reward_log = []
z1_log = []
z2_log = []


device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")


# load the pre-trained dynamic weight CNN+MLP model
pre_trained_cnn = PretrainedCNN()
pre_trained_cnn.load_state_dict(torch.load('logs/Model_epoch_470.pt', map_location=device))
pre_trained_cnn = pre_trained_cnn.to(device)
pre_trained_cnn.eval()


# load the data as conditions for the generator and critic
dataloader = get_loader(img_size=image_size,
                        batch_size=Batch_size,
                        z_dim=Z_dim,
                        points_path="E:/GANDQN/Training_data/1501eo0.csv",
                        img_folder=r'C:/Users/Administrator/Desktop/pythonProject/pr1/Training_Data/image/new',
                        points_original="E:/GANDQN/Training_data/np1e0_endemp.csv",
                        shuffle=False,
                        dataindex=dataindex)

# Initialize the generator and critic
netG = Generator(channels_noise=Z_dim, z_dim1=z_dim_1, z_dim2=z_dim_2, nc=nc, features_g=features_g,
                 num_conditions=num_conditions).to(device)
netD = Critic(image_size=image_size, nc=nc, features_d=features_d).to(device)
initialize_weights(netG)
initialize_weights(netD)

# Setup Adam optimizers for both G and D
opt_gen = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-4)
opt_critic = optim.Adam(netD.parameters(), lr=lr2, betas=(beta1, 0.999))

writer_real = SummaryWriter(f"{save_dir}/real")
writer_fake = SummaryWriter(f"{save_dir}/fake")
writer_reward = SummaryWriter(f"{save_dir}/reward")
writer_interpo = SummaryWriter(f"{save_dir}/interpolation")
# writer_cluster = SummaryWriter(f"{save_dir}/cluster")
writer_latent = SummaryWriter(f"{save_dir}/latent")
step = 0

env = ENV(generator=netG, critic=netD, pre_trained_cnn=pre_trained_cnn, device=device, num_layers=len(netG.net),
          opt_gen=opt_gen, stability_threshold=0.1, replay_buffer_size=10000, batch_size=64,stability_epochs=20)
dqn_agent = DQNAgent(state_dim=21, action_dim=5)
netG.train()
netD.train()

print('Starting training---')
for epoch in range(num_epochs):
    reward_sum = 0.0
    reward = 0.0  # Initialize reward in case action is not taken this epoch
    num_batch = 0
    for i, (points, img_real, points21, points_original) in enumerate(dataloader):
        # print(f"Batch {i} size before model: {points.size(0)}")
        points = points.to(device, dtype=torch.float)
        points21 = points21.to(device, dtype=torch.float)
        points_original = points_original.to(device, dtype=torch.float)
        cur_batch_size = points.shape[0]
        z1 = torch.randn(cur_batch_size, z_dim_1, 1, 1).to(device)
        z2 = torch.randn(cur_batch_size, z_dim_2, 1, 1).to(device)
        img_real = img_real.to(device, dtype=torch.float)

        if epoch == 0 and i == 0:
            state = env.reset()  # Initial state reset only at the start of training
        fake = netG(points, z1, z2, points21)

        if epoch < warmup_epochs:
            # Training the Critic and Generator without DQN influence
            for _ in range(CRITIC_ITERATIONS):
                critic_real = netD(img_real, points21).reshape(-1)
                critic_fake = netD(fake, points21).reshape(-1)
                gp = gradient_penalty(netD, points21, img_real, fake, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                netD.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            gen_fake = netD(fake, points21)
            loss_gen = -torch.mean(gen_fake)
            netG.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        else:
            # Train Critic and Generator as usual
            for _ in range(CRITIC_ITERATIONS):
                critic_real = netD(img_real, points21).reshape(-1)
                critic_fake = netD(fake, points21).reshape(-1)
                gp = gradient_penalty(netD, points21, img_real, fake, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                netD.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            gen_fake = netD(fake, points21).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            netG.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        if i % 200 == 0 and i > 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {loss_critic:.6f}, loss G: {loss_gen:.6f}")

            with torch.no_grad():
                fixed_noise_z1 = torch.randn(Samplesindex, z_dim_1, 1, 1).to(device)
                fixed_noise_z2 = torch.randn(Samplesindex, z_dim_2, 1, 1).to(device)
                fixed_points = torch.randn(Samplesindex, Z_dim, 1, 1).to(device)
                points21_fixed = points21[:Samplesindex]

                fake = netG(fixed_points, fixed_noise_z1, fixed_noise_z2, points21_fixed)

                img_grid_real = torchvision.utils.make_grid(img_real[:Samplesindex], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:Samplesindex], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                writer_real.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)
                writer_real.add_scalar("Loss/Critic", loss_critic.item(), global_step=step)
                # writer_reward.add_scalar("Reward", reward, global_step=step)
                # writer_reward.add_scalar("Average Reward", avg_reward, global_step=step)

                # num_steps = 5  # Adjust as needed
                #
                # # Select two random latent vectors for z1 and z2
                # z1_a = torch.randn(1, z_dim_1, 1, 1).to(device)
                # z1_b = torch.randn(1, z_dim_1, 1, 1).to(device)
                #
                # z2_a = torch.randn(1, z_dim_2, 1, 1).to(device)
                # z2_b = torch.randn(1, z_dim_2, 1, 1).to(device)
                #
                # # Fixed conditions for the interpolation
                # conditions = points21_fixed[:1]  # Use a single condition for simplicity
                #
                # # Interpolation loop
                # for alpha in np.linspace(0, 1, num_steps):
                #     z1_interp = (1 - alpha) * z1_a + alpha * z1_b
                #     z2_interp = (1 - alpha) * z2_a + alpha * z2_b
                #
                #     # Generate the interpolated output
                #     fake_interp = netG(fixed_points[:1], z1_interp, z2_interp, conditions)
                #
                #     # Log the interpolated images to TensorBoard
                #     img_grid_fake = torchvision.utils.make_grid(fake_interp, normalize=True)
                #     writer_interpo.add_image(f"Interpolated Images alpha={alpha:.2f}", img_grid_fake, global_step=epoch)

            if loss_gen.item() < best_gen_loss:
                best_gen_loss = loss_gen.item()
                torch.save(netG.state_dict(), save_dir + 'best_netG.pt')

            if loss_critic.item() < best_critic_loss:
                best_critic_loss = loss_critic.item()
                torch.save(netD.state_dict(), save_dir + 'best_netD.pt')

            if epoch % 10 == 0:
                torch.save(netG.state_dict(), save_dir + f'netG_epoch{epoch}.pt')
                torch.save(netD.state_dict(), save_dir + f'netD_epoch{epoch}.pt')

            G_loss.append(loss_gen.item())
            C_loss.append(loss_critic.item())

            step += 1


    if epoch >= warmup_epochs and epoch % action_interval == 0:
        avg_reward = reward_sum / len(dataloader)  # Compute average reward over the epoch
        print(f"Epoch {epoch} DQN action")
        action = dqn_agent.act(state)
        new_state, reward, done = env.step(action, z1, z2, points, points21, points_original,epoch,warmup_epochs,num_sample=50)
        dqn_agent.store_transition(state, action, reward, new_state, done)
        dqn_agent.replay()  # Train DQN only after the action interval
        state = new_state  # Update state only when action is taken

        reward_sum += reward
        num_batch += 1

        reward_log.append(reward)
        z1_log.append(z1.detach().cpu().numpy())
        z2_log.append(z2.detach().cpu().numpy())


        writer_reward.add_scalar("Reward", reward, global_step=epoch)
        writer_latent.add_histogram("z1", z1, global_step=epoch)
        writer_latent.add_histogram("z2", z2, global_step=epoch)


        if reward < dqn_best_performance:
            dqn_best_performance = reward
            dqn_no_improvement_counter = 0
        else:
            dqn_no_improvement_counter += 1

        if dqn_no_improvement_counter > dqn_patience:
            print("No improvement in DQN performance for {} epochs, stopping DQN.".format(dqn_patience))
            dqn_stopped = True

        if epoch >= warmup_epochs:
            dqn_agent.save(save_dir + f'dqn_agent_epoch{epoch}.pt')



print('Training completed')
print(f"logs saved in {save_dir}")
save_logging(save_dir,reward_log, z1_log, z2_log)
plot(G_loss, C_loss, plot = True,save_dir=save_dir)



