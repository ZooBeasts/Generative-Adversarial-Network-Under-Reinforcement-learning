
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import os

save_dir = 'results/'
os.path.exists(save_dir) or os.makedirs(save_dir)

def save_logging(save_dir = None, reward_log=None, z1_log=None, z2_log=None):
    reward_df = pd.DataFrame(reward_log, columns=['Reward'])
    reward_df.to_csv(save_dir + 'reward_log.csv', index=False)

    # Save z1 and z2 logs
    z1_df = pd.DataFrame([z.flatten() for z in z1_log])
    z1_df.to_csv(save_dir + 'z1_log.csv', index=False)

    z2_df = pd.DataFrame([z.flatten() for z in z2_log])
    z2_df.to_csv(save_dir + 'z2_log.csv', index=False)


def plot(G_loss=None, C_loss=None, plot=True, save_dir = save_dir):
    G_loss = np.array(G_loss)
    C_loss = np.array(C_loss)
    np.savetxt(f'{save_dir}/G_loss.csv', G_loss, delimiter=',')
    np.savetxt(f'{save_dir}/C_loss.csv', C_loss, delimiter=',')
    if plot:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Critic Loss ")
        plt.plot(G_loss, label="G Loss", color='blue')
        plt.plot(C_loss, label="C Loss", color='red')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig('GAN_GC_loss.png')
        plt.show()

def set_seed(manualSeed = None, save_dir = save_dir):
    # manualSeed = 42
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
    print("CUDA is available: {}".format(torch.cuda.is_available()))
    print("CUDA Device Count: {}".format(torch.cuda.device_count()))
    print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
    with open(os.path.join(save_dir, 'seed.txt'), 'w') as f:
        f.write(f"Random Seed: {manualSeed}\n")







