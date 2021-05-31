# Discovery and Learning of Navigation Goals from Pixels in Minecraft

Original repo: https://github.com/juanjo3ns/mineRL

Website 1: https://imatge-upc.github.io/PiCoEDL/

Website 2: https://imatge-upc.github.io/PixelEDL/

## Initial set up:

```
git clone update_with_upc_repo
conda create --name minerl python=3.6
conda activate minerl
conda install --file requirements.txt
```

To download the the MineRL dataset we need to set up the environment variable `$MINERL_DATA_ROOT` to the `data` path where we want to store all the trajectories.
Then, we can execute:
`python -m minerl.data.download 'MineRLNavigate-v0'`
You can download any other set of trajectories, check them out in this website: https://minerl.io/docs/environments/

## Disclaimer before training:

* Make sure to create an account in [Weights and Biases](https://wandb.ai/home) where you will log all of our metrics. Also remember to log in with your account in the server where you train your models.
* Take into account that MineRL library and PFRL are slightly modified. Instead of installing them through conda we cloned them within our repository. 
  - Most changes in MineRL are within `minerl/env/core.py` file and `minerl/viewer_custom/` folder. 
  - Changes in PFRL are within `pfrl/experiments/evaluator.py` and `pfrl/experiments/train_agent.py` files.



## Steps to reproduce a simple experiment:
In this example we reproduce one of the simplest experiments. The goal of the agent is to discover different regions in the state space by itself. Then in a second phase, the agent is encouraged to learn conditioned policies to reach those discovered regions.
1. [OPTIONAL] This step is required only if you want to discover skills from a custom map. 
2. 
    **Map creation:**
    First, create a flat map with 9 different regions in Minecraft. To do so, change the directory to:
    `cd src/minerl/env/Malmo/Minecraft`
    
    And execute:
    
    `./launchClient.sh`
    
    This will create a Minecraft instance. Create a map with a name. E.g: `CustomWorld_Simple`. Once it is created you may not be able to control the agent. In that case, open another Minecraft instance by executing `python3 -m minerl.interactor 6666` from another terminal and join the created by map opening the world to your LAN.
    
    Now, within Minecraft, press Ctrl+T to type commands inside the game. First, we move the agent to the center of the map by executing:
    
    `/tp 0 4 0`
    
    In case you don't have permissions to do it, first change the game mode to creative:
    
    `/gamemode creative`
    
    Then, we can create the different regions that contains our map. We generate larger regions than we need, so that the agent still sees the same regions in the boundaries of the map. (Which by default are set at 50 in all directions).
    ```
    /fill 15 3 15 100 3 100 sand
    ...
    /fill -100 3 -100 -15 3 -15 gold_block
    ```
    
    Remember to save the game from the instance launched first, this one will store the state of the game in this folder: `src/minerl/env/Malmo/Minecraft/run/saves/CustomWorld_Simple`
    
    **Generate trajectories:**
    
    First, generate random trajectories in a flat bounded map with 9 different regions. 
    
    - Modify your `ABSOLUTE_PATH` within the `./src/main/generate_trajectories.py` script. 
    
    - Make sure that the `world` value in `configs/train_0.yml` matches the name of the map created in the previous steps.
    
    - Finally, execute the following command to generate 500 trajectories with a frame skipping of 10:
    
    `python -m main.generate_trajectories train_0`

    **Plot generated trajectories:**
    
    Open `src/jupyter/3d_trajectories.ipynb` and execute the first five cells. Then in a new cell execute:
    ```
    goal_states = []
    initial_state = np.array([0,0], dtype=np.float32)
    experiment = 'name_of_your_folder_trajectories'
    t = Trajectory2D(experiment, initial_state, goal_states, fix_lim=True)
    t.plot_all_together()
    ```
2. Then in the skill-discovery phase, we learn the mappings from the distribution of the states to the latents.
In this step, you should modify either the `configs/curl.yml` or `configs/vqvae.yml` files, depending on the method you want to try. In this example, we proceed with the variational case (`vqvae.yml`). Some values that you might need to modify:
    - **experiment**: unique identifier for your experiment logs
    - **trajectories**: folder name where you stored the generated trajectories
    - **hyperparams**: try the default ones or modify with your own criteria

    Finally, execute:
    
    `python -m main.vqvae_train vqvae`
3. To assess this training we generate the index map and the centroids reconstructions.
    
    **Index map**
    
    - To generate the index map you should update the `path_weights` key in the `configs/vqvae.yml` file. E.g:
    
    `path_weights: 'vqvae_example/mineRL/3cygsbys/checkpoints/epoch=16-step=6374.ckpt'`
    
    - Also make sure to put `index` in the `type` value.

    Once we have something like this:
    ```
    ...
    test:
        type: 'index'
        shuffle: no
        limit: null
        path_weights: 'vqvae_example/mineRL/3cygsbys/checkpoints/epoch=16-step=6374.ckpt'
    ```
    Execute:
    
    `python -m main.vqvae_test vqvae`
    
    **Centroids reconstruction** (Only available for variational trainings)
    
    Open `src/jupyter/CustomVQVAE_Centroids.ipynb` and execute all the cells. Make sure to update the `path` variable with the weights full path.
    
4. In the skill-learning phase, we learn and condition our agent's policy with the categorical latents discovered in the previous phase.
    
    For executing this training, you should modify the configuration file: `configs/train_0.yml`.
    ```
    outdir: 'rl.test.0.0' # unique identifier of your training, it creates a folder inside ./src/results
    env: 'MineRLNavigate-v0' # runs navigate mission, although we don't aim to solve this task
    world: 'CustomWorld_Simple' # world name created at step 1
    ```
    Leave the rest of hyperparameters with the default values.
    
    Then, also modify the encoder dictionary with the name and epoch of the previous training.
    
    Finally, execute:
    
    `python -m main.train_rainbow train_0`
    
    
    These are a few reasons why your simulator might be running too slow:
    - `interactor: yes -> no`
    - `monitor: yes -> no`
    - Simulator executing in less than 4 cpu cores.
5. Evaluate the learned skills and generate some videos.
    
    Open `configs/val_0.yml` file and make sure it looks like this:
    ```
    ...
    demo: yes                                   # evaluation mode
    monitor: yes                                # stores a video of the trajectory followed by the agent
    load: './results/train_rl.test.0.0/best'    # folder where we stored best policy weights
    eval_n_runs: 100                            # if we have 10 different regions, this runs 10 evaluations for each one 
    ...
    ```
    The rest of the parametres should be all the same as we had in `configs/train_0.yml`.
    
    Then, execute:
    
    `python -m main.train_rainbow val_0`
    
    Once it finishes, we can find the generated videos within this folder: `src/results/eval_rl.test.0.0/MineRLNavigate-v0/`.
    
    We can also generate the top-view of the trajectories and the average reward over time.
    
    To do that, open again the `src/jupyter/3d_trajectories.ipynb` jupyter file.
    
    Add one more cell with the following content:
    ```
    goal_states = []
    initial_state = np.array([0, 0], dtype=np.float32)
    experiment = 'eval_rl.test.0.0'
    t = Trajectory2D(experiment, initial_state, goal_states, fix_lim=True)
    t.plot_per_goal_state()
    t.plot_smooth_reward_per_goal_state()
    ```

## Main references:
- MineRL library: https://github.com/minerllabs/minerl
- RL Baselines: https://github.com/pfnet/pfrl
- Contrastive inspired from: https://github.com/MishaLaskin/curl
- Variational inspired from: https://github.com/zalandoresearch/pytorch-vq-vae
