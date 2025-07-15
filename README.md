# Call-Centre-Routing-Optimisation

This study applies reinforcement learning (RL) to optimise routing policies in a simulated call centre, aiming to reduce client waiting time, staff idle time, and abandonment rates. The routing problem is modelled as a Markov Decision Process (MDP) within a Skill-Based Routing (SBR) framework. A simulation model is developed by integrating Discrete-Event Simulation (DES) with OpenAI Gym captures dynamics such as arrivals, abandonments, service duration, staff wrap-up phases, and callbacks mechanism. Real-world call centre data is used to model the system configuration and fit distributions for inter-arrival, abandonment, and service durations, while literature guides assumptions on wrap-up and callback mechanisms. Parameters are scaled to simulate high-load scenarios, with action masking used to filter invalid routing choices. Random and Maskable Proximal Policy Optimisation (PPO) policies are evaluated 1,000 runs in simulation model. Maskable PPO outperforms the baseline across most metrics, reducing abandonment and idle time through prioritising live clients for the fastest staff. Although callback wait time increases slightly, number of callbacks decrease improve overall efficiency.


`Call_Centre_Project.ipynb`
- Contains the main notebook for model exploration, training the Maskable PPO agent, and evaluating policy performance.

`Call-Center-Dataset.xlsx`
- PwC call centre dataset sourced from Kaggle. The simulation’s configuration and statistical distribution fitting are based on this data.

`CallCentreEnvironment.py`
- Custom simulation environment built using the OpenAI Gym interface. It defines the event-driven logic and is used for both training and evaluation of RL policies.

`Project_Data/`
- Stores pre-generated simulation data, including training histories, evaluation results, and saved PPO models.

`PwC data EDA and Distribution Fitting.ipynb`
- Notebook performing exploratory data analysis (EDA) and identifying best-fit probability distributions for inter-arrival, abandonment, and service durations using the PwC dataset.
