# Master-Of-Mazes: Robot Warehouse Navigation

This project is a classical example of how Reinforcement Learning can be implemented to real-world logistics scenario by building custom environments, policy learning and Intelligent agent behaviour classification.

This project simulates a warehouse environment where an autonomous robot learns to navigate through the environment, collect objects and deliver them to a designated drop point using Proximal Policy Optimization(PPO) algorithm, while actively dodging obstacles in it's path present in a dynamic 6×6 grid.

Coming to the approach, Proximal Policy Optimization(PPO) algorithm was used, which is a classic Reinforcement Learning algorithm. It enables balance between performance and computational efficiency by limiting drastic updates to policy training, which helps avoid instability and encourages steady learning. It makes it ideal for learning pick and deliver strategies in discrete action spaces with sparse reward, penalty mechanism.

![image](https://github.com/user-attachments/assets/ea284aae-91fa-4821-9f6e-defc03793a9d)

### 🚀 Key Highlights

* Custom environment was built with obstacles placed at random points.
* The core purpose was to collect items, and deliver them to the destination points.
* A reward-penalty system was designed to train the agent to work efficiently and dodge obstacles.
* The PPO agent was trained using Stable-Baselines3 over 500,000 steps.
* Performance visualizations, including heat maps and delivery analysis, were generated.


  ### 🧾 Reward Structure

| 🧠 Agent Action           | 💰 Reward Value | 🎯 Purpose                                   |
|---------------------------|----------------|----------------------------------------------|
| Move one step             | -1             | Penalize unnecessary movement                |
| Pick up an item           | +10            | Encourage locating and collecting items      |
| Deliver item to goal zone | +100           | Strongly reward task completion              |
| Exceed max steps          | Episode ends   | Prevents inefficient or endless wandering    |

### 📊 Results
📦 Delivery Performance Summary
Over 50 evaluation episodes, the trained PPO agent demonstrated the following performance:
| Delivery Count | Episodes | Total Deliveries  |
| -------------- | -------- | ----------------- |
| 2 Deliveries   | 17       | 34                |
| 1 Delivery     | 24       | 24                |
| 0 Deliveries   | 9        | 0                 |
| **Total**      | **50**   | **58 Deliveries** |

* The agent consistently completed the task in most episodes, with an average of 1.16 deliveries per episode.
* The highest efficiency was observed in episodes where the agent achieved 2 successful deliveries with high rewards (e.g., 255.99, 256.22).
* A few episodes had 0 deliveries, indicating potential failure in navigation or pickup. Hence refinement of training is possible.

The following iference was made after training the PPO agent for 500,000 steps.
| Criteria                | Observation                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| **Item Pickup**         | The agent successfully learned to identify and collect green item locations. |
| **Delivery Completion** | Items were delivered to the target zone (bottom-right corner) consistently.  |
| **Obstacle Avoidance**  | The agent exhibited improved path selection by avoiding black obstacles.     |
| **Efficiency**          | Reduced idle movements over time, optimizing delivery routes.                |

Visit Frequency Heatmap
* The visit heatmap shows high-density movement around the item spawn zones and the delivery area.
* Sparse visits to obstacle-heavy regions indicate learning to avoid penalties.
![image](https://github.com/user-attachments/assets/b0dc4318-14fd-4479-859c-03a1661b50ea)

Evaluation Performance Plot
* The reward curve steadily increased, showing progressive learning over episodes.
* Delivery-related spikes in reward (~+100) confirm successful item drops.
* Occasional drops signify early exploration or failed deliveries.
  ![image](https://github.com/user-attachments/assets/6063d35d-82d7-4845-a246-35ade04ec5e7)

### Conclusion
This project aims at demonstrating how effective RL-based control strategy is in a simulated warehouse setting. By leveraging PPO algorithm we got the purpose achieved. Although our training showed significant performance for 500,000 steps. There is still room for improvement by refinimg the model and agent training maybe by increasing the steps to a million or 2 million. Eventhough I have attempted for those time steps, I saw significant better performance but don't have concrete results to back it and this aspect is a possible extension that can be made better at a later time in the future.
