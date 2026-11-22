# Report

## Objective

[docs](docs/) has the details of this project. The main overview is that we have an environment where we are testing different Agents and see how they're performing. Currently we have two main RL algorithms implemented: [dqn](engine/aresim/algorithms/dqn/) , [ppo](engine/aresim/algorithms/ppo/) , [plan2explore](engine/aresim/algorithms/plan2explore/) , [dreamerv3](engine/aresim/algorithms/dreamerv3/) , [recurrent_dqn](engine/aresim/algorithms/recurrent_dqn/) . We have other model called [llm_dm](engine/aresim/algorithms/llm_dm/) in which LLMs acts as Learner along with memory module and either Decision Models or LLMs acts as Actors.

We already have the paper in [report](report/) for IJCAI 2027 (https://2027.ijcai.org/) on this AresSim project. Below are my requirements and details of the content in the paper. Go through all of it, project and udpate the paper writing accordingly. The paper in existing state has all the things covered but I still fele like the write-up is not upto the mark and feels more AI generated. I would like you to focus on optimizing or re-writing and re-organizing all the paper contents well and make it perfect. 

## Abstract, Title

- Title = AresSim: A Benchmark for Long-Horizon Resource Logistics under Partial Observability

### Abstract 
A rover that can find and collect a resource still has to bring it home. We introduce AresSim, a deterministic Mars rover environment in which agents explore procedurally generated terrain, collect and deliver multiple resources, and maintain a landing site while managing battery and colony survival. Agents act on rover-local observations, and recorded trajectories allow individual decisions to be replayed. In preliminary evaluations on 20 validation worlds, a scripted controller completed 13 missions, while the strongest learned checkpoint in the deterministic evaluation completed two. AresSim supports analysis of where a mission breaks down, from resource discovery and collection to return navigation and delivery.

## Sections

- Abstract
- Introduction
- Related Work
- AresSim
- RL Task Formulation
- Algorithms
    - Baseline
    - RL
        - PPO
        - DQN
        - Plan2Explore
        - Recurrent DQN
        - MuZero
        - DreamerV3
    - LLMs and Decision Models
- Training
- Results
    - Optimizing Reward Objective
    - Quantitative Analysis
    - Qualitative Analysis
    - Ablation Study
- Limitations
- Conclusion
- References

## Assets

### Figures
- Screenshot of AresSim from different views (My task)
- State Design (Draw.io) - Use CIons, Illustrations of different information we have in the State. Design based on their dimensions, you could even take some screenshots from our env if at al needed.
- RL Algorithms collage (Draw.io) - I want you to understand how each algo works, find core building blocks and design each algorithm illustration based on those group of building blocks
- LLM-DM formulation (Draw.io) - Understand our Learner, Memory, Actor pipeline and design this pipeline.
- Comparison of: (I would suggest to write a script to make these plots from already present results with us)
    - RL algos 
    - LLMs, LLM-DM - Placeholder, didn't do full analysis yet
    - LLMs, LLM-DM, RL algos - Include only Gemini-4-Argon:Jev for now from LLM-DM list. Will be adding more later on
    I want below plots for above comparisons
        - Losses
        - Path Heatmaps (Python)
        - Path AUC plot (Python)
        - Samples, Ice collection + Delivery (Python)
        - Feel free to suggest other plots that you think is worth having
- Ablation sutdy analysis plots of PPO (Python)

### Tables
- Reward
- Terrains
- Actions
- Battery Drainage
- Weather
- RL Algorithms Arch + Training details
- Ablation changes for PPO

### Equations (More in https://www.tensorwrites.com/posts/aressim-env)
- Solar Generation, Power Margin, Battery Drain
- Reward function


## Rules
- Make sure to follow paper's rules and instructions.
- The figures, tables and equations should be very well designed
- All the content should follow ASD-STE100 like writing style. Use the Skill at [research-paper-writing](.cursor/skills/research-paper-writing) for understanding how to write a research paper.
- Ensure all the references are correctly mentioned and nothing is left behind
- Keep placeholders for the places where you don't have the data and do let me know about it.
- For Results Figures, use appropriate type of plots for each of the results. They should be intuitive and easy to understand.
- The content in each section should be precise, accurate. Try to go through some related research papers to understand how to write the paper in this area.
- Feel free to re-organize, add, remove ro update the sections, sub-sections. Those are my proposals
- It is ok if we go beyond the page limit. let's keep it very extensive and detailed in draft and we can later remove or optimize things. Include as many details as you want to. make sure all the plots, tables, equations are useful, well designed, easy to interpret.