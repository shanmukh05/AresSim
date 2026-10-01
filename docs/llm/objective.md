# Objective

Build an experience-driven agent learning system that improves through repeated interaction with simulation games, without training the LLM during gameplay.

The LLM should learn from recorded outcomes: decide which parts of the visible state and history matter, propose better ways to preprocess them, and turn useful experience into compact, testable guidance. A fast actor—such as Jev, Laya, or a simple controller—can use that guidance to choose legal actions. A direct LLM actor remains a comparison.

Start in AresSim, then test whether the *learning method* transfers to other games. Compare OpenAI, Anthropic, Gemini, and OpenRouter models on both performance and the policies they discover. Measure success, safety, latency, and total cost on unseen scenarios; keep simulator rules and hidden state outside the agent's control.
