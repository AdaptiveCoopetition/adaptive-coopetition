# Adaptive Coopetition (AdCo)
## Abstract
Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While several existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although external, high-performing verifiers can detect reasoning errors, making them reliable requires substantial training efforts. To address these challenges, we introduce a novel inference-time multi-round, multi-agent framework in which LLM agents utilize an adaptive, UCB-based 'coopetition' mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on math reasoning benchmarks, delivering a relative improvement over baselines. Extensive experiments demonstrate that our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided 'coopetition' framework significantly enhances overall reasoning robustness by leveraging diverse prior knowledge and reasoning traces, also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems.
## Installation
AdCo requires the installation of [AutoGen](https://github.com/microsoft/autogen) and Python (3.10 or later).
## Citation
If our project is helpful for your research or applications, please cite our work using this BibTex:
```
@misc{miin2025adaptivecoopetition,
      title={Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning}, 
      author={Anastasia Miin and Wendy Yaqiao Liu and Rui Jerry Huang},
      year={2025} 
}
```
## Authors
By Anastasia Miin, Wendy Yaqiao Liu, Rui Jerry Huang, 2025.
