# ✨ Yi Jin Jing: An LLM-driven financial service solution integrating data foundation and intelligent investment


## 🧠 Project Background & Design Philosophy

The name “Yi Jin Jing” is inspired by the ancient Chinese classic *Yi Jin Jing*.

*Yi Jin Jing* emphasizes conditioning the tendons and unblocking meridians to reshape the body's internal operating structure. In this project, it serves as a metaphor: financial texts, market data, knowledge graphs, and intelligent models are viewed as the “meridian system” of financial information. By deeply integrating LLMs, knowledge bases, and intelligent investment models, we reconstruct the process of collecting, organizing, and transmitting financial knowledge, enabling a smoother, more efficient, and more intelligent flow and transformation of financial information within the system.

Yi Jin Jing is built upon generative AI, knowledge graphs, and graph machine learning technologies to provide an LLM-driven financial service solution that integrates a knowledge base and intelligent investment. Through the FinEX financial knowledge extraction agent, the system performs financial text analysis, financial event extraction, and structured knowledge generation from news, announcements, and public sentiment. Based on a temporal multimodal knowledge graph, it constructs an automated multimodal data foundation. With the self-developed MEHGT multimodal heterogeneous graph neural network prediction model, it generates trading signals and supports intelligent asset allocation. Ultimately, it forms a closed-loop system of “knowledge extraction — data foundation — intelligent decision-making.”



---

## 📁 Project Structure

This repository adopts a layered directory structure based on functional modules, enabling independent development and joint debugging.

    YiJinJing/
    ├── MEHGT model/
    │   ├── HGTConv.py
    │   ├── han_conv_edge_attr3.py
    │   ├── readme.md
    │   ├── 回测demo1.ipynb
    │   ├── 浪潮信息_Grid_顺序.py
    │   └── figs/                     # Stores MEHGT-related diagrams & backtest visualizations
    │
    ├── FinEX/
    │   ├── Merge.sh                  # LoRA weight merging script
    │   ├── Quant.sh                  # FinEX quantization & inference script
    │   ├── SFT_TF-14B.sh             # SFT (supervised finetuning) main script
    │   ├── Web_demo.sh               # Web Demo startup script
    │   └── readme.md
    │
    ├── a series of multimodal knowledge graphs/
    │   ├── HeteroG_eventall2.0.ipynb # Event-level hetero graph construction & visualization
    │   ├── readme.md
    │   ├── toexcel_gpu1.ipynb        # Graph export & feature warehousing
    │   └── 比亚迪（21-24）_deldata3.xlsx # Industry / stock example data
    │
    ├── LICENSE
    ├── README.md
    └── requirements.txt              # Python dependency environment

---

## 🔧 Runtime Environment & Dependencies

The project is primarily built on the Python ecosystem. Core dependencies include (but are not limited to):


- Deep learning & LLM frameworks：`PyTorch`、`transformers`、`peft`、`accelerate`
- Graph neural networks：`PyTorch Geometric` and extensions
- Data processing & visualization：`pandas`、`numpy`、`matplotlib`、`seaborn`
- Web & deployment：`gradio` or `fastapi`（depending on integration method）

It is recommended to create an isolated environment using `conda` or `venv` and install all dependencies with:

    conda create -n yjj python=3.10
    conda activate yjj
    pip install -r requirements.txt

---

# 🧩 Module 1: FinEX (Golden Bean) — Financial Knowledge Extraction Agent

## 📖 1.1 Overview

FinEX (Golden Bean) is a knowledge extraction and structured representation agent designed for the financial domain. It is fine-tuned on the Qwen (Tongyi Qianwen) large model via LoRA-based supervised finetuning. Using high-quality annotated financial corpora and task-specific instructions, it automatically identifies key information (financial events, financial entities, financial actions, etc.) from unstructured texts such as financial news, announcements, and research reports, generating structured tuples (triples/pairs) ready for knowledge storage and graph construction.


## 🧱 1.2 Technical Details & Implementation Roadmap (with diagrams)

-  **Finetuning Framework:** FinEX training follows and extends the LLaMA Factory framework. Training scripts and parameter configs can be found in `FinEX Finetuning/` `.sh` files. Source framework:
  
  - LLaMA Factory: <https://github.com/hiyouga/LLaMA-Factory>

- **Model Base & Configuration:**
  
  - Uses a bilingual financial-enhanced LLM (Tongyi-Finance 14B) as the base.
  - Adopts LoRA/QLoRA for parameter-efficient finetuning, focusing on output and some intermediate layers.
  - Full model weights and config:
    
    - ModelScope: <https://www.modelscope.cn/models/Madness977/FinEX>

- **Task Modeling & Instruction Design:**
  
  - Financial event extraction is formulated as an instruction-driven generation task, with output formats constrained by prompts.
  - Supports multi-level outputs: event triples, indicator pairs, multi-event joint parsing.

- **Key Training Scripts:**

    - `SFT_TF-14B.sh`：Main SFT workflow (data loading, training strategy, distributed training)
    - `Merge.sh`：Merge LoRA weights with base model
    - `Quant.sh`：4bit/8bit quantization & deployment optimization
    - `Web_demo.sh`：Launch FinEX web demo

- **Architecture Diagram**

    ![FinEX 模型总体架构示意图](figs/FinEX.png)

- **Web Deployment Example (Bilingual)**  
    ![FinEX 网页部署推理示例](figs/中文知识抽取示例.png)

---

## 📊 1.3 Evaluation & Visualization (with diagrams)

- **Tasks:** Financial event extraction & structured tuple generation.
- **Metrics:**
  
  - Text-level: Precision / Recall / F1, sentence-level parsing success rate
  - Structure-level: Triple recall, entity alignment accuracy, event coverage

- **Example Findings:**
  
  - On financial announcements & news datasets, FinEX outperforms baseline LLMs and non-finetuned models across Precision / Recall / F1.
  - Shows strong capability in parsing long texts with multiple events.
- **Example Comparison Results:**

    ![FinEX 事件抽取与结构化表示效果示例](figs/NLP结果.png)

---

# 🌐 Module 2: Temporal Multimodal Knowledge Graph

## 📖 2.1 Overview

This module uses FinEX output tuples, combined with market data, financial indicators, industry classification, and other information sources, to map discrete financial events, entities, and market signals into a temporal heterogeneous graph. It provides a “computable, inferable, traceable” data foundation for downstream prediction models.


Key focuses:

- Multimodal fusion (text, numerical, graph structure)
- Temporal sliding updates & version management

## 🧱 2.2 Technical Flow (with diagrams)

-  **Graph Construction Pipeline:**
  
  1. Entity Alignment: Normalize company names, stock tickers, industry labels.
  3. Temporal Slicing: Build daily graph snapshots or sliding-window block graphs.
  4. Graph Storage & Export: Using `HeteroG_eventall2.0.ipynb` & `toexcel_gpu1.ipynb` for construction, visualization, and exporting features.

- **Related Notebooks:**
  
  - `HeteroG_eventall2.0.ipynb`: Construction & visualization of temporal multimodal hetero graphs
  - `toexcel_gpu1.ipynb`：Export graph features to tables/feature warehouse
  - `BYD（21-24）_deldata3.xlsx`：Example dataset


## 📊 2.3 Example


- **neo4j-based Visualization Example:**

    ![时序多模态知识图谱可视化与应用示例](figs/neo4j.png)

---

# 📈 Module 3: MEHGT — Multimodal Edge-Enhanced Heterogeneous Graph Transformer

## 📖 3.1 Overview

MEHGT (Multimodal Edge-enhanced Heterogeneous Graph Transformer) is a GNN model featuring:

- A heterogeneous graph transformer architecture for diverse node/edge types
- Edge-level multimodal information (text events, numerical indicators, sentiment signals)
- Temporal modeling via sliding windows & backtesting for trend and risk prediction


## 🧱 3.2 Technical Details (with diagrams)

- **Model Structure:**
  
  - **Input:** Heterogeneous graph sequences (companies, events, industries) with multimodal edge features 
  - **Graph Encoding:** MEHGT / HGTConv / HANConv layers to learn type-specific projections and edge-enhanced attention  
  - **Temporal Modeling:** Transformer / TCN / BiLSTM over multi-day graph embeddings 
  - **Output:** Stock trend classification / risk labels + attention weights for interpretability

## 📊 3.3 Prediction & Backtest Evaluation (with diagrams)

- **Prediction Tasks:**
  
  - Short-term stock/industry trend prediction  
  - Detection of abnormal volatility around risk events  

- **Metrics:**
  
  - Classification: Accuracy, Precision, Recall, F1, AUC, MCC 
  - Backtesting: CRR, MDD, Sharpe, win rate, turnover 

- **Backtest Implementation:**
  
  - Logic implemented in `Backtest_demo1.ipynb`, including net value curve, drawdown curve, and comparisons  
  - Comparison with baseline strategies (buy-and-hold, factor models)

- **Example Results:**

    ![MEHGT 回测结果与指标对比示意图](figs/对比结果.png)
    ![MEHGT 回测结果与指标对比示意图](figs/回测.png)


---


# ⚠️ Notes

1. **Data Compliance & Privacy** 
   - Ensure all data sources are legal and compliant, especially announcements or institution data.    
   - Avoid uploading sensitive raw data; apply anonymization and aggregation.

2. **Resource Requirements** 
   - FinEX training requires high-end GPUs (A100/H100); adjust batch size & sequence length accordingly.   
   - MEHGT training is resource-intensive; plan sampling & batching strategies.

3. **Version Compatibility**  
   - PyTorch / CUDA / transformers versions may vary—follow `requirements.txt`.   
   - For multi-node clusters, ensure consistency across NCCL, accelerate, deepspeed, etc.

4. **Reproducibility**
   - Set random seeds and record hyperparameters & data splits.

---

# 🔮 Future Outlook

1. **Integration with Financial Service Platforms**  
The solution will be deeply integrated with banking, wealth management, and insurance platforms, providing reusable intelligent financial capabilities. Users will obtain more reliable market insights, risk alerts, and asset allocation suggestions.


2. **Expansion to More Financial Scenarios** 
Leveraging the MEHGT-LKG heterogeneous graph model, the system will expand to derivatives, smart risk control, personalized investment via user profiling, and cross-institution generalization.


3. **Domestic Financial Technology Independence**  
In collaboration with Ascend AI Computing Center (China), the team explores high-performance training & inference based on MindSpore, RAG, CANN, heterogeneous operators, and GNN-related innovations to achieve ecosystem independence and drive local fintech development.

---

# 📚 Appendix: Links & Citation Suggestions

- LLaMA Factory（FinEX finetuning framework reference）：  
  <https://github.com/hiyouga/LLaMA-Factory>

- FinEX model card & weights:
  <https://www.modelscope.cn/models/Madness977/FinEX>


