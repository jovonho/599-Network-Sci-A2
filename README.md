# Network Science Assignment 2

## Overview
In this assignment we first explore various node centrality measures and apply them to the [Enron email dataset](https://www.cs.cornell.edu/~arb/data/pvc-email-Enron/) to extract the most important nodes. `data/email-Enron/addresses-email-Enron.txt` maps each node to an email address, and we can further map these to a person and title in the company using [1] and [2].

Following this, we apply various community detection algorithms to three different classes of datasets: the `real-classic` datasets comprising mainly social networks from various sources, `real-label` which are three citation networks taken from [3], and generated LFR benchmark graphs.

---
<br />

## Setup
- unzip data.zip into data/
  
- create a virtual environment and activate it 
  ```
  python -m venv .venv 
  .venv/Scripts/activate
  ```
- download the requirements 
    ```
    pip install -r requirements.txt
    ```
- run the code:
    ```
    python ./A2.py
    ``` 
---
<br />

## References

[1] Tsipenyuk G., Crowcroft J. (2017) An email attachment is worth a thousand words, or is it? IML 2017. https://doi.org/10.1145/3109761.3109765

[2] Creamer G., Rowe R., Hershkop S., Stolfo S.J. (2009) Segmentation and Automated Social Hierarchy Detection through Email Network Analysis. SNAKDD 2007. https://doi.org/10.1007/978-3-642-00528-2_3

[3] Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017) https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81