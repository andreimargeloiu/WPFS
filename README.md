# Weight Predictor Networks with Feature Selection (WPFS)
[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-yellow)](https://arxiv.org/abs/2211.15616)
[![Video presentation](https://img.shields.io/badge/Youtube-Video%20presentation-red)](https://youtu.be/18mULGRf1N8) [![Poster](https://img.shields.io/badge/-Poster-yellow)](https://docs.google.com/presentation/d/1G9ElKvj7KEc1SuXRDJjwT_ypuHkEOfRxQ4aqIJ36eYw/edit?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/a-norcliffe/sonode/blob/master/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) 

Official code for the paper [**Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data**](https://arxiv.org/abs/2211.15616) accepted at [**AAAI Conference on Artificial Intelligence 2023**](https://aaai.org/Conferences/AAAI-23/)


by [Andrei Margeloiu](https://www.andrei.ai/),
[Nikola Simidjievski](https://simidjievskin.github.io/),
[Pietro Lio](https://www.cl.cam.ac.uk/~pl219/),
[Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/)

**TL;DR:** WPFS is a general framework for learning neural networks from high-dimensional and small-sample data by reducing the number of learnable parameters, and performing global feature selection. In addition to the predictor network, WPFS combines two small auxiliary networks: a weight predictor network that outputs the weight matrix of the first layer, and a feature-selection network that serves as an additional mechanism for regularisation.


![image](https://user-images.githubusercontent.com/18227298/215850753-e573c226-03b8-4191-aec7-3a87785b04d4.png)



![](https://github.com/andreimargeloiu/WPFS/blob/main/paper.gif)

**Paper abstract:** Tabular biomedical data is often high-dimensional but with a very small number of samples. Although recent work showed that well-regularised simple neural networks could outperform more sophisticated architectures on tabular data, they are still prone to overfitting on tiny datasets with many potentially irrelevant features. To combat these issues, we propose Weight Predictor Network with Feature Selection (WPFS) for learning neural networks from high-dimensional and small sample data by reducing the number of learnable parameters and simultaneously performing feature selection. In addition to the classification network, WPFS uses two small auxiliary networks that together output the weights of the first layer of the classification model. We evaluate on nine real-world biomedical datasets and demonstrate that WPFS outperforms other standard as well as more recent methods typically applied to tabular data. Furthermore, we investigate the proposed feature selection mechanism and show that it improves performance while providing useful insights into the learning task.


# Citation

For attribution in academic contexts, please cite this work as
```
@inproceedings{margeloiu2023weights,
  title={Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data},
  author={Margeloiu, Andrei and Simidjievski, Nikola and Lio, Pietro and Jamnik, Mateja},
  booktitle={37th AAAI Conference on Artificial Intelligence},
  year={2023}
}
```


# Code structure

- `src`
	- `main.py`: code for parsing arguments, and starting experiment
		- def parse_arguments - include all command-line arguments
		- def train - start training model
		- important command-line arguments
			- dataset
			- model
			- feature_extractor_dims - the size of the hidden layers in the dnn
			- max_steps - maximum training iterations
			- batchnorm, dropout_rate
			- lr, batch_size, patience_early_stopping
			- lr_scheduler - learning rate scheduler
	- `dataset.py`: loading the datasets
	- `models.py`: neural network architectures: WPFS, DietNetworks, FsNet and Concrete Autoencoders
	- `weights_predictor_network.py` - defines the Weight Predictor Networks (WPN)
	- `sparsity_network.py` - defines the Sparsity Network (SPN)
- `data` 
	- cll, lung, prostate, smk, toxicity



# Installation

**Requirement:** All project dependencies are included in `requirements.txt`. We assume you have **conda** installed.


**Installing WPFS**
```
conda create python=3.7.9 --name WPFS
conda activate WPFS
pip install -r requirements.txt
```
**Optional:** Change `BASE_DIR` from `/src/_config.py` to point to the project directory on your machine.


# Running an experiment

**Step 1:** Run the script `run_experiment.sh`

**Step 2:** Analyze the results in the notebook `analyze_experiments.ipynb`

**Adding a new dataset is straightforward:**. Search `your_custom_dataset` in the codebase and replace it with your dataset.
