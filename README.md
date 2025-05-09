# AM220 Final Project Code Repository

## Environment Configuration

To configure and activate the conda environment for this repository, run

```bash
conda env create -f environment.yml 
conda activate borf 
pip install -r requirements.txt
```

## Experiments

To run our node classification experiments with the AFR3 method, run the file `run_node_classification.py` as follows:

<pre> python run_node_classification.py --rewiring AFR3 </pre>

To run other structural rewiring methods, simply replace AFR3 with DR, borf, sdrf, or fosr in the rewiring argumnet.

To run feature shuffle rewiring, set the shuffle argument to true and specify the shuffle ratio as follows:

<pre> python run_node_classification.py --rewiring AFR3 --shuffle True --ratio 0.75 </pre>

To change the GNN architecture, specify the layer_type arguments as follows:

<pre> python run_node_classification.py --rewiring AFR3 --layer_type GCN </pre>

Other aspects of the experiments can be modified by specifying other arguments in the command line. See the `run_node_classification.py` file for a list of these arguments.

Note that much of this code is duplicated from Fesser and Weber et al. 2024: https://arxiv.org/abs/2309.09384
