# Sentence level Recurrent Neural Network  for Language Modeling.

Code for sentence level interpretation of recurrent neural network. The model is trained discriminatively using the objective function similar to one used in pairwise ranking. The code is implemented in Theano.

## Dependencies:
* Python 2.7.
* Theano 0.8:
> 	This version of theano is not available of the shelf and needs to be 	downloaded from the website. The instructions are [here]	(http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions).
* Numpy
* Six (If using python 2.7 which we are using)

## Training
The file to run for both training and testing is `code/main.py`.

Simplest training example with default distortion for default training file can be simply run as

```python main.py --mode 0```

This generates `best_model.pkl` which is serialized model file and is read by default while evaluation.

## Testing
Evaluation is done using Contrastive Entropy. Simplest test example for default test set and for default test distortion level can be run as

```python main.py --mode 1```

## Other Configuration

 * --mode MODE      Run mode: 0 => Train, 1 => Test, 2=>BOTH
 *  --train TRAIN    training text file
 *  --valid VALID    validation text file
 *  --test TEST      test text file
 *  --n N            dimension of embedding space
 *  --epochs EPOCHS  dimension of embedding space
 *  --l L            learning rate
 *  --trnd TRND      training distortion level
 *  --tstd TSTD      test distortion level
 *  --l2reg L2REG    lambda l2 reg value
 *  --pct PCT        Percentage of training data to use

 To train with default train set for 50 epochs, with l2 regularization coefficient 0.1 and learning rate 0.1 the command will be

``` python main.py --mode 0 --epochs 50 --l2reg 0.1 --l 0.1```

For testing with test distortion percentage 30% the command would be

``` python main.py --mode 1 --tstd 30```
