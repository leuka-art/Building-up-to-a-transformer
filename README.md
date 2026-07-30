# Building a Transformer from Scratch in NumPy

This project implements the foundations of modern deep learning from scratch using NumPy, including an automatic differentiation engine, neural network layers, optimisation algorithms, attention mechanisms and a transformer.

This repository documents my progression from classical machine learning to deep learning by implementing models from scratch in NumPy, starting with linear regression and building up to a transformer language model, complete with an automatic differentiation engine written from first principles.

## Contents

- [Linear Regression](#linear-regression)
- [Logistic Regression](#logistic-regression)
- [Autograd](#autograd)
- [Multi Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- [Optimisers](#optimisers)
- [Transformer](#transformer)

---

## Linear Regression

Implemented linear regression as a foundation for exploring how linear transformations are used in neural networks, with an emphasis on the analytical view.

Implemented:
- Least squares regression via closed-form solution and gradient descent
- Ridge regression closed-form solution
- Cross-validation for selecting optimal ridge lambda and comparing RMSE across methods
- Bootstrap resampling to study how ridge regression affects parameter variance and generalisation

Dataset: Kaggle abalone dataset, predicting number of rings (a proxy for age). Features standardised via z-score normalisation.

I learned: 
- The geometric intuition behind the closed-form solution
- How ridge regression reduces parameter variance
- How multicollinearity causes ill-conditioning and unstable parameters
- How cross-validation helps with limited data.

## Logistic Regression

Implemented multiclass logistic regression from first principles, building on the linear model with softmax and gradient descent (no closed-form solution here).

Implemented:
- Multiclass logistic regression trained via gradient descent, fully vectorised
- Numerically stable softmax and cross-entropy loss
- Training loss visualisation and accuracy evaluation

Dataset: Kaggle Phone Price Range dataset (four price classes), z-score normalised.

I learned: 
- The probabilistic interpretation of logistic regression
- The derivation of the cross-entropy gradient
- Why numerical stability matters in softmax.

## Autograd

Implemented a scalar autograd engine (micrograd, following Karpathy's approach) to understand the mechanics before extending it to a tensor-based engine used throughout the rest of the project.

I learned: 
- How forward/backward passes combine with the chain rule
- How to represent computation as a graph and topologically sort it via DFS
- How to propagate gradients correctly through broadcasting.

## Multi Layer Perceptron (MLP)

Built an MLP architecture from scratch, using the autograd engine for backpropagation.

Implemented:
- `Linear` and `ReLU` modules
- A `Sequential` container composing them into an MLP
- Full-batch and mini-batch gradient descent training

Trained on the same phone price dataset as the logistic regression model.

I learned: 
- Why ReLU is preferred over tanh/sigmoid (vanishing gradients)
- How non-linearities make the loss landscape non-convex
- The importance of Xavier initialisation
- Why SGD helps in non-convex optimisation.

## Optimisers

Implemented and compared a range of gradient-based optimisers on an image classification task (Fashion-MNIST).

Implemented: 
- Momentum
- Adagrad
- RMSProp
- Adadelta
- Adam

Adam achieved the best validation accuracy (89.0%) and lowest validation loss, followed closely by RMSProp and Adadelta; SGD converged slowest and generalised worst (81.3% accuracy).

![Optimiser convergence comparison on Fashion-MNIST](./Experimental_results/Optimiser_comparison.png)

I learned: 
- The importance behind adaptive per-parameter learning rates (Adagrad/RMSProp/Adadelta)
- Why momentum reduces oscillations
- How Adam combines both.

## Transformer

Implemented multi-head self attention and a character-level transformer language model from scratch, trained on Tiny Shakespeare, using the same autograd engine as the rest of the project.

Implemented:
- Single-head and multi-head self attention with causal masking
- Positional + token embeddings, layer normalisation and residual connections
- A full transformer block (attention, layer norm and FFN) and a stacked transformer model
- A transformer-specific training loop that samples context windows while preserving token order

### Ablation study

To understand the importance of each architectural component, I ran a controlled ablation study: removing positional embeddings, layer normalisation, and residual connections one at a time from an otherwise identical baseline (1 block, 4 heads, embedding dim 64, context length 64), and comparing training/validation loss and generated text.

| Model                     | Train Loss | Val Loss |
|---------------------------|:----------:|:--------:|
| Standard Transformer      | 2.223      | 2.313    |
| No positional embeddings  | 2.327      | 2.439    |
| No layer normalisation    | 2.331      | 2.496    |
| No residual connections   | 4.710      | 4.700    |

![Training loss for the baseline transformer vs each ablated model](/Experimental_results/Transformer_study.png)

Removing residual connections yielded the worst performance, the model failed to learn, with loss similar to initialisation and generated text collapsing into noise. Positional embeddings and layer normalisation each caused a smaller but consistent drop in performance; without positional information, the model could only rely on local character patterns. Without layer norm, training was less stable and converged slower.

I learned: 
- Residual connections provide a direct path backward, reducing the affect vanishing gradients have on the training
- Layer normalisation primarily aids optimisation stability rather than model capacity
- Self-attention has no inherent notion of token order/position, positional embeddings are what supply it.

**Generated samples.** Each model was prompted with `"Juliet!"` and used to generate 200 characters. The degradation matches the loss table closely — the standard model produces recognisable word fragments and character names, while removing residual connections collapses generation into near-random symbols:

<table>
<tr><th>Standard Transformer</th><th>No positional embeddings</th></tr>
<tr><td>

```
Juliet! sthak te we be Limb. Ky My ORIN
be A Muruthew pair of's mpeines I amas t
tannil bag Epaim?

NRIIf knt, ce polfars, avik mina min?

QUEEEEREN ETZABETH: I leire t mone, Whar
ive a: thyour g-whalm, And sthak se wit
oreneartir f ale keie ar muce my nortest.
```

</td><td>

```
Juliet! aiO t ewnth s o, ut Y I be ta
acouf tofl 'Gusutha thingoumel, I
ckindorthathind od h Grd in ay, buss or
pu es, Ofo I it heilaterofoll iver iciof
by. Thour hin f ce mand'sor:

Le Ro Cls, geangme thesheancotougansagld
l, Agaspanghioury burath My ur y s t t;
Hol souar od ombend.
```

</td></tr>
<tr><th>No layer normalisation</th><th>No residual connections</th></tr>
<tr><td>

```
Juliet! Dervime Vomy USIVII:

A: La y, ieeourd y herd, ber hat orgug ar
tag fave--

Angulapengofor'sct sly, andith tlfaven
paiimingaitil- I' And; hind sty hers m!
Ae ilirt ba don'lang I:

Whagat bls hanspe ee, bem, virtyatanit,
Ans kerat k an cathicof pin avenativent
pald-menee veare.
```

</td><td>

```
Juliet! VPdgk?JK &pvAj?jIA?A&grcI-H?PmIF
IgYNQhCjYibIHYImhpxICqrTfP?hiEFINEXf&3I$Rm
x:Zj;nj'OIjH?3sp,jzj:I3HJrC iO?,cF;NNc,Id
;I!jhkhLN'ycpIOBI' II3L HPgjN3HN'B;3Z
?NijjCNAjrjjk,,ji3wZ33cpBh Zm3?LcEjHLnI3I
ZNzsIxEkivNm;yF??B mj-q, bjmmcIjimYc&fmJj
```

</td></tr>
</table>

The standard and no-positional-embeddings models both produce roughly English-shaped tokens with occasional real words and character-name-like capitalised fragments; without layer norm the output is noticeably less structured; without residual connections the model fails to learn any character-level structure at all, generating what is effectively noise.

A full write-up of the methodology, gradient-checking results for the autograd engine, and complete experimental results is available in [`Transformer_From_Scratch_Report.pdf`](./Transformer_From_Scratch_Report.pdf).
