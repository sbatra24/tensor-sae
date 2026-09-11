# Tensor-SAE

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sbatra24/tensor-sae/blob/main/TensorSAE_reproduction.ipynb)

Code for our paper *Tensor-SAE: Structured Sparse Autoencoders for Interpretable and Efficient Image Representations*, published in the Proceedings of Machine Learning Research (PMLR) through the GRaM workshop at ICLR 2026.

## The idea

A sparse autoencoder explains an image as a short list of dictionary atoms: encode the image to a non-negative latent vector with only a few non-zero entries, then rebuild it as the weighted sum of the atoms those entries point at. On images the ordinary version has a problem. Every atom is a free 3×32×32 array, so a dictionary of a few thousand atoms costs millions of numbers, and when you look at the atoms they are mostly noise spread over the whole frame. That makes them expensive and hard to read.

Tensor-SAE keeps the encoder and the sparse latents but forces every atom to be a rank-one tensor: a colour vector times a row profile times a column profile, 3 + 32 + 32 numbers instead of 3072. An atom is then a patch with a single colour and a separable footprint, and the decoder is exactly linear in the latents. Adding α to one latent adds α times that patch to the image and nothing else, which is what you want from an editing handle. The paper compares this against a Dense-SAE and a convolutional autoencoder with the same parameter budget on CIFAR-10 and finds low-entropy spatial atoms, clean colour factors, linearly predictable interventions with R² of about 0.93, sparser latents, and better reconstruction per parameter, at the cost of some pixel-level fidelity.

## What is in the notebook

`TensorSAE_reproduction.ipynb` is one Colab notebook generated from `tensor_sae_colab.py` by `build_notebook.py`, so the `.py` file is the one to edit.

It builds the three models (Tensor-SAE with the factorised einsum decoder, Dense-SAE with a parameter-matched dictionary, ConvAE with a parameter-matched width), trains one triple per dictionary size on CIFAR-10 and tracks reconstruction MSE, PSNR, L0, dead-latent fraction and intervention strength after every epoch. It then produces every measurement the abstract names: reconstruction quality against parameters and against FLOPs for all models, the spatial entropy of the atoms with a histogram and the atom galleries, colour-factor cleanliness with a chromaticity plot, the intervention-linearity R² (measured directly at the decoder and after re-encoding the edited image) with predicted-against-actual scatter plots, the coefficient of variation of the intervention strength across training, L0 and activation histograms, and an editing demo that removes an image's strongest atom and adds a spatially localised one. Every table prints the paper's number next to the reproduced one where the abstract gives one, and says "not reported in abstract" otherwise.

All outputs land in `results/` as CSV, JSON and PNG, with a checkpoint for every trained model, so a Colab restart resumes from what is finished.

## How to run

The first code cell has a `RUN_MODE` switch.

`smoke` runs in about three minutes on a two-core CPU. It downloads nothing: the images are synthetic 32×32 scenes of coloured rectangles and blobs laid out on a 4×4 grid, the dictionaries have 32, 64 and 128 atoms, and the ConvAE gets 6 epochs to the SAEs' 40 because convolutions dominate CPU time. Its numbers test the code paths, not the paper's claims.

`full` downloads CIFAR-10 through torchvision, trains Tensor-SAEs with 1024, 2048 and 4096 atoms with their matched Dense-SAEs and ConvAEs for 30 epochs each, and runs every analysis on the 4096-atom triple. I estimate 30 to 60 minutes on a free Colab T4; the SAEs are cheap and the ConvAEs take most of it. Every count is a field of `Config`.

To run locally:

- `pip install -r requirements.txt`
- `TSAE_RUN_MODE=smoke python tensor_sae_colab.py`
- `TSAE_RUN_MODE=full python tensor_sae_colab.py`
- `python build_notebook.py`

Set `SAVE_TO_DRIVE = True` in the first cell to mirror `results/` to Google Drive at the end of a Colab run.

## What differs from the paper

The original experiment code was lost. This is a clean reimplementation, and it follows the paper's abstract and figures as described there: a sparse autoencoder that decodes through a bank of rank-one colour × height × width atoms, with the decoder factorised into colour and spatial factors and a light sparsity prior on the latents, compared on CIFAR-10 against a parameter-matched Dense-SAE and a ConvAE with an equivalent parameter budget. It was written against the abstract rather than line by line against the final manuscript, so the exact hyper-parameters, dictionary sizes and training schedule of the paper are not guaranteed to match; they are set to sensible values and are all exposed in `Config`: dictionaries of 1024 to 4096 atoms, Adam at 1e-3, batch 256, 30 epochs, L1 weight 0.3 on a per-image loss `‖x − x̂‖² + λ‖z‖₁`, no learning-rate schedule and no dead-latent resampling. The numbers this notebook produces are the numbers of this code.

The encoder is a free linear map for both SAEs; only the decoder is factorised, because that is what the abstract describes. A consequence is that the encoder holds 3072 parameters per atom in both models, so the parameter-matched Dense-SAE has about half as many atoms as the Tensor-SAE rather than a 46th; the decoder-only ratio is 46 and both counts are printed. Every atom is normalised to unit norm at each forward pass so that latent magnitudes are comparable across atoms.

The ConvAE is a plain autoencoder with a ReLU bottleneck of `d` channels at 8×8 and no sparsity penalty. Its architecture and the way the paper intervened on it are not in the abstract; here an intervention adds to one channel at the central bottleneck position.

The abstract does not say how the intervention R² was measured. The notebook reports two protocols. The direct one regresses the clamped decoded pixel change on α times the unit response of the latent, so for the SAEs the only non-linearity is the pixel clamp. The round-trip one re-encodes the edited image and decodes it again, which is the property a controllable edit needs and the one on which the dictionaries differ. Interventions are scaled by each latent's typical activation, so α is in natural units. The paper's 0.93 is printed next to both.

Intervention strength per unit latent change is at most 1 for unit-norm atoms by construction, so the notebook also tracks the strength per typical activation, which follows the scale the model gives its latents during training. Which of these the paper tracked is not stated.

FLOPs are analytic multiply-adds of the forward pass as implemented. A rank-one atom still has to be written to 3072 pixels, so per active atom the factorised decoder costs the same arithmetic as a dense one; the saving is in parameters. The table also gives the cost of a decoder that only touches the active atoms. Which accounting the paper used for its per-FLOP claim is not in the abstract.

Spatial entropy is the Shannon entropy of the atom's spatial energy map `Σ_c |a_k[c, h, w]|` normalised to sum to one. Colour cleanliness is the angle to the nearest of the red, green, blue and grey axes; for Dense-SAE atoms the colour vector is the leading singular vector of the atom reshaped to 3×1024. These are my definitions of the abstract's "low-entropy spatial atoms" and "clean colour factors".

I have run the smoke path end to end. The full path is written for a Colab T4 and was not executed on the machine where this repository was assembled.

## Citation

Tanush Shastry, Soham Batra, Laksh Patel, Aarav Lala, Andrew Bae, Siddharth Karuturi, Mithil Shah, Neel Shanbhag. Tensor-SAE: Structured Sparse Autoencoders for Interpretable and Efficient Image Representations. GRaM Workshop, ICLR 2026, published in PMLR. https://openreview.net/forum?id=MmpRG8AuHY

## License

MIT, see `LICENSE`.
