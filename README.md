# latentMapping

To build this repo, please use **python3.6.15** in a virtual environment: 

```
virtualenv venv --python=<PATHTOPYTHON3.6>
source venv/bin/activate
pip install -r requirements.txt
```

# Misc

This is bad practice but in order for figures to save correctly, please modify the file `/lib/python3.8/site-packages/scanpy/plotting/_tools/scatterplots.py:883`. Remove the "show" and replace it with empty string: 

```
_utils.savefig_or_show('', show=show, save=save)
```
## Running the code for 3Step_Latent_Mapping Branch:

### Please run [latentMapping.py](latentMapping.py)
## Description of key functionality of files:

### [scvi_three_step_training.py](scvi_three_step_training.py)
implements the 3 stage training protocol using the scvi vae

### [test_discriminator.py](test_discriminator.py)
testing script for the discriminator model

### [test_step_1.py](test_step_1.py)
trains scvi VAE model, for step 1

### [v1_three_step_training.py](v1_three_step_training.py)
first implementation of 3 step training, using scvi vae and sedr vgae

### [v2_three_step_training.py](v2_three_step_training.py)
In progress, implementing custom vae, using sedr vgae

### [test_sedr.py](test_sedr.py)
tests the sedr framework, with customizations
