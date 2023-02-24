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