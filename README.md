# ST-mLiver 

This repository provides data and code to reproduce data presented in "_Spatial
Transcriptomics to define transcriptional patterns of zonation and structural
components in the liver_", and show-cases the applicability and potential to
study tissue transcriptomics computationally. In this study, Spatial
Transcriptomics was performed on liver tissue sections of female wildtype mice
of 8-12 weeks of age. All scripts presented here are designed to be used with
any kind of spatial transcriptomics data and provide detailed documentation for
the reproduction of the data presented in this original study. Original data and
large files are placed at an external site save resources inlcuding count
matrices, spot files, HE-images, masks, h5ad-files, etc. and can be accessed at
[zenodo](10.5281/zenodo.4399655).

* [Overview](#overview)
* [System Requirements](#system-requirents)
* [hepaquery](#hepaquery)
    * [Dependencies](#dependencies)
    * [Installation](#installation)
    * [Data Preparation](#data-preparation)
* [Data Access](#data-access)
* [License](#license)

<hr>

## Overview

Below is an overview of the structure of this repository, including brief descriptions of respective item.

* `data` - contains processed data to be used in the analysis (files must be downloaded from an external resource, see below)
	* `gene lists`
        * `marker genes` - contains csv files with central and portal markers of the ST data and the SC data analyis performed for the liver data of the [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8)
        * `stereoscope` - contains lists of 5000 most variable genes of single cell [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8) data and [Halpern et al](https://www.nature.com/articles/nature21065) study for sterescope analysis
        * `veins` - contains shortlists of marker genes for portal and central veins for expression by distance analysis
    * `sterescope/sc` - single cell data used for the `stereoscope` analysis
* `scripts` - contains processing scripts and notebooks
	* `Liver-ST.Rmd` - contains a R markdown script to perform canonical correlation analyis, clustering and DGEA, tissue visualization, correlation analysis, visualization of single cell integration using single cell data of the [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8) and comparative analyses with published data from [Halpern et al](https://www.nature.com/articles/nature21065)
	* `MultiCCA.R` - contains code for the modified canonical correlation analysis function used in `Liver-ST.Rmd`
    * `cluster-interaction-analysis.ipynb` - notebook outlining the cluster interaction analysis (used to produce Supplementary Image 2)
    * `make-gene-list.R` - script to generate list of highly variable genes (hvgs) to use in `stereoscope` mapping.
    * `prepare-data.py` - program with CLI to generate `h5ad` files for spatial analysis (in `vein-analysis.ipynb`). See
    * `vein-analysis.ipynb` - notebook outlining the feature by distance analysis and vein type classification/prediction based on NEPs.
    * `proportion-analysis.ipynb` - similar to first part of `vein-analysis.ipynb` but looking at the cell type proportions  -compared to expression levels - (obtained from `stereoscope`) as a function of distance to the nearest vein.
* `res/sterescope-res/`
    * `CNX_ZY` - folder for each sample (X,Y, and Z are various parts of identifiers). Each folder contains a `W*.tsv` file which are the `sterescope` results, formatted as `[n_spots]x[n_types]` matrices.
    * `st_loss.txt` - loss output for st-model
    * `sc_loss.txt` - loss output for sc-model
    * `st_model.pt.gz` - gzipped fitted st-model
    * `sc_model.pt.gz` - gzipped fitted sc-model
    * `R.tsv.gz` - gzipped estimated R paramters (in `stereoscope` model)
    * `logits.tsv.gz` - gzipped esitmated logits parameters (in `stereoscope` model)

* `hepaquery` - files constituting the `hepaquery` package
* `setup.py` - installation file for `hepaquery` module

## System Requirements
The code should be compatible with all systems that support `R` and `python`. We recommend the following software versions:
- `R >= 3.5`
- `python >= 3.7`

To reproduce the analysis presented in the `jupyter-notebooks` and `R markdown` files, the packages listed below must be installed

- `jupyter-notebook` | [LINK](https://jupyter.org/install) | Python
- `rmarkdown` | [LINK](https://rmarkdown.rstudio.com/docs/) | R


## hepaquery
To facilitate reproduction of our results and enable easy exploration of similar
data sets using the methods we present in this work, we have packaged these into
a Python module called `hepaquery`. This, among other things, contains functions
to generate _feature\_by\_distance_ plots, classification of vein types based on
neighborhood expression profiles (NEPs), and evaluation of prediction results.

### Dependencies

The `hepaquery` package mainly relies on the python scientific framework, and
will be installed automatically when running the commands in the next section.
However, to list the packages and their recommended versions:

```
scikit-misc
numpy>=1.19.0
pandas>=1.0.0
anndata>=0.7.5
scipy>=1.5.4
scikit-learn>=0.23.2
matplotlib>=3.3.3
```

### Installation
To install this package, do:

1. Enter a terminal window, change directory to this repository and type

```sh
$> python3 setup.py install

```

Depending on your OS and user configurations, you might have to add `--user` for
this to work. The installation takes only a few seconds on a standard laptop
computer.

2. Next, to test if the installation work, enter the following into the terminal:

```sh
$> python3 -c "import hepaquery; print(hepaquery.__version__)"

```
If everything went as expected, this should print the version of `hepaquery` in your terminal.

The notebook `scripts/vein-analysis.ipynb` illustrates the usage of `hepaquery`.
Here the examples of the expected output when running `hepaquery` is found, and
instructions regarding how to reproduce the plots presented in the manuscript.
The analysis on the data in this study takes less than 5 minutes when run on a
laptop computer.

### Data Preparation

To conduct the feature by distance analysis, the data must first be curated and
formatted. For this purpose we provide the `scripts/prepare-data.py` script,
which offers a convenient CLI to easily produce the required `h5ad` files from
the raw count matrices, spot-files and HE-images. One additional _key_ element
are the *masks* indicating veins in the tissue. These masks should align with
the HE-images (same dimensions), and have each veins colored by their class.
YAML files are then used to specify which elements to include in the assembly of
the `h5ad` object.

For each item associated with a unique sample, create a YAML file
(`filename.yaml`) with the following structure:

```yaml

count_data: PATH_TO_COUNT_FILE
spot_data: PATH_TO_SPOT_FILE
image: PATH_TO_HE-IMAGE_FILE
mask: PATH_TO_MASK_FILE
rgb:
  class1: [R1,G1,B1]
  class2: [R2,G2,B2]
  class3: [R3,G3,B3]
```

where `class1` indicate the name of the first class (e.g., central) and `[R1,G1,B1]` gives the RGB value that indicate class1 in the mask. This is what we refer to as a *configuration file*

Next, simply go to the `scripts` folder and run in a terminal run:

```sh
$> python3 ./prepare-data.py -i CONFIG_FILE.yaml -n N_TYPES -s -d -o OUT_DIR
```
And `h5ad` files containing the essential information used in the
`vein-analysis.ipynb` files will be produced. For more information regarding the
parameters that may be used, simply do `python3 ./prepare-data.py -h`.

## Data Access

While GitHub supports storage of large files via the LFS system, we have placed
our files at an external site to prevent unnecessary use of resources. The count
matrices, spot files, HE-images and masks can be accessed at [this](10.5281/zenodo.4399655) link.
To download all data and place it in the expected (by the scripts) location, you
can also go to `scripts`, open a terminal and do:

```sh
$> chmod +x ./fetch-data.sh
$> ./fetch-data.sh ZENDO_LINK
```

In case the above does not work for you, simply go to the link referenced above,
download the files and place the content of `Hepaquery_data.zip` in the `data`
folder. This should be equivalent to what the script is doing for you.

## License
This work is covered under the **MIT License**.
