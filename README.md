# ST Liver 

This repository provides data and code to reproduce data presented in "Spatial Transcriptomics to define transcriptional patterns of zonation and structural components in the liver", and show-cases the applicability and potential to study tissue transcriptomics computationally. In this study, Spatial Transcriptomics was performed on liver tissue sections of female wildtype mice of 8-12 weeks of age. All scripts presented here are designed to be used with any kind of spatial transcriptomics data and provide detailed documentation for the reproduction of the data presented in this original study. Original data and large files are placed at an external site save resources inlcuding count matrices, spot files, HE-images, masks, h5ad-files, etc. and can be accessed at [insert link to zenodo repo]. 

## Structure

* `data` - contains processed data to be used in the analysis (files must be downloaded from an external resource, see below)
	* `gene lists`
    * `marker genes` - contains csv files with central and portal markers of the ST data and the SC data analyis performed for the liver data of the [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8)
    * `stereoscope` - contains lists of 5000 most variable genes of single cell [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8) data and [Halpern et al](https://www.nature.com/articles/nature21065) study for sterescope analysis
    * `veins` - contains shortlists of marker genes for portal and central veins for expression by distance analysis
	* `meta`- 
	* `stereoscope`- 
* `scripts` - contains processing scripts and notebooks
	* `Liver-ST.Rmd` - contains a R markdown script to perform canonical correlation analyis, clustering and DGEA, tissue visualization, correlation analysis, visualization of single cell integration using single cell data of the [Mouse Cell Atlas](https://www.cell.com/cell/fulltext/S0092-8674%2818%2930116-8) and comparative analyses with published data from [Halpern et al](https://www.nature.com/articles/nature21065)
	* `MultiCCA.R` - contains code for the modified canonical correlation analysis function used in `Liver-ST.Rmd`
    * `cluster-interaction-analysis.ipynb` - notebook outlining the cluster interaction analysis (used to produce Supplementary Image 2)
    * `make-gene-list.R` - script to generate list of highly variable genes (hvgs) to use in `stereoscope` mapping.
    * `prepare-data.py` - program with CLI to generate `h5ad` files for spatial analysis (in `vein-analysis.ipynb`). See
    * `vein-analysis.ipynb` - notebook outlining the feature by distance analysis and vein type classification/prediction based on NEPs.
    
* `hepaquery` - files constituting the `hepaquery` package
* `setup.py` - installation file for `hepaquery` module

## hepaquery Installation
To facilitate reproduction of our results and enable easy exploration of similar
data sets using the methods we present in this work, we have packaged these into
a Python module called `hepaquery`. This, among other things, contains functions
to generate _feature\_by\_distance_ plots, classification of vein types based on
neighborhood expression profiles (NEPs), and evaluation of prediction results.
To install this package, do:

1. Enter a terminal window, change directory to this repository and type

```sh
$> python3 setup.py install

```

Depending on your OS and user configurations, you might have to add `--user` for this to work.

2. Next, to test if the installation work, enter the following into the terminal:

```sh
$> python3 -c "import hepaquery; print(hepaquery.__version__)"

```
If everything went as expected, this should print the version of `hepaquery` in your terminal.

The notebook `scripts/vein-analysis.ipynb` illustrates the usage of `hepaquery`.

## Prepare data for hepaquery analysis

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

## Accessing Data

While GitHub supports storage of large files via the LFS system, we have placed our files at an external site to prevent unnecessary use of resources. The count matrices, spot files, HE-images and masks can be accessed at [this](link) link. To download all data and place it in the expected (by the scripts) location, you can also go to `scripts`, open a terminal and do:

```sh
$> chmod +x ./fetch-data.sh
$> ./fetch-data.sh
```
