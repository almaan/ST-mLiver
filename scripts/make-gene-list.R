#!/usr/bin/Rscript

## small script to extract top 5000 genes to include in
## the stereoscope analysis. The workflow suggested in the
## Seurat tutorial "https://satijalab.org/seurat/v3.2/pbmc3k_tutorial.html"
## has been used. 

library(Seurat)

pth <- "../data/stereoscope/sc/itzkowitz/cnt/ScLiver_formatted_rownames.tsv.gz"

data <- read.table(pth,sep = '\t',row.names = 1,header = 1)

se <- CreateSeuratObject(t(data))

se <- NormalizeData(se)

se <- FindVariableFeatures(se,
                           selection.method = "vst",
                           nfeatures = 5000)

top5000 <- VariableFeatures(se)

write.table(top5000,
            file = "../data/gene-lists/stereoscope/top5000-genes-001.txt",
            sep = '\n',
            row.names =F,
            col.names = F,
            quote = FALSE
            )
