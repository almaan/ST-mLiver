library(irlba)

#' Perform Canonical Correlation Analysis with more than two groups
#'
#' Runs a canonical correlation analysis
#'
#' @param object.list List of Seurat objects
#' @param genes.use Genes to use in mCCA.
#' @param add.cell.ids Vector of strings to pass to \code{\link{RenameCells}} to
#' give unique cell names
#' @param niter Number of iterations to perform. Set by default to 25.
#' @param num.ccs Number of canonical vectors to calculate
#' @param standardize standardize scale.data matrices to be centered (mean zero)
#' and scaled to have a standard deviation of 1.
#' @param renormalize Should the data be normalized with SCTransform after merging? [default : TRUE]
#' @param verbose Print messages
#'
#' @return Returns a combined Seurat object with the CCA stored in the @@dr$cca slot.
#'
#' @importFrom methods slot
#' @importFrom irlba irlba
#'
#'
#' @examples
#' pbmc_small
#' # As multi-set CCA requires more than two datasets, we will split our test object into
#' # three just for this example
#' pbmc1 <- SubsetData(pbmc_small,cells.use = pbmc_small@cell.names[1:30])
#' pbmc2 <- SubsetData(pbmc_small,cells.use = pbmc_small@cell.names[31:60])
#' pbmc3 <- SubsetData(pbmc_small,cells.use = pbmc_small@cell.names[61:80])
#' pbmc1@meta.data$group <- "group1"
#' pbmc2@meta.data$group <- "group2"
#' pbmc3@meta.data$group <- "group3"
#' pbmc.list <- list(pbmc1, pbmc2, pbmc3)
#' pbmc_cca <- RunMultiCCA(object.list = pbmc.list, genes.use = pbmc_small@var.genes, num.ccs = 3)
#' # Print results
#' PrintDim(pbmc_cca,reduction.type = 'cca')

RunMultiCCA <- function (
  object.list,
  genes.use = NULL,
  add.cell.ids = NULL,
  niter = 25,
  num.ccs = 1,
  standardize = TRUE,
  renormalize = TRUE,
  verbose = FALSE
) {

  set.seed(42)
  if(length(object.list) < 2){
    stop("Must give at least 2 objects/matrices for MultiCCA")
  }

  # Check that the sample column is present in each dataset
  check <- all(unlist(lapply(object.list, function(obj) {
    st.obj <- GetStaffli(obj)
    "sample" %in% colnames(st.obj[[]])
  })))
  if (!check) stop("A sample column must be present in all objects of object.list")

  if (verbose) print("Collecting variable fatures from Seurat objects ...")
  mat.list <- list()
  if(class(object.list[[1]]) == "Seurat"){
    if (is.null(genes.use)) {
      genes.use <-  Reduce(intersect, lapply(object.list, function(obj) {VariableFeatures(obj)}))
      if (length(x = genes.use) == 0) {
        stop("No variable genes present. ")
      }
    }
    for(obj in object.list) {
      genes.use <- CheckGenes(data.use = GetAssayData(object = obj, slot = "scale.data"), genes.use = genes.use)
    }
    for(i in 1:length(object.list)){
      mat.list[[i]] <- GetAssayData(object = object.list[[i]], slot = "scale.data")[genes.use, ]
    }
  } else{
    stop("input data not Seurat objects")
  }

  if (verbose) print("Checking that spot names are not duplicated ...")
  if (!is.null(add.cell.ids)) {
    if (length(add.cell.ids) != length(object.list)) {
      stop("add.cell.ids must have the same length as object.list")
    }
    object.list <- lapply(seq_along(object.list), function(i) {
      RenameCells(object = object.list[[i]], add.cell.id = add.cell.ids[i])
    })
  }
  # Check if there are duplicated spot ids
  names.list <- Reduce(c, lapply(object.list, colnames))
  if(sum(duplicated(names.list)) > 0) {
    stop("duplicate cell names detected, please set 'add.cell.ids'")
  }

  # Standardize matrices
  if (verbose) print("Scaling data ...")
  num.sets <- length(mat.list)
  if(standardize){
    mat.list <- lapply(mat.list, function(obj) {
      scale(obj, center = TRUE, scale = TRUE)
    })
  }

  # Compute partial SVD
  ws <- lapply(1:num.sets, function(i) {
    irlba(mat.list[[i]], nv = num.ccs)$v[, 1:num.ccs, drop = F]
  })
  ws.init <- ws
  ws.final <- lapply(1:length(ws), function(i) {
    matrix(0, nrow = ncol(mat.list[[i]]), ncol = num.ccs)
  })
  cors <- NULL

  if (verbose) print("Computing CCA vectors ...")
  for (cc in 1:num.ccs){
    if (verbose) print(paste0("Finished CC_", cc))
    #ws <- list()
    ws <- lapply(1:length(ws.init), function(i) {
      ws.init[[i]][, cc]
    })
    cur.iter <- 1
    crit.old <- -10
    crit <- -20
    storecrits <- NULL
    while(cur.iter <= niter && abs(crit.old - crit)/abs(crit.old) > 0.001 && crit.old !=0){
      crit.old <- crit
      crit <- GetCrit(mat.list, ws, num.sets)
      storecrits <- c(storecrits, crit)
      cur.iter <- cur.iter + 1
      for(i in 1:num.sets){
        ws[[i]] <- UpdateW(mat.list, i, num.sets, ws, ws.final)
      }
    }
    for(i in 1:length(ws)){
      ws.final[[i]][, cc] <- ws[[i]]
    }
    cors <- c(cors, GetCors(mat.list, ws, num.sets))
  }

  results <- list(ws = ws.final, ws.init = ws.init, num.sets = num.sets, cors = cors)

  object.list.filtered <- lapply(seq_along(object.list), function(i) {
    obj <- object.list[[i]]
    vst <- obj[["SCT"]]@misc$vst.out
    #rownames(vst$cell_attr) <- paste(add.cell.ids[i], rownames(vst$cell_attr), sep = "_")
    vst$cell_attr <- vst$cell_attr[Cells(obj), ]
    vst$cells_step1 <- paste(add.cell.ids[i], vst$cells_step1, sep = "_")
    vst$cells_step1 <- intersect(vst$cells_step1, Cells(obj))
    #colnames(vst$umi_corrected) <- paste(add.cell.ids, colnames(vst$umi_corrected), sep = "_")
    obj[["SCT"]]@misc$vst.out <- vst
    return(obj)
  })

  combined.object <- object.list.filtered[[1]]
  # Merge objects
  if (verbose) print("Merging Seurat objects ...")
  for(i in 2:length(object.list)){
    combined.object <- MergeSTData(x = combined.object, y = object.list.filtered[[i]], merge.data = TRUE)
  }

  if (verbose) print("Saving CCA results to merged Seurat object ...")
  combined.object@assays[[DefaultAssay(combined.object)]]@var.features <- genes.use
  cca.data <- results$ws[[1]]
  for(i in 2:length(object.list)){
    cca.data <- rbind(cca.data, results$ws[[i]])
  }
  rownames(cca.data) <- colnames(combined.object)
  cca.data <- apply(cca.data, MARGIN = 2, function(x){
    if(sign(x[1]) == -1) {
      x <- x * -1
    }
    return(x)
  })
  colnames(cca.data) <- paste0("CC", 1:ncol(cca.data))

  # Rerun SCTransform
  # Collect regression vars
  if (renormalize) {
    if (verbose) print("Rerunning SCTransform with merged data ...")
    tests <- lapply(object.list, function(obj) {
      vars.to.regress = obj@commands[["SCTransform.RNA"]]$vars.to.regress
    })
    identicalValue <- function(x, y) {
      if (identical(x, y)) {
        x
      } else {
        warning("Different variables were used for regression in Seurat objects and normalization will therefore be skipped. It is highly recommended to normalize the returned data.", call. = FALSE)
        NULL
      }
    }
    vars.to.regress <- Reduce(identicalValue, tests)

    if (!is.null(vars.to.regress)) {
      combined.object <- SCTransform(combined.object, vars.to.regress = vars.to.regress)
    } else {
      combined.object <- SCTransform(combined.object)
    }
  }

  combined.object[["cca"]] <- CreateDimReducObject(
    embeddings = cca.data,
    key = "CC_",
    assay = DefaultAssay(combined.object)
  )

  if (verbose) print("Projecting CCA vectors to genes ...")
  combined.object <- ProjectDim (
    object = combined.object,
    reduction = "cca",
    verbose = FALSE,
    overwrite = TRUE)

  return(combined.object)
}

#' MultiCCA helper function - calculates correlation
#'
#' Modified from PMA package
#' @references Witten, Tibshirani, and Hastie, Biostatistics 2009
#' @references \url{https://github.com/cran/PMA/blob/master/R/MultiCCA.R}
#'
#' @param mat.list list of matrices to calculate correlation
#' @param ws vector of projection vectors
#' @param num.sets number of datasets
#'
#' @return total correlation
#'
GetCors <- function (mat.list, ws, num.sets)
{
  cors <- 0
  for (i in 2:num.sets) {
    for (j in 1:(i - 1)) {
      thiscor <- cor(mat.list[[i]] %*% ws[[i]], mat.list[[j]] %*%
                       ws[[j]])
      if (is.na(thiscor))
        thiscor <- 0
      cors <- cors + thiscor
    }
  }
  return(cors)
}

CheckGenes <- function (data.use, genes.use)
{
  genes.var <- apply(X = data.use[genes.use, ], MARGIN = 1, FUN = var)
  genes.use <- genes.use[genes.var > 0]
  genes.use <- genes.use[!is.na(x = genes.use)]
  return(genes.use)
}

#' MultiCCA helper function - calculates critical value (when to stop iterating
#' in the while loop)
#'
#' Modified from PMA package
#' @references Witten, Tibshirani, and Hastie, Biostatistics 2009
#' @references \url{https://github.com/cran/PMA/blob/master/R/MultiCCA.R}
#'
#' @param mat.list list of matrices
#' @param ws vector of projection vectors
#' @param num.sets number of datasets
#'
#' @return returns updated critical value
#'
GetCrit <- function (mat.list, ws, num.sets)
{
  crit <- 0
  for (i in 2:num.sets) {
    for (j in 1:(i - 1)) {
      crit <- crit + t(ws[[i]]) %*% t(mat.list[[i]]) %*%
        mat.list[[j]] %*% ws[[j]]
    }
  }
  return(crit)
}


#' MultiCCA helper function - updates W
#'
#' Modified from PMA package
#' @references Witten, Tibshirani, and Hastie, Biostatistics 2009
#' @references \url{https://github.com/cran/PMA/blob/master/R/MultiCCA.R}
#'
#' @param mat.list list of matrices
#' @param i index of current matrix
#' @param num.sets number of datasets
#' @param ws initial vector of projection vectors
#' @param ws.final final vector of projection vectors
#'
#' @return returns updated w value
#'
UpdateW <- function(mat.list, i, num.sets, ws, ws.final){
  tots <- 0
  for(j in (1:num.sets)[-i]){
    diagmat <- (t(ws.final[[i]])%*%t(mat.list[[i]]))%*%(mat.list[[j]]%*%ws.final[[j]])
    diagmat[row(diagmat)!=col(diagmat)] <- 0
    tots <- tots + t(mat.list[[i]])%*%(mat.list[[j]]%*%ws[[j]]) - ws.final[[i]]%*%(diagmat%*%(t(ws.final[[j]])%*%ws[[j]]))
  }
  w <- tots/l2n(tots)
  return(w)
}

#' Calculates the l2-norm of a vector
#'
#' Modified from PMA package
#' @references Witten, Tibshirani, and Hastie, Biostatistics 2009
#' @references \url{https://github.com/cran/PMA/blob/master/R/PMD.R}
#'
#' @param vec numeric vector
#'
#' @return returns the l2-norm.
#'
l2n <- function(vec){
  a <- sqrt(sum(vec^2))
  if(a==0){
    a <- .05
  }
  return(a)
}
