# 导出 Seurat v4/v5 的原始 counts；由 rds_to_h5ad.py 调用。
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: Rscript export_seurat_counts.R input.rds out_dir assay")
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
obj <- readRDS(args[1])
if (!inherits(obj, "Seurat")) stop("RDS must contain a Seurat object")
assay_name <- args[3]
if (!assay_name %in% names(obj@assays)) stop("Requested assay is missing")
assay <- obj[[assay_name]]
if (inherits(assay, "Assay5")) {
  layers <- Layers(assay, search = "^counts($|\\.)")
  if (length(layers) == 0L) stop("No raw counts layer in requested assay")
  if (length(layers) > 1L || layers[1] != "counts") {
    assay <- JoinLayers(assay, layers = layers, new = "counts")
  }
  counts <- LayerData(assay, layer = "counts")
} else if (utils::packageVersion("SeuratObject") >= "5.0.0") {
  # 新版 SeuratObject 加载旧式 Assay 时同样禁止 slot 参数。
  counts <- GetAssayData(assay, layer = "counts")
} else {
  counts <- GetAssayData(assay, slot = "counts")
}
if (nrow(counts) == 0L || ncol(counts) == 0L) stop("Counts matrix is empty")
if (is.null(rownames(counts)) || is.null(colnames(counts))) stop("Missing gene/cell IDs")
if (anyDuplicated(rownames(counts)) || anyDuplicated(colnames(counts))) stop("Duplicate gene/cell IDs")
metadata <- obj[[]]
if (!all(colnames(counts) %in% rownames(metadata))) stop("Cells missing from metadata")
metadata <- metadata[colnames(counts), , drop = FALSE]
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
Matrix::writeMM(as(counts, "sparseMatrix"), file.path(args[2], "counts.mtx"))
writeLines(rownames(counts), file.path(args[2], "genes.txt"))
writeLines(colnames(counts), file.path(args[2], "cells.txt"))
write.csv(metadata, file.path(args[2], "metadata.csv"), na = "")
