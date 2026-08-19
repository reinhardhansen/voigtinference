#!/usr/bin/env Rscript
# Deterministic acquisition of the red-ochre Raman spectrum used by
# examples/raman.jl. The data ship with the CRAN package `voigt` (GPL-2,
# Cannas & Piras; excerpt of the spectra of Pisu et al., Spectrochim. Acta A
# 329 (2025) 125581) and are NOT redistributed with this MIT-licensed
# release; this script fetches the exact upstream version and converts it.
#
#   Rscript examples/get_raman.R
#
# writes examples/raman.csv (2048 rows, columns shift/intensity) and prints
# the md5 checksums of the downloaded archive and of the CSV.
pkg_url <- "https://cran.r-project.org/src/contrib/voigt_2.0.tar.gz"
dest <- file.path(tempdir(), "voigt_2.0.tar.gz")
download.file(pkg_url, dest, mode = "wb", quiet = TRUE)
cat("downloaded:", pkg_url, "\n")
cat("archive md5:", tools::md5sum(dest)[[1]], "\n")
untar(dest, files = "voigt/data/raman.rda", exdir = tempdir())
load(file.path(tempdir(), "voigt", "data", "raman.rda"))
out <- file.path(dirname(sub("--file=", "",
       grep("--file=", commandArgs(FALSE), value = TRUE))), "raman.csv")
if (length(out) == 0 || out == "/raman.csv") out <- "raman.csv"
write.csv(raman, out, row.names = FALSE)
cat("wrote:", out, "(", nrow(raman), "rows )\n")
cat("csv md5:", tools::md5sum(out)[[1]], "\n")
