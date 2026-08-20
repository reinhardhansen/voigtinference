#!/usr/bin/env Rscript
# Deterministic acquisition of the red-ochre Raman spectrum used by
# examples/raman.jl. The data ship with the CRAN package `voigt` (GPL-2,
# Cannas & Piras; excerpt of the spectra of Pisu et al., Spectrochim. Acta A
# 329 (2025) 125581) and are NOT redistributed with this MIT-licensed
# release; this script fetches the pinned upstream version, VERIFIES its
# checksum, and converts it.
#
#   Rscript examples/get_raman.R
#
# writes examples/raman.csv (2048 rows; columns x = Raman shift in 1/cm,
# y = intensity) and prints the md5 checksums of the archive and the CSV.
#
# The version is pinned: if voigt 2.0 rotates off the main CRAN index it is
# fetched from the permanent CRAN Archive URL instead, and a checksum
# mismatch stops the script rather than silently using different data.
pkg_ver <- "2.0"
expected_md5 <- "3bbe285a37a27e781385b75f1ca0a4e2"   # voigt_2.0.tar.gz (CRAN)
# derived raman.csv has md5 424b8969b7dbb0b7b9be44475fc24133 (2048 rows)
urls <- c(
  sprintf("https://cran.r-project.org/src/contrib/voigt_%s.tar.gz", pkg_ver),
  sprintf("https://cran.r-project.org/src/contrib/Archive/voigt/voigt_%s.tar.gz",
          pkg_ver)
)
dest <- file.path(tempdir(), sprintf("voigt_%s.tar.gz", pkg_ver))
ok <- FALSE
for (u in urls) {
  ok <- tryCatch({
    download.file(u, dest, mode = "wb", quiet = TRUE)
    cat("downloaded:", u, "\n")
    TRUE
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (ok) break
}
if (!ok) stop("could not download voigt_", pkg_ver, ".tar.gz from CRAN")
got_md5 <- tools::md5sum(dest)[[1]]
cat("archive md5:", got_md5, "\n")
if (expected_md5 == "PENDING-PIN")
  stop("expected_md5 is not pinned; record the value above in this script")
if (got_md5 != expected_md5)
  stop("checksum mismatch: got ", got_md5, ", expected ", expected_md5,
       " -- upstream file changed; do not use it unverified")
untar(dest, files = "voigt/data/raman.rda", exdir = tempdir())
load(file.path(tempdir(), "voigt", "data", "raman.rda"))
out <- file.path(dirname(sub("--file=", "",
       grep("--file=", commandArgs(FALSE), value = TRUE))), "raman.csv")
if (length(out) == 0 || out == "/raman.csv") out <- "raman.csv"
write.csv(raman, out, row.names = FALSE)
cat("wrote:", out, "(", nrow(raman), "rows )\n")
cat("csv md5:", tools::md5sum(out)[[1]], "\n")
