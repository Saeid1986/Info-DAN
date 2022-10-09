####################################################################
###################        INFOMAX-BASED         ###################
###################   DEEP AUTOENCODER NETWORK   ###################
###################          (INFO-DAN)          ###################
####################################################################


# Developer: Saeid ESMAEILOGHLI
# March 14, 2022
# --------------------------
# E-mail: esmaeiloghli@gmail.com
# ORCID: 0000-0002-7786-657X
# Website: http://esmaeiloghli.mining.iut.ac.ir
# GoogleScholar: http://scholar.google.com/citations?user=EZKZcwQAAAAJ&hl
# ResearchGate: https://www.researchgate.net/profile/Saeid_Esmaeiloghli
# ACADEMIA: https://cnrs.academia.edu/SaeidEsmaeiloghli


# ----------------------------- Inputs -----------------------------

setwd("C:/Users/SAEID/Desktop/Info-DAN")

data <- read.table("C:/Users/SAEID/Desktop/Info-DAN/Input/Test data.txt",
                   header = TRUE,
                   sep = ""
                   )

# --------------------------- Main Script --------------------------

# A. Load packages

library(compositions)
library(ica)
library(ANN2)
library(GGally)


# B. Transform original raw data into ilr-transformed data

ggpairs(data)

x <- ilr(data,
         V = ilrBase(data)
         )

x <- as.data.frame(x)

ggpairs(x)

write.table(x,
            file = 'ILR data.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )


# C. Implement information maximazation (Infomax) processor

im <- icaimax(
  X = x,              # Input data matrix.
  nc = 8,             # Number of source signals to extract.
  center = TRUE,      # If TRUE, columns of X are mean-centered.
  maxit = 100,        # Maximum number of iterations to allow.
  tol = 1e-6,         # Convergence tolerance.
  Rmat = diag(nc),    # Initial estimate of nc × nc orthogonal
                      # rotation matrix.
  alg = "gradient",   # Algorithm to use: "newton" (Newton
                      # iteration), "gradient" (gradient descent).
  fun = "tanh",       # Nonlinear function: "tanh" (hyperbolic
                      # tangent), "log" (logistic), "ext" (extended
                      # Infomax).
  signs = rep(1, nc), # Vector of length nc such that signs[j] == 1
                      # if j-th source signal is super-Gaussian and
                      # signs[j] == -1 if j-th source signal is
                      # sub-Gaussian.
  signswitch = TRUE,  # If TRUE, signs vector is automatically
                      # determined from the data.
  rate = 1,           # Learing rate for gradient descent algorithm.
  rateanneal = NULL   # Annealing angle and proportion for gradient
                      # descent learing rate.
  )

write.table(im$W,
            file = 'Unmixing Matrix (W).txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = TRUE,
            col.names = TRUE
            )

write.table(im$S,
            file = 'Source Signals (S).txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )

write.table(im$vafs,
            file = 'Variances.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )

sources <- read.table("C:/Users/SAEID/Desktop/Info-DAN/Output/Source Signals (S).txt",
                      header = TRUE,
                      sep = ""
                      )

ggpairs(sources)

z <- as.data.frame(im[["S"]])


# D. Rescale source signals into range [0,1]

minMax <- function(f) {
  (f - min(f)) / (max(f) - min(f))
  }

y1 <- as.data.frame(lapply(z, minMax))


# E. Implement Info-DAN algorithm

im.sae <- autoencoder(
  y1,                       # Input data matrix.
  hidden.layers = c(32,24,16,8,16,24,32), # Vector specifying the
                            # number of neurons in each layer.
  standardize = FALSE,      # Logical indicating if input data
                            # should be standardized.
  loss.type = "squared",    # Loss function: "squared", "absolute",
                            # "huber", "pseudo-huber".
  huber.delta = 1,          # Used only in case of loss functions
                            # "huber" and "pseudo-huber".
  activ.functions = "tanh", # Activation function: "tanh",
                            # "sigmoid", "relu", "linear", "ramp",
                            # "step".
  step.H = 5,               # Number of steps (step activation
                            # function).
  step.k = 300,             # Smoothness (step activation function).
  optim.type = "sgd",       # Optimizer: "sgd", "rmsprop", "adam".
  learn.rates = 0.0001,     # Learning rate.
  L1 = 0,                   # L1 regularization.
  L2 = 0,                   # L2 regularization.
  sgd.momentum = 0.9,       # Momentum.
  rmsprop.decay = 0.9,      # Level of decay in the rms term.
  adam.beta1 = 0.9,         # Level of decay in the first moment
                            # estimate.
  adam.beta2 = 0.999,       # Level of decay in the second moment
                            # estimate.
  n.epochs = 100,           # Number of epochs.
  batch.size = 32,          # Batch size.
  drop.last = TRUE,         # Logical, Only applicable if the size
                            # of the training set is not perfectly
                            # devisible by the batch size.
  val.prop = 0.1,           # Proportion of validation set.
  verbose = TRUE,           # Logical indicating if additional
                            # information should be printed.
  random.seed = NULL        # Seed for random number generator.
  )


# F. Plot loss during training of Info-DAN model

plot(im.sae)


# G. Make reconstruction plot of Info-DAN model

reconstruction_plot(im.sae,
                    y1
                    )


# H. Make compression plot of Info-DAN model

compression_plot(im.sae,
                 y1
                 )


# I. Reconstruct data and calculate anomaly scores from Info-DAN

rec.im.sae <- reconstruct(im.sae,
                          y1
                          )

write.table(rec.im.sae$reconstructed,
            file = 'Reconstruct of Info-DAN.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = FALSE
            )

write.table(rec.im.sae$anomaly_scores,
            file = 'Anomaly Scores of Info-DAN.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = FALSE
            )


# J. Rescale ilr-transformed data into range [0,1]

y2 <- as.data.frame(lapply(x, minMax))


# K. Implement stand-alone DAN algorithm

sae <- autoencoder(
  y2,
  hidden.layers = c(32,24,16,8,16,24,32),
  standardize = FALSE,
  loss.type = "squared",
  huber.delta = 1,
  activ.functions = "tanh",
  step.H = 5,
  step.k = 300,
  optim.type = "sgd",
  learn.rates = 0.0001,
  L1 = 0,
  L2 = 0,
  sgd.momentum = 0.9,
  rmsprop.decay = 0.9,
  adam.beta1 = 0.9,
  adam.beta2 = 0.999,
  n.epochs = 100,
  batch.size = 32,
  drop.last = TRUE,
  val.prop = 0.1,
  verbose = TRUE,
  random.seed = NULL
  )


# L. Plot loss during training of DAN model

plot(sae)


# M. Make reconstruction plot of DAN model

reconstruction_plot(sae,
                    y2
                    )


# N. Make compression plot of DAN model

compression_plot(sae,
                 y2
                 )


# O. Reconstruct data and calculate anomaly scores from DAN

rec.sae <- reconstruct(sae,
                       y2
                       )

write.table(rec.sae$reconstructed,
            file = 'Reconstruct of DAN.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = FALSE
            )

write.table(rec.sae$anomaly_scores,
            file = 'Anomaly Scores of DAN.txt',
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = FALSE
            )
