#!/usr/bin/env python3
#
# usage: kmeans.py <datafile> <nclust> [savedir] [maxiter]

import struct
import sys
import os
import math

class KMData:                      # Data set to be clustered
  ndata    = 0                     # count of data
  dim      = 0                     # dimension of features for data
  features = []                    # pointers to individual features
  assigns  = []                    # cluster to which data is assigned
  labels   = []                    # label for data if available
  nlabels  = []                    # max value of labels +1, number 0,1,...,nlabel0

class KMClust:                     # Cluster information
  nclust   = 0                     # number of clusters, the "k" in kmeans
  dim      = 0                     # dimension of features for data
  features = []                    # 2D indexing for individual cluster center features
  counts   = []                    # number of data in each cluster
  

# Load a data set from the named file. Data should be formatted as a
# text file as:
# 
# 7 :  84 185 159 151  60  36   0   0   0   0   0   0
# 2 :   0  77 251 210  25   0   0   0 122 248 253  65
# 1 :   0   0   0   0   0   0   0   0   0  45 244 150
# 0 :   0   0   0   0 110 190 251 251 251 253 169 109
# 4 :   0   0   0   4 195 231   0   0   0   0   0   0
# 1 :   0   0   0   0   0   0   0   0   0  81 254 254
# 4 :   0  20 189 253 147   0   0   0   0   0   0   0
# 9 :   0   0   0   0  91 224 253 253  19   0   0   0
# 5 :   0   0   0   0   0   0  63 253 253 253 253 253
# 9 :   0   0   0   0   0   0   0  36  56 137 201 199
#
# with the lead number being an optional correct label for the data
# and remaining numbers being floating point values that are space
# separated which are the feature vector for each data. The abve
# example does not have any fractional values for features but it
# could.
def kmdata_load(datafile):
  data = KMData()
  with open(datafile,"r") as fin:
    for line in fin:
      data.ndata += 1
      tokens = line.split()
      data.labels.append(int(tokens[0]))
      feats = []
      for f in tokens[2:]:
        feats.append(float(f))
      data.features.append(feats)
  
  data.dim = len(data.features[0])
  data.nlabels = max(data.labels)+1
  return data

# Allocate space for clusters in an object
def kmclust_new(nclust, dim):
  clust = KMClust()
  clust.nclust = nclust
  clust.dim = dim
  for c in range(nclust):
    clust.features.append([0.0] * dim)
    clust.counts.append(0)
  return clust
  

# Save clust centers in the PGM (portable gray map) image format;
# plain text and can be displayed in many image viewers. File names re
# cent_0000.pgm and so on.
def save_pgm_files(clust,savedir):
  dim_root = int(math.sqrt(clust.dim))
  if clust.dim % dim_root == 0:                    # check if this looks like a square image
    print(f"Saving cluster centers to {savedir}/cent_0000.pgm ...")
    maxfeat = max(map(max,clust.features))
    for c in range(clust.nclust):
      outfile=f"{savedir}/cent_{c:04}.pgm"
      with open(outfile,"w") as pgm:               # output the cluster centers as
        print("P2",file=pgm)                       # pgm files, a simple image format which
        print(f"{dim_root} {dim_root}",file=pgm)   # can be viewed in most image
        print(f"{maxfeat:.0f}",file=pgm)           # viewers to show how the cluster center
        for d in range(clust.dim):                 # actually appears; nomacs is a good viwer
          if d > 0 and d%dim_root == 0:
            print(file=pgm)
          print(f"{clust.features[c][d]:3.0f} ",end="",file=pgm)
        print(file=pgm)

#### MAIN FUNCTION ####
def main():
  if len(sys.argv) < 3:
    sys.exit("usage: kmeans.py <datafile> <nclust> [savedir] [maxiter]")
  
  datafile = sys.argv[1]
  nclust = int(sys.argv[2])
  savedir = "."
  MAXITER = 100                        # bounds the iterations
  
  if len(sys.argv) > 3:                # create save directory if specified
    savedir = sys.argv[3]
    os.system(f"mkdir -p {savedir}")
  
  if len(sys.argv) > 4:
    MAXITER = int(sys.argv[4])
  
  print("datafile:",datafile)
  print("nclust:",nclust)
  print("savedir:",savedir)
  

  data = kmdata_load(datafile)         # read in the data file, allocate cluster space
  clust = kmclust_new(nclust, data.dim)

  print("ndata:",data.ndata)
  print("dim:",data.dim)
  print()
  
  for i in range(data.ndata):          # random, regular initial cluster assignment
    c = i % clust.nclust
    data.assigns.append(c)

  for c in range(clust.nclust):
    icount = data.ndata // clust.nclust  # IMPORTANT: use integer division via //
    extra = 0
    if c < (data.ndata % clust.nclust):
      extra = 1                        # extras in earlier clusters
    clust.counts[c] = icount + extra;
  
  
  ################################################################################
  # THE MAIN ALGORITHM
  curiter = 1                          # current iteration
  nchanges = data.ndata                # check for changes in cluster assignment; 0 is converged
  print(f"==CLUSTERING: MAXITER {MAXITER}==")
  print(f"ITER NCHANGE CLUST_COUNTS")
  
  while nchanges > 0 and curiter <= MAXITER: # loop until convergence

    # DETERMINE NEW CLUSTER CENTERS
    for c in range(clust.nclust):      # reset cluster centers to 0.0
      for d in range(clust.dim):
        clust.features[c][d] = 0.0
  
    for i in range(data.ndata):        # sum up data in each cluster
      c = data.assigns[i]
      for d in range(clust.dim):
        clust.features[c][d] += data.features[i][d]
  
    for c in range(clust.nclust):      # divide by ndatas of data to get mean of cluster center
      if clust.counts[c] > 0:
        for d in range(clust.dim):        
          clust.features[c][d] = clust.features[c][d] / clust.counts[c]
        
    # DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
    for c in range(clust.nclust):      # reset cluster counts to 0
      clust.counts[c] = 0              # re-init here to support first iteration
  
    nchanges = 0
    for i in range(data.ndata):        # iterate over all data
      best_clust = None
      best_distsq = float("inf")
      for c in range(clust.nclust):    # compare data to each cluster and assign to closest
        distsq = 0.0
        for d in range(clust.dim):     # calculate squared distance to each data dimension
          diff = data.features[i][d] - clust.features[c][d]
          distsq += diff*diff
          # print(f"DBG: curiter {curiter} i {i} c {c} d {d} diff {diff:.6f}")
        if distsq < best_distsq:       # if closer to this cluster than current best
          best_clust = c
          best_distsq = distsq
      clust.counts[best_clust] += 1
      if best_clust != data.assigns[i]:  # assigning data to a different cluster?
        # print(f"DBG: data {i} changing clusters: {data_assign[i]} to {best_clust}")
        nchanges += 1                    # indicate cluster assignment has changed
        data.assigns[i] = best_clust     # assign to new cluster
    
    # Print iteration information at the end of the iter
    print(f"{curiter:3}: {nchanges:5} |", end="") 
    for c in range(nclust):
      print(f" {clust.counts[c]:4}",end="")
    print()
    curiter += 1
  
  # Loop has converged
  if curiter > MAXITER:
    print(f"WARNING: maximum iteration {MAXITER} exceeded, may not have conveged")
  else:
    print(f"CONVERGED: after {curiter} iterations")
  print()
  

  ################################################################################
  # CLEANUP + OUTPUT

  # CONFUSION MATRIX
  confusion = []                         # confusion matrix: labels * clusters big
  for i in range(data.nlabels):
    confusion.append([0] * nclust)
  
  for i in range(data.ndata):            # count which labels in which clusters
    confusion[data.labels[i]][data.assigns[i]] += 1
  
  print("==CONFUSION MATRIX + COUNTS==")
  print("LABEL \ CLUST")
  
  print(f"{'':2} ",end="")               # confusion matrix header
  for j in range(clust.nclust):
    print(f" {j:>4}",end="")
  print(f" {'TOT':>4}")
  
  for i in range(data.nlabels):          # each row of confusion matrix
    print(f"{i:>2}:",end="")
    tot = 0
    for j in range(clust.nclust):
      print(f" {confusion[i][j]:>4}",end="")
      tot += confusion[i][j]
    print(f" {tot:>4}")
  
  print("TOT",end="")                    # final total row of confusion matrix
  tot = 0
  for c in range(clust.nclust):
    print(f" {clust.counts[c]:>4}",end="")
    tot += clust.counts[c]
  print(f" {tot:>4}")
  print()
  
  # LABEL FILE OUTPUT
  outfile = f"{savedir}/labels.txt"    
  print(f"Saving cluster labels to file {outfile}")
  with open(outfile,"w") as fout:
    for i in range(data.ndata):
      print(f"{data.labels[i]:2} {data.assigns[i]:2}",file=fout)
  

  # SAVE PGM FILES CONDITIONALLY
  save_pgm_files(clust,savedir)
### END MAIN ###      
  
main()
