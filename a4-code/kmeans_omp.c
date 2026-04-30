// kmeans_omp.c
// C port of the provided kmeans.py program.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <float.h>
#include <sys/types.h>
#include <stddef.h>
#include <omp.h>

#ifndef _SSIZE_T_DEFINED
typedef ptrdiff_t ssize_t;
#define _SSIZE_T_DEFINED
#endif

// Provided in kmeans_util.c
int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

// Data set to be clustered
typedef struct {
  int ndata;        // count of data
  int dim;          // dimension of features
  double *features; // flat array: ndata x dim
  int *assigns;     // cluster assignment for each data point
  int *labels;      // label for each data point
  int nlabels;      // max label + 1
} KMData;

// Cluster information
typedef struct {
  int nclust;       // number of clusters
  int dim;          // dimension of features
  double *features; // flat array: nclust x dim
  int *counts;      // number of data points in each cluster
} KMClust;

// Load data from file. Format per line: label : feat0 feat1 feat2 ...
KMData kmdata_load(char *datafile){
  KMData data;
  data.ndata = 0;
  data.dim = 0;

  ssize_t tot_tokens, tot_lines;
  filestats(datafile, &tot_tokens, &tot_lines);
  data.ndata = (int)tot_lines;

  // tokens per line = 1 (label) + 1 (:) + dim
  int tokens_per_line = (int)(tot_tokens / tot_lines);
  data.dim = tokens_per_line - 2;  // subtract label and colon

  data.features = malloc(sizeof(double) * data.ndata * data.dim);
  data.labels   = malloc(sizeof(int) * data.ndata);
  data.assigns  = malloc(sizeof(int) * data.ndata);

  FILE *fin = fopen(datafile, "r");
  for(int i = 0; i < data.ndata; i++){
    int label;
    char colon;
    fscanf(fin, "%d %c", &label, &colon);
    data.labels[i] = label;
    for(int d = 0; d < data.dim; d++){
      fscanf(fin, "%lf", &data.features[i * data.dim + d]);
    }
  }
  fclose(fin);

  // Compute nlabels = max label + 1
  data.nlabels = 0;
  for(int i = 0; i < data.ndata; i++){
    if(data.labels[i] + 1 > data.nlabels){
      data.nlabels = data.labels[i] + 1;
    }
  }

  return data;
}

// Allocate and initialize cluster struct
KMClust kmclust_new(int nclust, int dim){
  KMClust clust;
  clust.nclust = nclust;
  clust.dim = dim;
  clust.features = malloc(sizeof(double) * nclust * dim);
  clust.counts   = malloc(sizeof(int) * nclust);
  for(int c = 0; c < nclust; c++){
    clust.counts[c] = 0;
    for(int d = 0; d < dim; d++){
      clust.features[c * dim + d] = 0.0;
    }
  }
  return clust;
}

// Save cluster centers as PGM image files
void save_pgm_files(KMClust *clust, char *savedir){
  int dim_root = (int)sqrt((double)clust->dim);
  if(clust->dim % dim_root != 0){
    return;
  }
  printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

  // Find max feature value across all cluster centers
  double maxfeat = 0.0;
  for(int c = 0; c < clust->nclust; c++){
    for(int d = 0; d < clust->dim; d++){
      if(clust->features[c * clust->dim + d] > maxfeat){
        maxfeat = clust->features[c * clust->dim + d];
      }
    }
  }

  for(int c = 0; c < clust->nclust; c++){
    char outfile[512];
    snprintf(outfile, sizeof(outfile), "%s/cent_%04d.pgm", savedir, c);
    FILE *pgm = fopen(outfile, "w");
    fprintf(pgm, "P2\n");
    fprintf(pgm, "%d %d\n", dim_root, dim_root);
    fprintf(pgm, "%.0f\n", maxfeat);
    for(int d = 0; d < clust->dim; d++){
      if(d > 0 && d % dim_root == 0){
        fprintf(pgm, "\n");
      }
      fprintf(pgm, "%3.0f ", clust->features[c * clust->dim + d]);
    }
    fprintf(pgm, "\n");
    fclose(pgm);
  }
}

int main(int argc, char **argv){
  if(argc < 3){
    printf("usage: %s <datafile> <nclust> [savedir] [maxiter]\n", argv[0]);
    return 1;
  }

  char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  char *savedir = ".";
  int MAXITER = 100;

  if(argc > 3){
    savedir = argv[3];
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", savedir);
    system(cmd);
  }
  if(argc > 4){
    MAXITER = atoi(argv[4]);
  }

  printf("threads: %d\n", omp_get_max_threads());
  printf("datafile: %s\n", datafile);
  printf("nclust: %d\n", nclust);
  printf("savedir: %s\n", savedir);

  KMData data = kmdata_load(datafile);
  KMClust clust = kmclust_new(nclust, data.dim);

  printf("ndata: %d\n", data.ndata);
  printf("dim: %d\n", data.dim);
  printf("\n");

  // Initial cluster assignment: round-robin
  for(int i = 0; i < data.ndata; i++){
    data.assigns[i] = i % nclust;
  }

  // Initial cluster counts
  for(int c = 0; c < nclust; c++){
    int icount = data.ndata / nclust;
    int extra = (c < (data.ndata % nclust)) ? 1 : 0;
    clust.counts[c] = icount + extra;
  }

  // Main K-means loop
  int curiter = 1;
  int nchanges = data.ndata;
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while(nchanges > 0 && curiter <= MAXITER){
    size_t feat_sz = (size_t)nclust * data.dim;

    // Reset cluster centers to 0
    memset(clust.features, 0, feat_sz * sizeof(double));

    // Sum features for each cluster (parallel reduction via thread-local buffers)
    int nthreads = omp_get_max_threads();
    double *local_sums = calloc((size_t)nthreads * feat_sz, sizeof(double));

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double *my_sums = &local_sums[(size_t)tid * feat_sz];

#pragma omp for schedule(static)
      for(int i = 0; i < data.ndata; i++){
        int c = data.assigns[i];
        double *sum_base = &my_sums[(size_t)c * data.dim];
        for(int d = 0; d < data.dim; d++){
          sum_base[d] += data.features[(size_t)i * data.dim + d];
        }
      }
    }

    for(int t = 0; t < nthreads; t++){
      double *thread_sums = &local_sums[(size_t)t * feat_sz];
      for(size_t idx = 0; idx < feat_sz; idx++){
        clust.features[idx] += thread_sums[idx];
      }
    }
    free(local_sums);

    // Divide by count to get mean
#pragma omp parallel for schedule(static)
    for(int c = 0; c < nclust; c++){
      if(clust.counts[c] > 0){
        for(int d = 0; d < data.dim; d++){
          clust.features[c * data.dim + d] /= clust.counts[c];
        }
      }
    }

    // Reset cluster counts
    memset(clust.counts, 0, (size_t)nclust * sizeof(int));

    // Assign each data point to nearest cluster
    nchanges = 0;
#pragma omp parallel
    {
      int *local_counts = calloc(nclust, sizeof(int));
      int local_changes = 0;

#pragma omp for schedule(static)
      for(int i = 0; i < data.ndata; i++){
        int best_clust = 0;
        double best_distsq = DBL_MAX;
        for(int c = 0; c < nclust; c++){
          double distsq = 0.0;
          for(int d = 0; d < data.dim; d++){
            double diff = data.features[i * data.dim + d] - clust.features[c * data.dim + d];
            distsq += diff * diff;
          }
          if(distsq < best_distsq){
            best_clust = c;
            best_distsq = distsq;
          }
        }
        local_counts[best_clust]++;
        if(best_clust != data.assigns[i]){
          local_changes++;
          data.assigns[i] = best_clust;
        }
      }

#pragma omp critical
      {
        for(int c = 0; c < nclust; c++){
          clust.counts[c] += local_counts[c];
        }
        nchanges += local_changes;
      }
      free(local_counts);
    }

    // Print iteration info
    printf("%3d: %5d |", curiter, nchanges);
    for(int c = 0; c < nclust; c++){
      printf(" %4d", clust.counts[c]);
    }
    printf("\n");
    curiter++;
  }

  // Convergence message
  if(curiter > MAXITER){
    printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
  } else {
    printf("CONVERGED: after %d iterations\n", curiter);
  }
  printf("\n");

  // Confusion matrix
  int *confusion = calloc(data.nlabels * nclust, sizeof(int));
  for(int i = 0; i < data.ndata; i++){
    confusion[data.labels[i] * nclust + data.assigns[i]]++;
  }

  printf("==CONFUSION MATRIX + COUNTS==\n");
  printf("LABEL \\ CLUST\n");

  // Header
  printf("%2s ", "");
  for(int j = 0; j < nclust; j++){
    printf(" %4d", j);
  }
  printf(" %4s\n", "TOT");

  // Rows
  for(int i = 0; i < data.nlabels; i++){
    printf("%2d:", i);
    int tot = 0;
    for(int j = 0; j < nclust; j++){
      printf(" %4d", confusion[i * nclust + j]);
      tot += confusion[i * nclust + j];
    }
    printf(" %4d\n", tot);
  }

  // Total row
  printf("TOT");
  int tot = 0;
  for(int c = 0; c < nclust; c++){
    printf(" %4d", clust.counts[c]);
    tot += clust.counts[c];
  }
  printf(" %4d\n", tot);
  printf("\n");

  // Save labels file
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/labels.txt", savedir);
  printf("Saving cluster labels to file %s\n", outfile);
  FILE *fout = fopen(outfile, "w");
  for(int i = 0; i < data.ndata; i++){
    fprintf(fout, "%2d %2d\n", data.labels[i], data.assigns[i]);
  }
  fclose(fout);

  // Save PGM files
  save_pgm_files(&clust, savedir);

  // Free memory
  free(confusion);
  free(data.features);
  free(data.labels);
  free(data.assigns);
  free(clust.features);
  free(clust.counts);

  return 0;
}
