#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <stddef.h>
#include <omp.h>

#ifndef _SSIZE_T_DEFINED
typedef ptrdiff_t ssize_t;
#define _SSIZE_T_DEFINED
#endif

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

typedef struct {
  int ndata;
  int dim;
  double *features;
  int *assigns;
  int *labels;
  int nlabels;
} KMData;

typedef struct {
  int nclust;
  int dim;
  double *features;
  int *counts;
} KMClust;

static inline double *data_feat(KMData *data, int i) {
  return &data->features[(size_t)i * data->dim];
}

static inline double *clust_feat(KMClust *clust, int c) {
  return &clust->features[(size_t)c * clust->dim];
}

static void ensure_savedir(const char *savedir) {
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", savedir);
  system(cmd);
}

static void die_usage(void) {
  printf("usage: kmeans_omp <datafile> <nclust> [savedir] [maxiter]\n");
  exit(1);
}

static void kmdata_load(const char *datafile, KMData *data) {
  ssize_t ntokens = 0, nlines = 0;
  if (filestats((char *)datafile, &ntokens, &nlines) != 0) {
    exit(1);
  }

  data->ndata = (int)nlines;
  data->dim = (int)(ntokens / nlines) - 2;
  data->features = calloc((size_t)data->ndata * data->dim, sizeof(double));
  data->assigns = calloc(data->ndata, sizeof(int));
  data->labels = calloc(data->ndata, sizeof(int));
  data->nlabels = 0;
  if (!data->features || !data->assigns || !data->labels) {
    printf("memory allocation failure\n");
    exit(1);
  }

  FILE *fin = fopen(datafile, "r");
  if (fin == NULL) {
    printf("Failed to open file '%s'\n", datafile);
    exit(1);
  }

  for (int i = 0; i < data->ndata; i++) {
    int label = -1;
    if (fscanf(fin, "%d", &label) != 1) {
      printf("failed reading label at line %d\n", i + 1);
      exit(1);
    }
    data->labels[i] = label;
    if (label + 1 > data->nlabels) {
      data->nlabels = label + 1;
    }

    char colon = '\0';
    if (fscanf(fin, " %c", &colon) != 1 || colon != ':') {
      printf("failed reading ':' at line %d\n", i + 1);
      exit(1);
    }

    double *feat = data_feat(data, i);
    for (int d = 0; d < data->dim; d++) {
      if (fscanf(fin, "%lf", &feat[d]) != 1) {
        printf("failed reading feature at line %d dim %d\n", i + 1, d);
        exit(1);
      }
    }
  }

  fclose(fin);
}

static void kmclust_new(int nclust, int dim, KMClust *clust) {
  clust->nclust = nclust;
  clust->dim = dim;
  clust->features = calloc((size_t)nclust * dim, sizeof(double));
  clust->counts = calloc(nclust, sizeof(int));
  if (!clust->features || !clust->counts) {
    printf("memory allocation failure\n");
    exit(1);
  }
}

static void save_pgm_files(KMClust *clust, const char *savedir) {
  int dim_root = (int)sqrt((double)clust->dim);
  if (dim_root <= 0 || clust->dim % dim_root != 0) {
    return;
  }

  printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

  double maxfeat = 0.0;
  for (int c = 0; c < clust->nclust; c++) {
    double *cf = clust_feat(clust, c);
    for (int d = 0; d < clust->dim; d++) {
      if (cf[d] > maxfeat) {
        maxfeat = cf[d];
      }
    }
  }

  for (int c = 0; c < clust->nclust; c++) {
    char outfile[1024];
    snprintf(outfile, sizeof(outfile), "%s/cent_%04d.pgm", savedir, c);
    FILE *pgm = fopen(outfile, "w");
    if (pgm == NULL) {
      printf("Failed opening output file '%s'\n", outfile);
      exit(1);
    }
    fprintf(pgm, "P2\n");
    fprintf(pgm, "%d %d\n", dim_root, dim_root);
    fprintf(pgm, "%.0f\n", maxfeat);
    double *cf = clust_feat(clust, c);
    for (int d = 0; d < clust->dim; d++) {
      if (d > 0 && d % dim_root == 0) {
        fprintf(pgm, "\n");
      }
      fprintf(pgm, "%3.0f ", cf[d]);
    }
    fprintf(pgm, "\n");
    fclose(pgm);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    die_usage();
  }

  const char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  const char *savedir = ".";
  int MAXITER = 100;

  if (argc > 3) {
    savedir = argv[3];
    ensure_savedir(savedir);
  }
  if (argc > 4) {
    MAXITER = atoi(argv[4]);
  }

  printf("threads: %d\n", omp_get_max_threads());
  printf("datafile: %s\n", datafile);
  printf("nclust: %d\n", nclust);
  printf("savedir: %s\n", savedir);

  KMData data = {0};
  KMClust clust = {0};
  kmdata_load(datafile, &data);
  kmclust_new(nclust, data.dim, &clust);

  printf("ndata: %d\n", data.ndata);
  printf("dim: %d\n", data.dim);
  printf("\n");

  for (int i = 0; i < data.ndata; i++) {
    int c = i % clust.nclust;
    data.assigns[i] = c;
  }
  for (int c = 0; c < clust.nclust; c++) {
    int icount = data.ndata / clust.nclust;
    int extra = (c < (data.ndata % clust.nclust)) ? 1 : 0;
    clust.counts[c] = icount + extra;
  }

  int curiter = 1;
  int nchanges = data.ndata;
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while (nchanges > 0 && curiter <= MAXITER) {
    size_t feat_sz = (size_t)clust.nclust * clust.dim;
    memset(clust.features, 0, feat_sz * sizeof(double));

    int nthreads = omp_get_max_threads();
    double *local_sums = calloc((size_t)nthreads * feat_sz, sizeof(double));
    if (!local_sums) {
      printf("memory allocation failure\n");
      exit(1);
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double *mine = &local_sums[(size_t)tid * feat_sz];
#pragma omp for schedule(static)
      for (int i = 0; i < data.ndata; i++) {
        int c = data.assigns[i];
        double *df = data_feat(&data, i);
        double *sumc = &mine[(size_t)c * clust.dim];
        for (int d = 0; d < clust.dim; d++) {
          sumc[d] += df[d];
        }
      }
    }

    for (int t = 0; t < nthreads; t++) {
      double *src = &local_sums[(size_t)t * feat_sz];
      for (size_t idx = 0; idx < feat_sz; idx++) {
        clust.features[idx] += src[idx];
      }
    }
    free(local_sums);

#pragma omp parallel for schedule(static)
    for (int c = 0; c < clust.nclust; c++) {
      if (clust.counts[c] > 0) {
        double *cf = clust_feat(&clust, c);
        for (int d = 0; d < clust.dim; d++) {
          cf[d] /= clust.counts[c];
        }
      }
    }

    memset(clust.counts, 0, (size_t)clust.nclust * sizeof(int));
    nchanges = 0;

#pragma omp parallel
    {
      int *local_counts = calloc(clust.nclust, sizeof(int));
      int local_changes = 0;
      if (!local_counts) {
        printf("memory allocation failure\n");
        exit(1);
      }

#pragma omp for schedule(static)
      for (int i = 0; i < data.ndata; i++) {
        int best_clust = -1;
        double best_distsq = INFINITY;
        double *df = data_feat(&data, i);
        for (int c = 0; c < clust.nclust; c++) {
          double distsq = 0.0;
          double *cf = clust_feat(&clust, c);
          for (int d = 0; d < clust.dim; d++) {
            double diff = df[d] - cf[d];
            distsq += diff * diff;
          }
          if (distsq < best_distsq) {
            best_distsq = distsq;
            best_clust = c;
          }
        }
        local_counts[best_clust]++;
        if (best_clust != data.assigns[i]) {
          local_changes++;
          data.assigns[i] = best_clust;
        }
      }

#pragma omp critical
      {
        for (int c = 0; c < clust.nclust; c++) {
          clust.counts[c] += local_counts[c];
        }
        nchanges += local_changes;
      }

      free(local_counts);
    }

    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++) {
      printf(" %4d", clust.counts[c]);
    }
    printf("\n");
    curiter++;
  }

  if (curiter > MAXITER) {
    printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
  } else {
    printf("CONVERGED: after %d iterations\n", curiter);
  }
  printf("\n");

  int *confusion = calloc((size_t)data.nlabels * nclust, sizeof(int));
  if (!confusion) {
    printf("memory allocation failure\n");
    exit(1);
  }

  for (int i = 0; i < data.ndata; i++) {
    confusion[(size_t)data.labels[i] * nclust + data.assigns[i]]++;
  }

  printf("==CONFUSION MATRIX + COUNTS==\n");
  printf("LABEL \\ CLUST\n");
  printf("%2s ", "");
  for (int j = 0; j < clust.nclust; j++) {
    printf(" %4d", j);
  }
  printf(" %4s\n", "TOT");

  for (int i = 0; i < data.nlabels; i++) {
    printf("%2d:", i);
    int tot = 0;
    for (int j = 0; j < clust.nclust; j++) {
      int v = confusion[(size_t)i * nclust + j];
      printf(" %4d", v);
      tot += v;
    }
    printf(" %4d\n", tot);
  }

  printf("TOT");
  int tot = 0;
  for (int c = 0; c < clust.nclust; c++) {
    printf(" %4d", clust.counts[c]);
    tot += clust.counts[c];
  }
  printf(" %4d\n", tot);
  printf("\n");

  char outfile[1024];
  snprintf(outfile, sizeof(outfile), "%s/labels.txt", savedir);
  printf("Saving cluster labels to file %s\n", outfile);
  FILE *fout = fopen(outfile, "w");
  if (fout == NULL) {
    printf("Failed opening output file '%s'\n", outfile);
    exit(1);
  }
  for (int i = 0; i < data.ndata; i++) {
    fprintf(fout, "%2d %2d\n", data.labels[i], data.assigns[i]);
  }
  fclose(fout);

  save_pgm_files(&clust, savedir);

  free(confusion);
  free(clust.counts);
  free(clust.features);
  free(data.labels);
  free(data.assigns);
  free(data.features);
  return 0;
}
