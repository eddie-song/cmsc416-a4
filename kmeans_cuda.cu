#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <ctype.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                            \
    cudaError_t _e = (expr);                                             \
    if (_e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(_e));               \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// Host-side data structures. Features are stored row-major in a single
// flat buffer (data_flat[i*dim + d]) so the same layout can be shipped
// to the GPU directly.
typedef struct {
  int    ndata;
  int    dim;
  double *features;   // flat ndata*dim
  int    *assigns;
  int    *labels;
  int    nlabels;
} KMData;

typedef struct {
  int    nclust;
  int    dim;
  double *features;   // flat nclust*dim
  int    *counts;
} KMClust;

// ===================== Device kernels =====================

// One thread per (cluster, dim) cell. Each thread sums the d-th
// feature over all data points assigned to cluster c, then divides by
// counts[c] to get the mean. No atomics, deterministic sum order.
__global__ void kernel_compute_centers(int ndata, int dim, int nclust,
                                       const double *data, const int *assigns,
                                       const int *counts, double *centers) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nclust * dim;
  if (tid >= total) return;
  int c = tid / dim;
  int d = tid % dim;

  double sum = 0.0;
  for (int i = 0; i < ndata; i++) {
    if (assigns[i] == c)
      sum += data[i * dim + d];
  }
  // Match serial: leave at 0.0 if cluster is empty (no division).
  centers[tid] = (counts[c] > 0) ? (sum / (double)counts[c]) : 0.0;
}

// Zero out an int array (used for counts and nchanges).
__global__ void kernel_zero_ints(int *arr, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = 0;
}

// One thread per data point: find nearest cluster, update assignment,
// atomically bump counts and nchanges.
__global__ void kernel_assign(int ndata, int dim, int nclust,
                              const double *data, const double *centers,
                              int *assigns, int *counts, int *nchanges) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndata) return;

  int    best_clust  = -1;
  double best_distsq = DBL_MAX;
  for (int c = 0; c < nclust; c++) {
    double distsq = 0.0;
    for (int d = 0; d < dim; d++) {
      double diff = data[i * dim + d] - centers[c * dim + d];
      distsq += diff * diff;
    }
    if (distsq < best_distsq) {
      best_clust  = c;
      best_distsq = distsq;
    }
  }
  atomicAdd(&counts[best_clust], 1);
  if (best_clust != assigns[i]) {
    atomicAdd(nchanges, 1);
    assigns[i] = best_clust;
  }
}

// ===================== Host I/O helpers =====================

static int count_lines(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open file '%s': %s\n", filename, strerror(errno));
    exit(1);
  }
  int count = 0;
  char buf[65536];
  while (fgets(buf, sizeof(buf), f)) count++;
  fclose(f);
  return count;
}

static int count_dim(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open file '%s': %s\n", filename, strerror(errno));
    exit(1);
  }
  char buf[65536];
  if (!fgets(buf, sizeof(buf), f)) {
    fprintf(stderr, "ERROR: empty data file '%s'\n", filename);
    fclose(f); exit(1);
  }
  fclose(f);

  int dim = 0;
  char *tok = strtok(buf, " \t\n\r");
  tok = strtok(NULL, " \t\n\r");
  while ((tok = strtok(NULL, " \t\n\r")) != NULL) dim++;
  return dim;
}

KMData *kmdata_load(const char *datafile) {
  int ndata = count_lines(datafile);
  int dim   = count_dim(datafile);

  KMData *data = (KMData *)malloc(sizeof(KMData));
  if (!data) { perror("malloc"); exit(1); }
  data->ndata = ndata;
  data->dim   = dim;

  data->features = (double *)malloc(sizeof(double) * (size_t)ndata * dim);
  data->labels   = (int *)malloc(sizeof(int) * ndata);
  data->assigns  = (int *)malloc(sizeof(int) * ndata);
  if (!data->features || !data->labels || !data->assigns) {
    perror("malloc"); exit(1);
  }

  FILE *f = fopen(datafile, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open file '%s': %s\n", datafile, strerror(errno));
    exit(1);
  }

  char buf[65536];
  int idx = 0;
  while (fgets(buf, sizeof(buf), f) && idx < ndata) {
    char *tok = strtok(buf, " \t\n\r");
    data->labels[idx] = atoi(tok);
    tok = strtok(NULL, " \t\n\r");

    for (int d = 0; d < dim; d++) {
      tok = strtok(NULL, " \t\n\r");
      data->features[(size_t)idx * dim + d] = tok ? atof(tok) : 0.0;
    }
    idx++;
  }
  fclose(f);

  int maxlabel = 0;
  for (int i = 0; i < ndata; i++) {
    if (data->labels[i] > maxlabel) maxlabel = data->labels[i];
  }
  data->nlabels = maxlabel + 1;
  return data;
}

void kmdata_free(KMData *data) {
  if (!data) return;
  free(data->features);
  free(data->labels);
  free(data->assigns);
  free(data);
}

KMClust *kmclust_new(int nclust, int dim) {
  KMClust *clust = (KMClust *)malloc(sizeof(KMClust));
  if (!clust) { perror("malloc"); exit(1); }
  clust->nclust = nclust;
  clust->dim    = dim;
  clust->features = (double *)calloc((size_t)nclust * dim, sizeof(double));
  clust->counts   = (int *)calloc(nclust, sizeof(int));
  if (!clust->features || !clust->counts) { perror("calloc"); exit(1); }
  return clust;
}

void kmclust_free(KMClust *clust) {
  if (!clust) return;
  free(clust->features);
  free(clust->counts);
  free(clust);
}

void save_pgm_files(KMClust *clust, const char *savedir) {
  int dim_root = (int)sqrt((double)clust->dim);
  if (clust->dim % dim_root != 0) return;

  printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

  double maxfeat = 0.0;
  for (int c = 0; c < clust->nclust; c++) {
    for (int d = 0; d < clust->dim; d++) {
      double v = clust->features[(size_t)c * clust->dim + d];
      if (v > maxfeat) maxfeat = v;
    }
  }

  for (int c = 0; c < clust->nclust; c++) {
    char outfile[1024];
    snprintf(outfile, sizeof(outfile), "%s/cent_%04d.pgm", savedir, c);
    FILE *pgm = fopen(outfile, "w");
    if (!pgm) {
      fprintf(stderr, "WARNING: cannot open '%s' for writing: %s\n", outfile, strerror(errno));
      continue;
    }
    fprintf(pgm, "P2\n");
    fprintf(pgm, "%d %d\n", dim_root, dim_root);
    fprintf(pgm, "%.0f\n", maxfeat);
    for (int d = 0; d < clust->dim; d++) {
      if (d > 0 && d % dim_root == 0) fprintf(pgm, "\n");
      fprintf(pgm, "%3.0f ", clust->features[(size_t)c * clust->dim + d]);
    }
    fprintf(pgm, "\n");
    fclose(pgm);
  }
}

// ===================== Main =====================

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: kmeans_cuda <datafile> <nclust> [savedir] [maxiter]\n");
    return 1;
  }

  const char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  const char *savedir = ".";
  int MAXITER = 100;

  if (argc > 3) {
    savedir = argv[3];
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", savedir);
    system(cmd);
  }
  if (argc > 4) MAXITER = atoi(argv[4]);

  printf("datafile: %s\n", datafile);
  printf("nclust: %d\n", nclust);
  printf("savedir: %s\n", savedir);

  KMData  *data  = kmdata_load(datafile);
  KMClust *clust = kmclust_new(nclust, data->dim);

  printf("ndata: %d\n", data->ndata);
  printf("dim: %d\n", data->dim);
  printf("\n");

  // Initial assignments: round-robin (matches serial).
  for (int i = 0; i < data->ndata; i++)
    data->assigns[i] = i % clust->nclust;

  for (int c = 0; c < clust->nclust; c++) {
    int icount = data->ndata / clust->nclust;
    int extra = (c < (data->ndata % clust->nclust)) ? 1 : 0;
    clust->counts[c] = icount + extra;
  }

  // ===================== GPU setup =====================
  size_t data_bytes    = sizeof(double) * (size_t)data->ndata * data->dim;
  size_t centers_bytes = sizeof(double) * (size_t)clust->nclust * clust->dim;
  size_t assigns_bytes = sizeof(int)    * (size_t)data->ndata;
  size_t counts_bytes  = sizeof(int)    * (size_t)clust->nclust;

  double *d_data, *d_centers;
  int    *d_assigns, *d_counts, *d_nchanges;

  CUDA_CHECK(cudaMalloc((void **)&d_data,     data_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_centers,  centers_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_assigns,  assigns_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_counts,   counts_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_nchanges, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_data,    data->features, data_bytes,    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_assigns, data->assigns,  assigns_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_counts,  clust->counts,  counts_bytes,  cudaMemcpyHostToDevice));

  // Launch parameters.
  const int THREADS_PER_BLOCK = 256;
  int centers_total = clust->nclust * clust->dim;
  int blocks_centers = (centers_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocks_data    = (data->ndata    + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocks_counts  = (clust->nclust  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // ===================== Main loop =====================
  int curiter = 1;
  int nchanges = data->ndata;
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while (nchanges > 0 && curiter <= MAXITER) {
    // Phase 1: compute new centers (sum + divide fused).
    kernel_compute_centers<<<blocks_centers, THREADS_PER_BLOCK>>>(
        data->ndata, clust->dim, clust->nclust,
        d_data, d_assigns, d_counts, d_centers);

    // Reset counts[] and nchanges for the assignment phase.
    kernel_zero_ints<<<blocks_counts, THREADS_PER_BLOCK>>>(d_counts, clust->nclust);
    CUDA_CHECK(cudaMemset(d_nchanges, 0, sizeof(int)));

    // Phase 2: reassign each data point to nearest center.
    kernel_assign<<<blocks_data, THREADS_PER_BLOCK>>>(
        data->ndata, clust->dim, clust->nclust,
        d_data, d_centers, d_assigns, d_counts, d_nchanges);

    // Pull back nchanges and counts for printing + convergence check.
    CUDA_CHECK(cudaMemcpy(&nchanges,      d_nchanges, sizeof(int),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clust->counts,  d_counts,   counts_bytes, cudaMemcpyDeviceToHost));

    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++)
      printf(" %4d", clust->counts[c]);
    printf("\n");
    curiter++;
  }

  if (curiter > MAXITER)
    printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
  else
    printf("CONVERGED: after %d iterations\n", curiter);
  printf("\n");

  // Bring centers and assignments back for PGM + confusion matrix.
  CUDA_CHECK(cudaMemcpy(clust->features, d_centers, centers_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(data->assigns,   d_assigns, assigns_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_data);
  cudaFree(d_centers);
  cudaFree(d_assigns);
  cudaFree(d_counts);
  cudaFree(d_nchanges);

  // ===================== Confusion matrix + outputs =====================
  int **confusion = (int **)malloc(sizeof(int *) * data->nlabels);
  if (!confusion) { perror("malloc"); exit(1); }
  for (int i = 0; i < data->nlabels; i++) {
    confusion[i] = (int *)calloc(nclust, sizeof(int));
    if (!confusion[i]) { perror("calloc"); exit(1); }
  }
  for (int i = 0; i < data->ndata; i++)
    confusion[data->labels[i]][data->assigns[i]] += 1;

  printf("==CONFUSION MATRIX + COUNTS==\n");
  printf("LABEL \\ CLUST\n");
  printf("%2s ", "");
  for (int j = 0; j < clust->nclust; j++) printf(" %4d", j);
  printf(" %4s\n", "TOT");

  for (int i = 0; i < data->nlabels; i++) {
    printf("%2d:", i);
    int tot = 0;
    for (int j = 0; j < clust->nclust; j++) {
      printf(" %4d", confusion[i][j]);
      tot += confusion[i][j];
    }
    printf(" %4d\n", tot);
  }

  printf("TOT");
  int tot = 0;
  for (int c = 0; c < clust->nclust; c++) {
    printf(" %4d", clust->counts[c]);
    tot += clust->counts[c];
  }
  printf(" %4d\n", tot);
  printf("\n");

  char outfile[1024];
  snprintf(outfile, sizeof(outfile), "%s/labels.txt", savedir);
  printf("Saving cluster labels to file %s\n", outfile);
  FILE *fout = fopen(outfile, "w");
  if (!fout) {
    fprintf(stderr, "WARNING: cannot open '%s' for writing: %s\n", outfile, strerror(errno));
  } else {
    for (int i = 0; i < data->ndata; i++)
      fprintf(fout, "%2d %2d\n", data->labels[i], data->assigns[i]);
    fclose(fout);
  }

  save_pgm_files(clust, savedir);

  for (int i = 0; i < data->nlabels; i++) free(confusion[i]);
  free(confusion);
  kmclust_free(clust);
  kmdata_free(data);

  return 0;
}

