// usage: kmeans_cuda <datafile> <nclust> [savedir] [maxiter]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <cuda_runtime.h>

// Macro to check that cuda operations succeed and bail if not
// (lecture slide 18 pattern). Used around every cudaMalloc / cudaMemcpy
// / cudaMemset call below.
#define CUDA_CHECK(expr) do {                                            \
    cudaError_t _e = (expr);                                             \
    if (_e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(_e));               \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// Data set to be clustered
typedef struct {
  int    ndata;       // count of data
  int    dim;         // dimension of features for data
  double *features;   // flat ndata*dim row-major (Needs single contiguous block to ship to GPU)
  int    *assigns;    // cluster to which data is assigned
  int    *labels;     // label for data if available
  int    nlabels;     // max value of labels +1, number 0,1,...,nlabel0
} KMData;

// Cluster information
typedef struct {
  int    nclust;      // number of clusters, the "k" in kmeans
  int    dim;         // dimension of features for data
  double *features;   // flat nclust*dim row-major (Needs single contiguous block to ship to GPU)
  int    *counts;     // number of data in each cluster
} KMClust;

static int count_lines(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open file '%s': %s\n", filename, strerror(errno));
    exit(1);
  }
  int count = 0;
  char buf[65536];
  while (fgets(buf, sizeof(buf), f)) {
    count++;
  }
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
    fclose(f);
    exit(1);
  }
  fclose(f);

  int dim = 0;
  char *tok = strtok(buf, " \t\n\r");  // label
  tok = strtok(NULL, " \t\n\r");        // colon ":"
  while ((tok = strtok(NULL, " \t\n\r")) != NULL) {
    dim++;
  }
  return dim;
}

// Load a data set from the named file. Data should be formatted as a
// text file as:
//
// 7 :  84 185 159 151  60  36   0   0   0   0   0   0
// 2 :   0  77 251 210  25   0   0   0 122 248 253  65
// 1 :   0   0   0   0   0   0   0   0   0  45 244 150
// 0 :   0   0   0   0 110 190 251 251 251 253 169 109
// 4 :   0   0   0   4 195 231   0   0   0   0   0   0
// 1 :   0   0   0   0   0   0   0   0   0  81 254 254
// 4 :   0  20 189 253 147   0   0   0   0   0   0   0
// 9 :   0   0   0   0  91 224 253 253  19   0   0   0
// 5 :   0   0   0   0   0   0  63 253 253 253 253 253
// 9 :   0   0   0   0   0   0   0  36  56 137 201 199
//
// with the lead number being an optional correct label for the data
// and remaining numbers being floating point values that are space
// separated which are the feature vector for each data. The abve
// example does not have any fractional values for features but it
// could.
KMData *kmdata_load(const char *datafile) {
  int ndata = count_lines(datafile);
  int dim   = count_dim(datafile);

  KMData *data = (KMData *)malloc(sizeof(KMData));
  if (!data) { perror("malloc"); exit(1); }

  data->ndata = ndata;
  data->dim   = dim;

  // Single flat ndata*dim block (vs serial's array-of-pointers)
  // so the same buffer can be shipped straight to the GPU.
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
    char *tok = strtok(buf, " \t\n\r");   // label
    data->labels[idx] = atoi(tok);
    tok = strtok(NULL, " \t\n\r");          // colon ":"

    for (int d = 0; d < dim; d++) {
      tok = strtok(NULL, " \t\n\r");
      if (tok) {
        data->features[(size_t)idx * dim + d] = atof(tok);
      } else {
        data->features[(size_t)idx * dim + d] = 0.0;
      }
    }
    idx++;
  }
  fclose(f);

  int maxlabel = 0;
  for (int i = 0; i < ndata; i++) {
    if (data->labels[i] > maxlabel)
      maxlabel = data->labels[i];
  }
  data->nlabels = maxlabel + 1;

  return data;
}

void kmdata_free(KMData *data) {
  if (!data) return;
  free(data->features); // single buffer, single free
  free(data->assigns);
  free(data->labels);
  free(data);
}

// Allocate space for clusters in an object
KMClust *kmclust_new(int nclust, int dim) {
  KMClust *clust = (KMClust *)malloc(sizeof(KMClust));
  if (!clust) { perror("malloc"); exit(1); }

  clust->nclust = nclust;
  clust->dim    = dim;

  // single flat nclust*dim block (vs serial's array-of-pointers)
  clust->features = (double *)calloc((size_t)nclust * dim, sizeof(double));
  clust->counts   = (int *)calloc(nclust, sizeof(int));
  if (!clust->features || !clust->counts) {
    perror("calloc"); exit(1);
  }

  return clust;
}

void kmclust_free(KMClust *clust) {
  if (!clust) return;
  free(clust->features); // single buffer, single free
  free(clust->counts);
  free(clust);
}

// Save clust centers in the PGM (portable gray map) image format;
// plain text and can be displayed in many image viewers. File names re
// cent_0000.pgm and so on.
void save_pgm_files(KMClust *clust, const char *savedir) {
  int dim_root = (int)sqrt((double)clust->dim);
  if (clust->dim % dim_root != 0)      // check if this looks like a square image
    return;

  printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

  double maxfeat = 0.0;
  for (int c = 0; c < clust->nclust; c++) {
    for (int d = 0; d < clust->dim; d++) {
      double v = clust->features[(size_t)c * clust->dim + d];
      if (v > maxfeat)
        maxfeat = v;
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
    fprintf(pgm, "P2\n");                           // output the cluster centers as
    fprintf(pgm, "%d %d\n", dim_root, dim_root);    // pgm files, a simple image format which
    fprintf(pgm, "%.0f\n", maxfeat);                // can be viewed in most image
    for (int d = 0; d < clust->dim; d++) {           // viewers to show how the cluster center
      if (d > 0 && d % dim_root == 0)
        fprintf(pgm, "\n");
      fprintf(pgm, "%3.0f ", clust->features[(size_t)c * clust->dim + d]);
    }
    fprintf(pgm, "\n");
    fclose(pgm);
  }
}

//// CUDA KERNELS ////
// Three GPU kernels replace the inner loops of the serial main-loop.
// Strategy follows 14-gpu-cuda.txt (slide 46/27 "1 thread per output
// cell" pattern) so that no atomic reductions over doubles are needed.

// Compute new cluster centers. One thread per (cluster, dim)
// cell. Each thread scans assigns[] and accumulates the d-th feature
// of points in cluster c, then divides by counts[c]. NO atomics: each
// output cell is written by exactly one thread, and the inner sum
// uses i = 0..ndata-1 in the same order as serial -> bit-identical.
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
  // Plain division (NOT multiply-by-reciprocal) so that .5 ties round
  // the same way serial does under printf("%3.0f").
  centers[tid] = (counts[c] > 0) ? (sum / (double)counts[c]) : 0.0;
}

// Zero an int array on device. Used to reset counts[] each
// iteration before the assignment kernel.
__global__ void kernel_zero_ints(int *arr, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = 0;
}

// Reassign each data point to its nearest cluster. One thread
// per data point. Distance computation uses the same nested-loop order
// as serial. Each thread writes its own assigns[i] (no race). Counts
// and nchanges use integer atomicAdd which is order-independent.
__global__ void kernel_assign(int ndata, int dim, int nclust,
                              const double *data, const double *centers,
                              int *assigns, int *counts, int *nchanges) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndata) return;

  int    best_clust  = -1;
  double best_distsq = DBL_MAX;
  for (int c = 0; c < nclust; c++) {              // compare data to each cluster and assign to closest
    double distsq = 0.0;
    for (int d = 0; d < dim; d++) {                // calculate squared distance to each data dimension
      double diff = data[i * dim + d] - centers[c * dim + d];
      distsq += diff * diff;
    }
    if (distsq < best_distsq) {                   // if closer to this cluster than current best
      best_clust  = c;
      best_distsq = distsq;
    }
  }
  atomicAdd(&counts[best_clust], 1);
  if (best_clust != assigns[i]) {                 // assigning data to a different cluster?
    atomicAdd(nchanges, 1);                       // indicate cluster assignment has changed
    assigns[i] = best_clust;                      // assign to new cluster
  }
}

//// MAIN FUNCTION ////
int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: kmeans_cuda <datafile> <nclust> [savedir] [maxiter]\n");
    return 1;
  }

  const char *datafile = argv[1];
  int nclust = atoi(argv[2]);
  const char *savedir = ".";
  int MAXITER = 100;                        // bounds the iterations

  if (argc > 3) {                           // create save directory if specified
    savedir = argv[3];
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", savedir);
    system(cmd);
  }

  if (argc > 4) {
    MAXITER = atoi(argv[4]);
  }

  printf("datafile: %s\n", datafile);
  printf("nclust: %d\n", nclust);
  printf("savedir: %s\n", savedir);

  // read in the data file, allocate cluster space
  KMData  *data  = kmdata_load(datafile);
  KMClust *clust = kmclust_new(nclust, data->dim);

  printf("ndata: %d\n", data->ndata);
  printf("dim: %d\n", data->dim);
  printf("\n");

  // random, regular initial cluster assignment
  for (int i = 0; i < data->ndata; i++) {
    int c = i % clust->nclust;
    data->assigns[i] = c;
  }

  for (int c = 0; c < clust->nclust; c++) {
    int icount = data->ndata / clust->nclust;   // IMPORTANT: use integer division
    int extra = 0;
    if (c < (data->ndata % clust->nclust))
      extra = 1;                                 // extras in earlier clusters
    clust->counts[c] = icount + extra;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Device memory setup
  // Allocate mirrors of the host buffers on the GPU and ship the data + initial assigns + initial counts over once.
  // Centers and assignments live on the device for the rest of the loop and only come back at the end.
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

  // Launch parameters. One block size for all kernels, block
  // count is sized per kernel based on which axis it parallelizes over.
  const int THREADS_PER_BLOCK = 256;
  int centers_total  = clust->nclust * clust->dim;
  int blocks_centers = (centers_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocks_data    = (data->ndata    + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocks_counts  = (clust->nclust  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  //////////////////////////////////////////////////////////////////////////////
  // THE MAIN ALGORITHM
  int curiter  = 1;                              // current iteration
  int nchanges = data->ndata;                    // check for changes in cluster assignment; 0 is converged
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while (nchanges > 0 && curiter <= MAXITER) {   // loop until convergence

    // DETERMINE NEW CLUSTER CENTERS
    // One kernel fuses zero-centers + sum + divide-by-counts.
    // Reads d_counts (still holding previous iter's values, same as
    // serial which divides before zeroing counts).
    kernel_compute_centers<<<blocks_centers, THREADS_PER_BLOCK>>>(
        data->ndata, clust->dim, clust->nclust,
        d_data, d_assigns, d_counts, d_centers);

    // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
    // Reset cluster counts and nchanges before the assignment kernel.
    kernel_zero_ints<<<blocks_counts, THREADS_PER_BLOCK>>>(d_counts, clust->nclust);
    CUDA_CHECK(cudaMemset(d_nchanges, 0, sizeof(int)));

    kernel_assign<<<blocks_data, THREADS_PER_BLOCK>>>(
        data->ndata, clust->dim, clust->nclust,
        d_data, d_centers, d_assigns, d_counts, d_nchanges);

    // Pull back nchanges (for the convergence check) and counts
    // (for the iteration print line). Centers and assigns stay on the
    // device until after the loop.
    CUDA_CHECK(cudaMemcpy(&nchanges,     d_nchanges, sizeof(int),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clust->counts, d_counts,   counts_bytes, cudaMemcpyDeviceToHost));

    // Print iteration information at the end of the iter
    printf("%3d: %5d |", curiter, nchanges);
    for (int c = 0; c < nclust; c++)
      printf(" %4d", clust->counts[c]);
    printf("\n");
    curiter++;
  }

  // Loop has converged
  if (curiter > MAXITER)
    printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
  else
    printf("CONVERGED: after %d iterations\n", curiter);
  printf("\n");

  // Bring final centers and assignments back for the PGM files
  // and confusion matrix. Then release device memory.
  CUDA_CHECK(cudaMemcpy(clust->features, d_centers, centers_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(data->assigns,   d_assigns, assigns_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_data);
  cudaFree(d_centers);
  cudaFree(d_assigns);
  cudaFree(d_counts);
  cudaFree(d_nchanges);

  //////////////////////////////////////////////////////////////////////////////
  // CLEANUP + OUTPUT

  // CONFUSION MATRIX
  int **confusion = (int **)malloc(sizeof(int *) * data->nlabels); // confusion matrix: labels * clusters big
  if (!confusion) { perror("malloc"); exit(1); }
  for (int i = 0; i < data->nlabels; i++) {
    confusion[i] = (int *)calloc(nclust, sizeof(int));
    if (!confusion[i]) { perror("calloc"); exit(1); }
  }

  for (int i = 0; i < data->ndata; i++)          // count which labels in which clusters
    confusion[data->labels[i]][data->assigns[i]] += 1;

  printf("==CONFUSION MATRIX + COUNTS==\n");
  printf("LABEL \\ CLUST\n");

  printf("%2s ", "");                             // confusion matrix header
  for (int j = 0; j < clust->nclust; j++)
    printf(" %4d", j);
  printf(" %4s\n", "TOT");

  for (int i = 0; i < data->nlabels; i++) {       // each row of confusion matrix
    printf("%2d:", i);
    int tot = 0;
    for (int j = 0; j < clust->nclust; j++) {
      printf(" %4d", confusion[i][j]);
      tot += confusion[i][j];
    }
    printf(" %4d\n", tot);
  }

  printf("TOT");                                  // final total row of confusion matrix
  int tot = 0;
  for (int c = 0; c < clust->nclust; c++) {
    printf(" %4d", clust->counts[c]);
    tot += clust->counts[c];
  }
  printf(" %4d\n", tot);
  printf("\n");

  // LABEL FILE OUTPUT
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

  // SAVE PGM FILES CONDITIONALLY
  save_pgm_files(clust, savedir);

  for (int i = 0; i < data->nlabels; i++)
    free(confusion[i]);
  free(confusion);
  kmclust_free(clust);
  kmdata_free(data);

  return 0;
}
/// END MAIN ///
