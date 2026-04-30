// usage: kmeans <datafile> <nclust> [savedir] [maxiter]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Data set to be clustered
typedef struct {
  int    ndata;       // count of data
  int    dim;         // dimension of features for data
  double **features;  // pointers to individual features
  int    *assigns;    // cluster to which data is assigned
  int    *labels;     // label for data if available
  int    nlabels;     // max value of labels +1, number 0,1,...,nlabel0
} KMData;

// Cluster information
typedef struct {
  int    nclust;      // number of clusters, the "k" in kmeans
  int    dim;         // dimension of features for data
  double **features;  // 2D indexing for individual cluster center features
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

  KMData *data = malloc(sizeof(KMData));
  if (!data) { perror("malloc"); exit(1); }

  data->ndata = ndata;
  data->dim   = dim;

  data->features = malloc(sizeof(double *) * ndata);
  data->labels   = malloc(sizeof(int) * ndata);
  data->assigns  = malloc(sizeof(int) * ndata);
  if (!data->features || !data->labels || !data->assigns) {
    perror("malloc"); exit(1);
  }
  for (int i = 0; i < ndata; i++) {
    data->features[i] = malloc(sizeof(double) * dim);
    if (!data->features[i]) { perror("malloc"); exit(1); }
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
        data->features[idx][d] = atof(tok);
      } else {
        data->features[idx][d] = 0.0;
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
  for (int i = 0; i < data->ndata; i++)
    free(data->features[i]);
  free(data->features);
  free(data->assigns);
  free(data->labels);
  free(data);
}

// Allocate space for clusters in an object
KMClust *kmclust_new(int nclust, int dim) {
  KMClust *clust = malloc(sizeof(KMClust));
  if (!clust) { perror("malloc"); exit(1); }

  clust->nclust = nclust;
  clust->dim    = dim;

  clust->features = malloc(sizeof(double *) * nclust);
  clust->counts   = malloc(sizeof(int) * nclust);
  if (!clust->features || !clust->counts) {
    perror("malloc"); exit(1);
  }

  for (int c = 0; c < nclust; c++) {
    clust->features[c] = calloc(dim, sizeof(double));
    if (!clust->features[c]) { perror("calloc"); exit(1); }
    clust->counts[c] = 0;
  }

  return clust;
}

void kmclust_free(KMClust *clust) {
  if (!clust) return;
  for (int c = 0; c < clust->nclust; c++)
    free(clust->features[c]);
  free(clust->features);
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
      if (clust->features[c][d] > maxfeat)
        maxfeat = clust->features[c][d];
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
      fprintf(pgm, "%3.0f ", clust->features[c][d]);
    }
    fprintf(pgm, "\n");
    fclose(pgm);
  }
}

//// MAIN FUNCTION ////
int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: kmeans <datafile> <nclust> [savedir] [maxiter]\n");
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
  double *diffs = malloc(sizeof(double) * data->dim);
  if (!diffs) { perror("malloc"); exit(1); }

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
  // THE MAIN ALGORITHM
  int curiter  = 1;                              // current iteration
  int nchanges = data->ndata;                    // check for changes in cluster assignment; 0 is converged
  printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
  printf("ITER NCHANGE CLUST_COUNTS\n");

  while (nchanges > 0 && curiter <= MAXITER) {   // loop until convergence

    // DETERMINE NEW CLUSTER CENTERS
    for (int c = 0; c < clust->nclust; c++)     // reset cluster centers to 0.0
      for (int d = 0; d < clust->dim; d++)
        clust->features[c][d] = 0.0;

    for (int i = 0; i < data->ndata; i++) {      // sum each data row into its assigned center
      int c = data->assigns[i];
      cblas_daxpy(clust->dim, 1.0, data->features[i], 1, clust->features[c], 1);
    }

    for (int c = 0; c < clust->nclust; c++) {    // scale sums into means
      if (clust->counts[c] > 0)
        cblas_dscal(clust->dim, 1.0 / clust->counts[c], clust->features[c], 1);
    }

    // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
    for (int c = 0; c < clust->nclust; c++)      // reset cluster counts to 0
      clust->counts[c] = 0;                     // re-init here to support first iteration

    nchanges = 0;
    for (int i = 0; i < data->ndata; i++) {       // iterate over all data
      int    best_clust  = -1;
      double best_distsq = DBL_MAX;
      for (int c = 0; c < clust->nclust; c++) {   // compare data to each cluster and assign to closest
        cblas_dcopy(clust->dim, data->features[i], 1, diffs, 1);
        cblas_daxpy(clust->dim, -1.0, clust->features[c], 1, diffs, 1);
        double distsq = cblas_ddot(clust->dim, diffs, 1, diffs, 1);
        if (distsq < best_distsq) {               // if closer to this cluster than current best
          best_clust  = c;
          best_distsq = distsq;
        }
      }
      clust->counts[best_clust] += 1;
      if (best_clust != data->assigns[i]) {       // assigning data to a different cluster?
        nchanges += 1;                             // indicate cluster assignment has changed
        data->assigns[i] = best_clust;             // assign to new cluster
      }
    }

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

  //////////////////////////////////////////////////////////////////////////////
  // CLEANUP + OUTPUT

  // CONFUSION MATRIX
  int **confusion = malloc(sizeof(int *) * data->nlabels); // confusion matrix: labels * clusters big
  if (!confusion) { perror("malloc"); exit(1); }
  for (int i = 0; i < data->nlabels; i++) {
    confusion[i] = calloc(nclust, sizeof(int));
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
  free(diffs);
  kmclust_free(clust);
  kmdata_free(data);

  return 0;
}
/// END MAIN ///
