# Big Data Computing Homeworks (Group 17)

Three homeworks for the **Big Data Computing** course, University of Padova (A.Y. 2024/25), implemented in **Java with Apache Spark**:

1. **HW1:** standard vs fair clustering objectives and cluster statistics in MapReduce.
2. **HW2:** a MapReduce implementation of **Fair Lloyd** (fair k-means), compared with Spark's standard k-means, plus a synthetic data generator.
3. **HW3:** **Count-Min Sketch** and **Count Sketch** on a data stream, with heavy-hitter accuracy experiments.

**Authors (Group 17):** Diana Cristina Andrade Damian, Diego Alonso Brule Galleguillos, Nicolas Matias Tamara

---

## Repository contents

| File | Description |
|---|---|
| `G17HW1.java` | HW1: k-means centroids, standard objective Δ(U,C), fair objective Φ(A,B,C) and per-cluster group counts |
| `G17HW1analysis.docx` | HW1: analysis of the local space (M_L) used by `MRPrintStatistics` |
| `G17HW2.java` | HW2: `MRFairLloyd`, centroid selection, and timing of the standard and fair algorithms |
| `G17GEN.java` | HW2: generator of a 2-D dataset with demographic imbalance inside each cluster |
| `G17HW2form.docx` | HW2: scalability tests (running times) and description of the generator |
| `G17HW3.java` | HW3: streaming program with exact counts, Count-Min Sketch and Count Sketch |
| `G17HW3table.docx` | HW3: accuracy tests for varying K and W |

The course data files (`artificial1M7D100K.txt`, `artificial4M7D100K.txt`) and the HW3 stream server belong to the course and are **not** included.

## Input data format

Each line is a point with comma-separated coordinates followed by a demographic group label, `A` or `B`:

```
x,y,A
x,y,B
```

- HW1 expects 2-D points (`x,y,group`).
- HW2 accepts any dimension (`x1,...,xd,group`). The course test files are 7-D.
- `G17GEN.java` writes 2-D data in this same format.

---

## HW1: Standard and fair k-means objectives

Clusters the points with Spark MLlib's `KMeans` and evaluates the result with MapReduce-style computations.

**Usage:** `G17HW1 <file_path> <L> <K> <M>`  (L = number of partitions, K = number of clusters, M = number of k-means iterations)

**Design:**
- The input is parsed once into a single `RDD<Vector>` with structure `(x, y, group)`, where group A is mapped to 0 and group B to 1.
- A second RDD `DCi` holds `(squared distance to closest centroid, centroid index, group)`. Everything else is computed from it with filters.
- Standard objective: Δ(U,C) is the mean squared distance of all points to their closest centroid.
- Fair objective: Φ(A,B,C) is the **maximum** of the mean squared distances of group A and group B.
- `MRPrintStatistics` returns, for each centroid, the number of points of group A and B. It returns a list of triples so the program can print a clean output after the Spark logs.

**Output:** input parameters, `N`, `NA`, `NB`, `Delta(U, C)`, `Phi(A, B, C)`, and one line per centroid with its coordinates and `NA_i`, `NB_i`.

The local-space analysis is in `G17HW1analysis.docx`. The map steps work point by point, and the counting phase is bounded by O(N/L) because the data is split into L partitions.

---

## HW2: Fair Lloyd in MapReduce

Implements `MRFairLloyd`, a fair variant of Lloyd's algorithm. It tries to minimise the fair objective Φ so that neither group is much worse served than the other.

**Usage:** `G17HW2 <file_path> <L> <K> <M>`

**How it works:**
1. Initial centroids come from Spark's `KMeans` (0 iterations).
2. Each of the M iterations assigns every point to its closest centroid and aggregates, per `(cluster, group)`, the sum of coordinates and point count (`aggregateByKey`). From this it obtains the group means `MA_i` and `MB_i`.
3. `computeVectors` derives `α_i`, `β_i` (the share of each group's points in cluster i), the group means and the distances `ℓ_i = ||MA_i − MB_i||`.
4. `CentroidSelection` computes the within-group costs and calls `computeVectorX`, which finds the offsets `x_i` along each segment `MA_i → MB_i` with a binary-search-style update of a weight γ (10 steps). The new centroid is `((ℓ_i − x_i)/ℓ_i)·MA_i + (x_i/ℓ_i)·MB_i`.
5. If a cluster has points from only one group, its centroid is simply that group's mean.

**Output:** parameters and group sizes, the fair objective for standard centers and for fair centers, and running times for computing each set of centers and each objective (data loading is excluded from timings, and RDDs are cached).

### Scalability tests (time in ms, L=16, K=100, M=10)

`artificial1M7D100K.txt`

| Executors | Spark Lloyd | MRFairLloyd | Objective (std centers) | Objective (fair centers) |
|---|---|---|---|---|
| 2 | 3,895.97 | 214,699.12 | 1,743.49 | 1,119.57 |
| 4 | 3,551.87 | 208,289.01 | 1,687.85 | 1,297.95 |
| 8 | 3,736.27 | 304,922.26 | 1,734.71 | 1,454.75 |
| 16 | 3,822.51 | 229,623.10 | 1,669.41 | 1,250.38 |

`artificial4M7D100K.txt`

| Executors | Spark Lloyd | MRFairLloyd | Objective (std centers) | Objective (fair centers) |
|---|---|---|---|---|
| 2 | 19,259.16 | 254,484.69 | 6,052.67 | 5,120.89 |
| 4 | 13,650.59 | 224,012.03 | 5,739.10 | 4,787.76 |
| 8 | 13,384.52 | 218,944.65 | 5,839.91 | 4,794.50 |
| 16 | 13,531.47 | 216,991.88 | 6,079.29 | 4,599.49 |

**Observations:**
- Fair Lloyd is far slower than Spark's standard k-means, roughly 13–80× in these runs (about 16× on the 4M file with 4 or more executors).
- Adding executors gives only a small speedup, mostly from 2 to 4 executors on the larger file, and almost none beyond that.
- A likely cause is that `computeVectors` and `CentroidSelection` loop over the K clusters and launch several Spark jobs per cluster in every iteration. With K=100 and M=10 this adds up to thousands of small jobs, which limits parallel scaling.

### `G17GEN.java`: synthetic data generator

Creates a 2-D dataset that highlights the difference between standard and fair Lloyd: every cluster is **imbalanced** with respect to the groups.

**Usage:** `java G17GEN <N> <K>`

- Draws K random cluster centers in [0,100] × [0,100].
- Splits the N points almost evenly across the clusters.
- In each cluster, 80% of the points belong to the majority group (alternating A and B between clusters) and 20% to the other group. The minority points are shifted by (+5, +5) from the center.
- Points are Gaussian around their center (σ = 3), shuffled, written to `gen_data.csv` and also printed.
- Constraint: **N must be larger than K**, and N ≫ K gives more realistic data.

Test on `G17GEN` data (N = 1000, K = 5): Spark Lloyd 4,473.3 ms, MRFairLloyd 26,608.24 ms, objective with standard centers 162.1 ms, objective with fair centers 158.75 ms. See `G17HW2form.docx`.

---

## HW3: Count-Min Sketch vs Count Sketch on a stream

Processes a stream of integers received over a socket with Spark Streaming and estimates item frequencies with two sketches, comparing them with exact counts.

**Usage:** `G17HW3 <port> <T> <D> <W> <K>`

| Argument | Meaning |
|---|---|
| `port` | Port of the stream server (`algo.dei.unipd.it`) |
| `T` | Number of stream items to process before stopping |
| `D` | Number of rows (hash functions) of each sketch |
| `W` | Number of columns of each sketch |
| `K` | Number of top heavy hitters used for the evaluation |

**Design:**
- Micro-batches of 100 ms. Processing stops once at least T items have been read, using a semaphore for a clean shutdown.
- **Exact frequencies** are kept in a hash map as ground truth.
- **Count-Min Sketch:** D×W counters, hash `((a·x + b) mod p) mod W` with p = 8191; the estimate is the **minimum** over the D counters.
- **Count Sketch:** D×W counters with an additional ±1 hash; the estimate is the **median** of the D signed counters.
- The top-K items by true frequency are extracted with a min-heap, and the **average relative error** `|estimate − true| / true` is computed for each sketch.

**Output:** number of processed and distinct items, average relative error of the top-K heavy hitters for CM and CS, and, if K ≤ 10, each heavy hitter with its true and CM-estimated frequency.

### Test 1: varying K (D=9, W=100, T=100000, port 8886)

| K | CM run 1 | CM run 2 | CM run 3 | CS run 1 | CS run 2 | CS run 3 |
|---|---|---|---|---|---|---|
| 10 | 0.02369 | 0.02359 | 0.02355 | 4.79e-4 | 4.64e-4 | 4.40e-4 |
| 20 | 77.26195 | 85.33698 | 77.36194 | 2.72525 | 2.2753 | 2.72524 |
| 50 | 141.50476 | 188.29471 | 190.76471 | 5.05009 | 5.76009 | 5.86009 |
| 100 | 161.43738 | 228.80236 | 165.91738 | 5.55005 | 6.91004 | 5.66505 |

### Test 2: varying W (D=9, K=10, T=100000, port 8886)

| W | CM run 1 | CM run 2 | CM run 3 | CS run 1 | CS run 2 | CS run 3 |
|---|---|---|---|---|---|---|
| 5 | 0.90241 | 0.89715 | 0.89954 | 0.30488 | 0.30134 | 0.30169 |
| 10 | 0.24879 | 0.25025 | 0.24921 | 0.00420 | 0.00671 | 0.00441 |
| 20 | 0.12231 | 0.12221 | 0.1223 | 0.0022 | 0.00217 | 0.00205 |
| 50 | 0.04821 | 0.04828 | 0.04833 | 0.00145 | 0.00112 | 0.0011 |

**Observations:**
- **Count Sketch is more accurate than Count-Min** in these tests, often by one or two orders of magnitude.
- **More columns reduce the error** for both sketches. From W=5 to W=50, CM drops from about 0.90 to 0.048 and CS from about 0.30 to about 0.001.
- **Larger K hurts CM much more.** With W=100 the top 10 items are estimated very well, but from K=20 the CM error jumps. Lower-frequency items in the top-K are heavily overestimated because collisions add counts from other items, and CM can only overestimate. CS errors grow too, but stay far smaller, since its signed updates let collision noise cancel out.

---

## How to run

### Requirements
- Java 8 or later
- Apache Spark with MLlib and Streaming (the course used a local `local[*]` master). Artifacts needed: `spark-core`, `spark-mllib`, `spark-streaming` and `commons-lang3`.
- HW2 imports `javax.xml.bind.SchemaOutputResolver`, which is unused. On Java 11 or later it is not available, so **delete that import** if the file does not compile.

### Compile and run
Build the Java files with your preferred tool (Maven/Gradle, or `javac` with the Spark jars on the classpath), then run with `spark-submit`:

```bash
spark-submit --class G17HW1 <your-jar> <path/to/data.csv> <L> <K> <M>
spark-submit --class G17HW2 <your-jar> <path/to/data.txt> <L> <K> <M>
spark-submit --class G17HW3 <your-jar> <port> <T> <D> <W> <K>
java G17GEN <N> <K>          # writes gen_data.csv, which can be used as input for G17HW1 / G17HW2
```

Example: `spark-submit --class G17HW2 <your-jar> gen_data.csv 16 5 10`

### Notes
- All three programs set the master to `local[*]` in the code. For cluster runs (as used in the HW2 scalability tests), remove that line and pass the master and number of executors to `spark-submit`.
- **HW3 needs the course stream server.** It connects to `algo.dei.unipd.it`, which may not be reachable from outside the university. To run it elsewhere, change the host in `socketTextStream` and feed it one integer per line, for example with `nc -lk <port>`.
