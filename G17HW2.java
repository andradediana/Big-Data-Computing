import org.apache.commons.lang3.tuple.Triple;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import javax.xml.bind.SchemaOutputResolver;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class G17HW2 {
    public static JavaPairRDD<Vector, Integer> CreateDatasets(JavaRDD<String> rdd){
        return rdd.mapToPair(line -> {
            String[] parts = line.split(",");
            double[] coordinates = new double[parts.length-1];
            for (int i=0; i<parts.length-1; i++){
                coordinates[i] = Double.parseDouble(parts[i]);
            }
            int group;
            if (parts[parts.length-1].equals("A")){
                group = 0;
            } else {
                group = 1;
            }
            return new Tuple2<Vector,Integer>(Vectors.dense(coordinates),group);
        });
    };
    public static Vector[] Kmeans_C(JavaPairRDD<Vector,Integer> train, int K, int M) {
        JavaRDD<Vector> points = train.map(point -> {
            return point._1; //returns the first element in the tuple, which it's the coordinates
        });
        KMeansModel model = KMeans.train(points.rdd(), K, M);
        //Collect centroids in a list of Vectors
        Vector[] C = new Vector[K];
        for (int i = 0; i < K; i++) C[i] = model.clusterCenters()[i];
        return C;
    }
    public static JavaRDD<Vector> CalculateDistances(JavaPairRDD<Vector,Integer> U, Vector[] C){
        JavaRDD<Vector> DCi = U.map(point -> {
            int My_C = 0;
            double group = point._2;
            Vector coords = point._1;
            double min = Vectors.sqdist(coords, C[0]);
            for (int i = 0; i < C.length; i++){
                double dist = Vectors.sqdist(coords, C[i]);
                if (dist < min){
                    min = dist;
                    My_C = i;
                }
            }
            return Vectors.dense(min, My_C, group);
        });
        return DCi;
    }
    public static double MRComputeFairObjective(JavaRDD<Vector> DCi){
        JavaRDD<Double> A_Sqdist = DCi.filter(line -> line.apply(2) == 0).map(point -> {
            double dist = point.apply(0);
            return dist;
        });
        JavaRDD<Double> B_Sqdist = DCi.filter(line -> line.apply(2)==1).map(point -> {
            double dist = point.apply(0);
            return dist;
        });
        double sumA = A_Sqdist.reduce((a,b) -> a+b);
        double sumB = B_Sqdist.reduce((a,b) -> a+b);
        double objA = sumA / A_Sqdist.count();
        double objB = sumB / B_Sqdist.count();
        double obj_fair_val = Math.max(objA, objB);

        return obj_fair_val;
    }
    public static Vector[] MRFairLloyd(JavaPairRDD<Vector,Integer> U, int K, int M) {
        JavaRDD<Vector> vector = U.map(p -> p._1); //get a vector of our points, taking out the 1st tuple.
        KMeansModel model = KMeans.train(vector.rdd(), K, 0); // here we do step1: set {c1,c2,…,ck} of k centroids
        Vector[] centroids = model.clusterCenters(); //get the coordinates of our centroids

        for (int i = 0; i < M; i++) {
            Vector[] finalCentroids = centroids;

            //map points to ((clusterIdx, label), (vector, 1))
            JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Vector, Integer>> pairRDD = U.mapToPair(p -> {
                Vector coords = p._1;
                int label = p._2;
                int idx = idx_close(coords, finalCentroids);
                return new Tuple2<>(new Tuple2<>(idx, label), new Tuple2<>(coords, 1));
            });

            //sum and count per idx plus label
            JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Vector, Integer>> agg = pairRDD.aggregateByKey(
                    new Tuple2<>(Vectors.zeros(finalCentroids[0].size()), 0),
                    (acc, val) -> {
                        double[] accArr = acc._1.toArray();
                        double[] valArr = val._1.toArray();
                        for (int d = 0; d < accArr.length; d++) accArr[d] += valArr[d];
                        return new Tuple2<>(Vectors.dense(accArr), acc._2 + val._2);
                    },
                    (acc1, acc2) -> {
                        double[] arr1 = acc1._1.toArray();
                        double[] arr2 = acc2._1.toArray();
                        for (int d = 0; d < arr1.length; d++) arr1[d] += arr2[d];
                        return new Tuple2<>(Vectors.dense(arr1), acc1._2 + acc2._2);
                    }
            );

            //compute means
            JavaPairRDD<Tuple2<Integer, Integer>, Vector> means = agg
                    .filter(t -> t._2._2 > 0)
                    .mapValues(sumCount -> {
                        double[] arr = sumCount._1.toArray();
                        for (int d = 0; d < arr.length; d++) arr[d] /= sumCount._2;
                        return Vectors.dense(arr);
                    });

            //create Rdds to contain A and B
            JavaPairRDD<Integer, Vector> A = means
                    .filter(t -> t._1._2 == 0)
                    .mapToPair(t -> new Tuple2<>(t._1._1, t._2)).cache();

            JavaPairRDD<Integer, Vector> B = means
                    .filter(t -> t._1._2 == 1)
                    .mapToPair(t -> new Tuple2<>(t._1._1, t._2)).cache();

            //centroids for A and B
            centroids = CentroidSelection(A, B, K);
        }

        return centroids;
    }

    //this function gets the index of the centroid which is closest to coordinates, calculating the sqdist
        public static int idx_close(Vector coordinates, Vector[] centroids) {
            double min = Double.MAX_VALUE;
            int idx = -1;
            for (int i = 0; i < centroids.length; i++) {
                double dist = Vectors.sqdist(coordinates, centroids[i]);
                if (dist < min) {
                    min = dist;
                    idx = i;
                }
            }
            return idx;
        }
    // Function to compute alpha, beta, MA, MB, and L given groups A, B and number of clusters k for CentroidSelection
    //we use result that helps us bring multiple results of the same function.
    public static Result computeVectors(JavaPairRDD<Integer, Vector> A, JavaPairRDD<Integer, Vector> B, int K) {
        // Calculate total number of points in group A and group B, since A and B contain all the points belonging to
        //each group regardless the cluster, we just use the function count() on A and B directly. (clusterID, Vector)
        long total_A = A.count();
        long total_B = B.count();
        System.out.println(total_A);

        double[] alpha = new double[K];; //initialize alpha to have K elements
        double[] beta = new double[K]; //initialize beta
        Vector[] MA = new Vector[K]; //initialize MA
        Vector[] MB = new Vector[K]; //initialize MB
        double[] L = new double[K]; //initialize l

        // loop within the clusters
        for (int i = 0; i < K; i++) {
            Integer index = (Integer) i;
            JavaPairRDD<Integer, Vector> A_Ci = A.filter(line -> line._1().equals(index)); //filters A by clusters
            JavaPairRDD<Integer, Vector> B_Ci = B.filter(line -> line._1().equals(index)); //filters B by clusters
            long sizeA = A_Ci.count(); //Counts number of elements per cluster in A
            long sizeB = B_Ci.count(); //Counts number of elements per cluster in B

            alpha[i] = (double) sizeA / total_A; //compute alpha
            beta[i] = (double) sizeB / total_B; //compute beta

            //Compute vector averages for A using function below that calculates the averages per cluster
            MA[i] = meanVector(A_Ci);

            //Compute vector averages for B using function below that calculates the averages per cluster
            MB[i] = meanVector(B_Ci);

            // If MA[i] is null (no points in A for cluster i) -> MA[i] = MB[i]
            if (MA[i] == null && MB[i] != null) {
                MA[i] = MB[i];
            }

            // If MB[i] is null (no points in B for cluster i) -> MB[i] = MA[i]
            if (MB[i] == null && MA[i] != null) {
                MB[i] = MA[i];
            }

            // Euclidean distance to complete L
            if (MA[i] == null || MB[i] == null) {
                L[i] = 0; // No distance if centroids undefined
            } else {
                L[i] = Math.sqrt(Vectors.sqdist(MA[i], MB[i]));
            }
        }
        // Return all vectors need for function Centroid Selection
        return new Result(alpha, beta, MA, MB, L);
    }

    //function to compute mean vector of a list of vectors, calculates MA and MB
    private static Vector meanVector(JavaPairRDD<Integer, Vector> AB_Ci) {
        JavaRDD<Vector> points = AB_Ci.map(point -> point._2);
        if (points.isEmpty()) return null;

        // Use reduce to sum the vectors
        Vector sumVector = points.reduce((v1, v2) -> {
            double[] arr1 = v1.toArray();
            double[] arr2 = v2.toArray();
            double[] sum = IntStream.range(0, arr1.length)
                    .mapToDouble(i -> arr1[i] + arr2[i])
                    .toArray();
            return Vectors.dense(sum);
        });

        //Get the number of points.
        long count = points.count();
        // Divide the sum by the number of points to get the mean
        double[] sumArr = sumVector.toArray();
        for(int i = 0; i< sumArr.length; i++){
            sumArr[i] /= count;
        }
        return Vectors.dense(sumArr);
    }

    //we use result that helps us bring multiple results of the same function. In this case we bring the results
    //of function computeVectors, so we can implement them in CentroidSelection
    public static class Result {public double[] alpha;public double[] beta;public Vector[] MA;public Vector[] MB;public double[] L;

        public Result(double[] alpha, double[] beta, Vector[] MA, Vector[] MB, double[] L) {
            this.alpha = alpha;
            this.beta = beta;
            this.MA = MA;
            this.MB = MB;
            this.L = L;
        }
    }

    //Centroid Selection that receives vector A,B, k clusters and moreover receives the computed vectors plus
    //the function that computes Vector x.
    public static Vector[] CentroidSelection(JavaPairRDD<Integer, Vector> A, JavaPairRDD<Integer, Vector> B, int K) {
        //Step 1 compute vectors
        Result r = computeVectors(A, B, K);
        double[] alpha = r.alpha;
        double[] beta = r.beta;
        Vector[] MA = r.MA;
        Vector[] MB = r.MB;
        double[] L = r.L;

        //step 2 calculate first denominator in fixedA and fixedB
        int total_A = (int) A.count();
        int total_B = (int) B.count();

        //Step 2 calculate first numerator delta(A, MA) and delta(B, MB)

        double deltaA = 0;
        double deltaB = 0;

        for (int i = 0; i < K; i++) {
            Integer index = (Integer) i; //filters by cluster and get the coordinates of each point
            JavaRDD<Vector> coordsA = A.filter(line -> line._1().equals(index))
                    .map(point -> point._2()).cache();
            JavaRDD<Vector> coordsB = B.filter(line -> line._1().equals(index))
                    .map(point -> point._2()).cache();
            double clusterDeltaA = coordsA.map(point -> Vectors.sqdist(point, MA[index]))
                    .reduce((a, b) -> a + b);
            double clusterDeltaB = coordsB.map(point -> Vectors.sqdist(point, MB[index]))
                    .reduce((a, b) -> a + b);
            deltaA += clusterDeltaA;
            deltaB += clusterDeltaB;
        }

        //Step 2 calculate fixedA and fixedB
        double fixedA = deltaA / total_A;
        double fixedB = deltaB / total_B;

        //Step 3 obtain vector x from function
        Vector X = computeVectorX(fixedA, fixedB, alpha, beta, L, K);

        //Step 4 obtain centroids
        Vector[] centroids = new Vector[K];
        for (int i = 0; i < K; i++) {
            if (L[i] == 0) {
                centroids[i] = MA[i];
            } else {
                double Xi = X.apply(i);
                double li = L[i];
                double coefA = (li - Xi) / li;
                double coefB = Xi / li;

                double[] MAarr = MA[i].toArray();
                double[] MBarr = MB[i].toArray();
                double[] cArr = new double[MA[i].size()];

                for (int d = 0; d < MA[i].size(); d++) {
                    cArr[d] = coefA * MAarr[d] + coefB * MBarr[d];
                }
                centroids[i] = Vectors.dense(cArr);
            }
        }
        return centroids;
    }
    public static Vector computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t=1; t<=T; t++){
            fA = fixedA;
            fB = fixedB;
            power = power/2;
            for (int i=0; i<K; i++) {
                double temp = (1-gamma)*beta[i]*ell[i]/(gamma*alpha[i]+(1-gamma)*beta[i]);
                xDist[i]=temp;
                fA += alpha[i]*temp*temp;
                temp=(ell[i]-temp);
                fB += beta[i]*temp*temp;
            }
            if (fA == fB) {break;}
            gamma = (fA > fB) ? gamma+power : gamma-power;
        }
        Vector X = Vectors.dense(xDist);
        return X;
    }
    public static void main(String[] args){

        //Checking the arguments received through the command line
        //if (args.length != 4) {
        //throw new IllegalArgumentException("USAGE: file_path L K M");
        //}

        //Store arguments into variables from the command line
        String inputFile = args[0];
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);

        //String inputFile = "artificial1M7D100K.txt";
        //int L = 16;
        //int K = 100;
        //int M = 10;

        //We start setting up Spark environment
        SparkConf conf = new SparkConf()
                .setAppName("KMeansClustering")
                .setMaster("local[*]")
                .set("spark.driver.host", "localhost");

        // Initialize JavaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);
        // Read the data into a string RDD, and subdivides it into L random partitions
        //Cache is keeping the RDD in memory after the first computation
        JavaRDD<String> inputPoints = sc.textFile(inputFile).repartition(L).cache();

        //Calculates the number of total points based on the number of lines in inputPoints
        long N = inputPoints.count();
        //Filters the lines that end with A and counts the number of points A
        JavaRDD<String> NArdd = inputPoints.filter(line -> line.endsWith("A"));
        long NA = NArdd.count();
        //Computes the number of points B, substracting NA from N
        int NB = (int)(N-NA);

        //Vector Pair RDD U is created, which contains a tuple with two elements, the first one is the
        //vector with the coordinates and the second one an integer with the groups represented with zeros
        //and ones
        JavaPairRDD<Vector, Integer> U = CreateDatasets(inputPoints).cache();

        //COMPUTATION OF STD CENTERS
        //U is used to create set C of K centroids using the function Kmeans
        U.count();
        long startStd = System.nanoTime();
        Vector[] C = Kmeans_C(U, K, M);
        long endStd = System.nanoTime();

        //This function creates an RDD with the distances between the point and the closest cluster
        //It saves the data with the next structure (DCi, C, Group), where DCi corresponds to the distance
        //C to the cluster, and group to the group A or B to which the point belong
        long startObjStd = System.nanoTime();
        JavaRDD<Vector> DCi = CalculateDistances(U, C).cache();
        //In this way we will use just one RDD to compute the Std Objective and Fair Objetive function
        //We'll filter the RDD based on groups for the Fair function
        //We saved our results in these two variables to use them later during the printing
        double Obj_std_val = MRComputeFairObjective(DCi);
        long endObjStd = System.nanoTime();

        //COMPUTATION OF FAIR CENTERS
        //Computes fair centroids
        //U.count();
        long startFair = System.nanoTime();
        Vector[] C_fair = MRFairLloyd(U, K, M);
        long endFair = System.nanoTime();

        //Calculates min distance
        long startObjFair = System.nanoTime();
        JavaRDD<Vector> DCi_fair = CalculateDistances(U, C_fair);
        //Computes Fair Objective Function using fair centroids
        double Obj_fair_std_val = MRComputeFairObjective(DCi_fair);
        long endObjFair = System.nanoTime();

        //Prints the outputs in the requested format
        System.out.println("Input file path = " + inputFile + ", L = " + L + ", K = " + K + ", M = " + M);
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);
        System.out.println("Fair Objective with Standard Centers = " + Obj_std_val);
        System.out.println("Fair Objective with Fair Centers = " + Obj_fair_std_val);
        System.out.println("Time to compute Standard Centers = " + ((endStd - startStd) / 1_000_000.0) + " ms");
        System.out.println("Time to compute Fair Centers = " + ((endFair - startFair) / 1_000_000.0) + " ms");
        System.out.println("Time to compute Objective with Standard Centers = " + ((endObjStd - startObjStd) / 1_000_000.0) + " ms");
        System.out.println("Time to compute Objective with Fair Centers = " + ((endObjFair - startObjFair) / 1_000_000.0) + " ms");

        sc.close();
    }
}
