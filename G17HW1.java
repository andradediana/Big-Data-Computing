import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.commons.lang3.tuple.Triple;

import java.util.ArrayList;
import java.util.List;

public class G17HW1 {
    public static JavaRDD<Vector> CreateDatasets(JavaRDD<String> rdd){
        JavaRDD<Vector> U = rdd.map(line -> {
            String[] parts = line.split(",");
            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);
            int group;
            if (parts[2].equals("A")){
                group = 0;
            } else {
                group = 1;
            }
            return Vectors.dense(x,y,group);
        });
        return U;
    };
    public static Vector[] Kmeans_C(JavaRDD<Vector> train, int K, int M) {
        JavaRDD<Vector> points = train.map(point -> {
            double x = point.apply(0);
            double y = point.apply(1);
            return Vectors.dense(x,y);
        });
        KMeansModel model = KMeans.train(points.rdd(), K, M);
        //Collect centroids in a list of Vectors
        Vector[] C = new Vector[K];
        for (int i = 0; i < K; i++) C[i] = model.clusterCenters()[i];
        return C;
    }
    public static JavaRDD<Vector> CalculateDistances(JavaRDD<Vector> U, Vector[] C){
        JavaRDD<Vector> DCi = U.map(point -> {
            double min = 1000000000;
            int My_C = 0;
            double group = point.apply(2);
            Vector coords = Vectors.dense(point.apply(0), point.apply(1));
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
    public static double MRComputeStandardObjective(JavaRDD<Vector> DCi){
        JavaRDD<Double> dist_sq = DCi.map(point ->{
            double dist = point.apply(0);
            return dist;
        });
        double sumsq = dist_sq.reduce((a,b) -> a+b);
        double obj_val = sumsq / DCi.count();
        return obj_val;
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

    //Triple<Integer, Integer, Integer>
    public static List<Triple> MRPrintStatistics(JavaRDD<Vector> DCi, Vector[] C){
        JavaRDD<Double> A_Ci = DCi.filter(line -> line.apply(2)==0).map(point -> {
            double centroid = point.apply(1);
            return centroid;
        });
        JavaRDD<Double> B_Ci = DCi.filter(line -> line.apply(2)==1).map(point -> {
            double centroid = point.apply(1);
            return centroid;
        });

        int[] NACounts = new int[C.length];
        int[] NBCounts = new int[C.length];

        for (int i = 0; i<C.length; i++){
            Double index = (double) i;
            long Acount = A_Ci.filter(Ci -> (double) Ci == index).count();
            long Bcount = B_Ci.filter(Ci -> (double) Ci == index).count();
            NACounts[i] = (int) Acount;
            NBCounts[i] = (int) Bcount;
        }
        List<Triple> Triplets = new ArrayList<>();

        for (int i = 0; i < C.length; i++){
            Triplets.add(i, Triple.of(i, NACounts[i], NBCounts[i]));
        }
        return Triplets;
    }
    public static void main(String[] args){

        //Checking the arguments received through the command line
        if (args.length != 4) {
        throw new IllegalArgumentException("USAGE: file_path L K M");
        }

        //Store arguments into variables from the command line
        String inputFile = args[0];
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);

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

        //Vector RDD U is created, which contains all the points coordinates
        JavaRDD<Vector> U = CreateDatasets(inputPoints);
        //U is used to create set C of K centroids using the function Kmeans
        Vector[] C = Kmeans_C(U, K, M);

        //This function creates an RDD with the distances between the point and the closest cluster
        //It saves the data with the next structure (DCi, C, Group), where DCi corresponds to the distance
        //C to the cluster, and group to the group A or B to which the point belongs
        JavaRDD<Vector> DCi = CalculateDistances(U, C);

        //In this way we will use just one RDD to compute the Std Objective and Fair Objetive function
        //We'll filter the RDD based on groups for the Fair function
        //We saved our results in these two variables to use them later during the printing
        double Obj_fun_val = MRComputeStandardObjective(DCi);
        double Obj_fair_fun_val = MRComputeFairObjective(DCi);

        //We decided to return a List of Triplets instead of printing our results directly in the MRPrintStatistics
        //In this way, it was easier for us to obtain a "cleaner" output in the terminal without the Spark commands
        //interrupting our results
        List<Triple> Triplets = MRPrintStatistics(DCi,C);

        //Prints the outputs in the requested format
        System.out.println("Input file path = " + inputFile + ", L = " + L + ", K = " + K + ", M = " + M);
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);
        System.out.println("Delta(U, C) = " + Obj_fun_val);
        System.out.println("Phi(A, B, C) = " + Obj_fair_fun_val);

        //Triplets contains our results from the number of points A and B per cluster, so we need a for to go through
        //all the data
        for (int i = 0; i < Triplets.size(); i++) {
            Triple currentTriple = Triplets.get(i);
            int NACounts = (int) currentTriple.getMiddle();
            int NBCounts = (int) currentTriple.getRight();
            System.out.println("i= " + i + " center = " + C[i] + " NA" + i + " = " + NACounts + " NB" + i + " = " + NBCounts);
        }
        sc.close();
    }
}
