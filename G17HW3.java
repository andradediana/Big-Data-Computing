import java.util.Random;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;

public class G17HW3 {

    // After how many items should we stop?
    // public static final int THRESHOLD = 1000000;

    // Separate class for the Count-Min Sketch
    static class CountMinSketch{
        private int D; // Number of hash functions (rows)
        private int W; // Number of columns (counters per row)
        private int p = 8191; // A prime number for hashing, often chosen to be large
        private int[][] tableCM; // The 2D array of counters
        private int[] a; // Random coefficients for hash functions
        private int[] b; // Random coefficients for hash functions

        // Constructor for the CountMinSketch
        public CountMinSketch(int D, int W){
            this.D = D;
            this.W = W;
            this.tableCM = new int[D][W];
            this.a = new int[D];
            this.b = new int[D];

            //Creating D random numbers for a and b, these values will be used to create hash function
            //We save these values because we need to make sure all the elements are using the same hash function
            Random sharedRand = new Random(123); //Since we don't care about the order of the sequence,
            //we can use the same random generator for both variables a and b

            for(int i=0; i<D; i++){
                this.a[i] = 1 + (int)(sharedRand.nextInt()*(p-1)); //range(1, p-1)
                this.b[i] = (int)(sharedRand.nextInt()*p); //range(0, p-1)
            }
        }

        //Function that uses the previous arrays to compute the column index for every hash function
        //It returns an array with the column indexes of the current item
        private int[] hash(long item){
            int h [] = new int [D];
            for (int i=0; i<D; i++){
                long hashVal = ((long) a[i] * item + b[i]) % p;
                if (hashVal < 0) hashVal += p;  // make sure positive
                h[i] = (int)(hashVal % W);
                if (h[i] < 0) h[i] += W;
            }
            return h;
        }

        //Here, we are calling the function hash to obtain the indexes of the columns, and then we are
        //comparing the values inside these counters and increasing only the one with the min value (variant of CMS)
        public void add(long item){
            int[] col = hash(item);
            for (int i = 0; i < D; i++) {
                tableCM[i][col[i]]++;
            }
        }

        //This function reviews the counters related to the current number and returns the one with the minimum value
        public long estimate(long item) {
            long min = Integer.MAX_VALUE;
            int[] col = hash(item);
            for (int i = 0; i < D; i++) {
                min = Math.min(min, tableCM[i][col[i]]);
            }
            return min;
        }
    }

    static class CountSketch {
        private final int D; //number of hash rows
        private final int W; //number of columns
        private final int p = 8191; //prime for modular hashing
        private final int[][] tableCS; //D×W table
        private final int[] a; //for column hash h_j
        private final int[] b; //for column hash h_j
        private final int[] gA;  //hsh g_j
        private final int[] gB;    // hash g_j

        public CountSketch(int D, int W) {
            this.D = D;
            this.W = W;
            this.tableCS = new int[D][W];

            this.a = new int[D];
            this.b = new int[D];
            this.gA = new int[D];
            this.gB = new int[D];

            Random sharedRand = new Random(321);
            for (int j = 0; j < D; j++) {
                this.a[j] = 1 + sharedRand.nextInt(p - 1);
                this.b[j] = sharedRand.nextInt(p);


                this.gA[j] = 1 + sharedRand.nextInt(p - 1);
                this.gB[j] = sharedRand.nextInt(p);
            }
        }

        private int[] hash(long item) {
            int[] col = new int[D];
            for (int j = 0; j < D; j++) {
                long v = ((long)a[j] * item + b[j]) % p;
                if (v < 0) v += p;
                col[j] = (int)(v % W);
            }
            return col;
        }

        private int[] sign(long item) {
            int[] signs = new int[D];
            for (int j = 0; j < D; j++) {
                long hashValue = ((long) gA[j] * item + gB[j]) % p;
                if (hashValue < 0) {
                    hashValue = hashValue + p;
                }
                if (hashValue % 2 == 0) {
                    signs[j] = +1;
                } else {
                    signs[j] = -1;
                }
            }
            return signs;
        }

        public void add(long item) {
            int[] col = hash(item);
            int[] s   = sign(item);
            for (int j = 0; j < D; j++) {
                tableCS[j][ col[j] ] += s[j];
            }
        }

        public long estimate(long item) {
            int[] col = hash(item);
            int[] s   = sign(item);
            long[] estimate = new long[D];
            for (int j = 0; j < D; j++) {
                estimate[j] = s[j] * tableCS[j][ col[j] ];
            }
            Arrays.sort(estimate);
            return estimate[D/2];  //return median
        }

    }
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            throw new IllegalArgumentException("USAGE: port, threshold, D, W, K");
        }
        // IMPORTANT: the master must be set to "local[*]" or "local[n]" with n > 1, otherwise
        // there will be no processor running the streaming computation and your
        // code will crash with an out of memory (because the input keeps accumulating).
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]") // remove this line if running on the cluster
                .setAppName("DistinctExample");

        // The definition of the streaming spark context  below, specifies the amount of
        // time used for collecting a batch, hence giving some control on the batch size.
        // Beware that the data generator we are using is very fast, so the suggestion is to
        // use batches of less than a second, otherwise you might exhaust the JVM memory.
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        // TECHNICAL DETAIL:
        // The streaming spark context and our code and the tasks that are spawned all
        // work concurrently. To ensure a clean shut down we use this semaphore. The 
        // main thread will first acquire the only permit available, and then it will try
        // to acquire another one right after spinning up the streaming computation.
        // The second attempt at acquiring the semaphore will make the main thread
        // wait on the call. Then, in the `foreachRDD` call, when the stopping condition
        // is met the semaphore is released, basically giving "green light" to the main
        // thread to shut down the computation. We cannot call `sc.stop()` directly in `foreachRDD`
        // because it might lead to deadlocks.

        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        int portExp = Integer.parseInt(args[0]);

        int THRESHOLD = Integer.parseInt(args[1]);
        //System.out.println("Threshold = " + THRESHOLD);
        int D = Integer.parseInt(args[2]);
        //System.out.println("D = " + D); //number of rows or hash functions
        int W = Integer.parseInt(args[3]);
        //System.out.println("W = " + W); //number of columns
        int K = Integer.parseInt(args[4]);
        //System.out.println("K = " + K); //number of top elements
        //System.out.printf("Port = %10d T = %10d D = %10d W = %10d K = %10d", portExp, THRESHOLD, D, W, K);


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Variable streamLength below is used to maintain the number of processed stream items.
        // It must be defined as a 1-element array so that the value stored into the array can be
        // changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
        // long would not work since the lambda would not be allowed to update it.
        long[] streamLength = new long[1]; // Stream length (an array to be passed by reference)
        streamLength[0]=0L;
        //HashMap<Long, Long> histogram = new HashMap<>(); // Hash Table for the distinct elements
        Map<Long, Long> frequencyMap = new HashMap<>(); //Tracks real frequencies per ID
        Map<Long, Long> frequencyCM = new HashMap<>(); //Tracks estimated frequencies per ID using CountMinSketch
        Map<Long, Long> frequencyCS = new HashMap<>(); //Tracks estimated frequencies per ID using CountSketch

        //Next line is calling the CountMinSketch structure to create a CMS data structure of dimensions DXW
        CountMinSketch cms = new CountMinSketch(D,W);
        CountSketch cs = new CountSketch(D,W);

        // CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                // For each batch, to the following.
                // BEWARE: the `foreachRDD` method has "at least once semantics", meaning
                // that the same data might be processed multiple times in case of failure.
                .foreachRDD((batch, time) -> {
                    // this is working on the batch at time `time`.
                    if (streamLength[0] < THRESHOLD) {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;
                        if (batchSize > 0) {
                            System.out.println("Batch size at time [" + time + "] is: " + batchSize);

                            // Convert the RDD of strings to an RDD of Longs
                            //In this way we can access to the input in order to use it with CM and CS algorithms
                            JavaRDD<Long> longBatchRDD = batch.map(s -> Long.parseLong(s));

                            // Extract the distinct items from the batch to compute real frequencies
                            Map<Long, Long> batchItemsRealFeq = longBatchRDD
                                    .mapToPair(s -> new Tuple2<>(s, 1L)) //Converts to (key, value) pair
                                    .reduceByKey(Long::sum) //frequency per batch
                                    .collectAsMap(); //brings that map to the driver

                            //Now, the frequencies will be estimated using the CountMinSketch algorithm

                            List<Long> items = longBatchRDD.collect();
                            for (Long item : items) {
                                cms.add(item);
                                cs.add(item);
                            }

                            // Update the streaming state. If the overall count of processed items reaches the
                            // THRESHOLD value (among all batches processed so far), subsequent items of the
                            // current batch are ignored, and no further batches will be processed
                            //It updates the streaming state for frequencyMap, we need to do this for every batch
                            //in order not to lose the current computations
                            for (Map.Entry<Long, Long> pair : batchItemsRealFeq.entrySet()) {
                                Long id = pair.getKey(); //gets the current number
                                Long count = pair.getValue(); //gets the frequency of this number

                                //Updates the frequencies tables
                                //It returns a 0 as default value for count when the element doesnt exist yet
                                frequencyMap.put(id, frequencyMap.getOrDefault(id, 0L)+count); //Real freq
                            }


                            // If we wanted, here we could run some additional code on the global histogram
                            if (streamLength[0] >= THRESHOLD) {
                                // Stop receiving and processing further batches
                                stoppingSemaphore.release();
                            }

                        }
                    }
                });

        // MANAGING STREAMING SPARK CONTEXT
        System.out.println("Starting streaming engine");
        sc.start();
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire();
        System.out.println("Stopping the streaming engine");

        /* The following command stops the execution of the stream. The first boolean, if true, also
           stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
           the processing of all received data to be completed. You might get some error messages when
           the program ends, but they will not affect the correctness. You may also try to set the second
           parameter to true.
        */

        sc.stop(false, false);
        System.out.println("Streaming engine stopped");

        //Estimate values for each key using the CMS, we do it outside of the batch loop to ensure we are not
        //miscalculating the frequencies, since the CMS accumulates the frequencies in the counters overall batches
        for (Map.Entry<Long, Long> entry : frequencyMap.entrySet()) {
            frequencyCM.put(entry.getKey(), (long) cms.estimate(entry.getKey())); //Real freq
        }

        for (Map.Entry<Long, Long> entry : frequencyMap.entrySet()) {
            frequencyCS.put(entry.getKey(), cs.estimate(entry.getKey()));
        }

        System.out.println("Port = " + portExp + " T = " + THRESHOLD + " D = " + D + " W = " + W + " K = " + K);
        // COMPUTE AND PRINT FINAL STATISTICS
        System.out.println("Number of processed items = " + streamLength[0]);
        System.out.println("Number of distinct items = " + frequencyMap.size());

        long max = 0L;
        ArrayList<Long> distinctKeys = new ArrayList<>(frequencyMap.keySet());
        Collections.sort(distinctKeys, Collections.reverseOrder());
       // System.out.println("Largest item = " + distinctKeys.get(0));


        //Extract Top-K from ground truth using a min-heap
        PriorityQueue<Map.Entry<Long, Long>> minHeap =
                new PriorityQueue<>(Comparator.comparingLong(Map.Entry::getValue));

        for (Map.Entry<Long, Long> entry : frequencyMap.entrySet()) {
            minHeap.offer(entry);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        //Sort Top-K descending by true frequency
        List<Map.Entry<Long, Long>> topK = new ArrayList<>(minHeap);
        topK.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));

        //Compare with frequencyCM and frequencyCS
        double totalRelErrorCM = 0.0;
        double totalRelErrorCS = 0.0;

        for (Map.Entry<Long, Long> entry : topK) {
            long item = entry.getKey();
            long trueFreq = entry.getValue();

            long estCM = frequencyCM.getOrDefault(item, 0L);
            long estCS = frequencyCS.getOrDefault(item, 0L);

            double relErrCM = Math.abs(estCM - trueFreq) / (double) trueFreq;
            double relErrCS = Math.abs(estCS - trueFreq) / (double) trueFreq;

            totalRelErrorCM += relErrCM;
            totalRelErrorCS += relErrCS;
        }

        //Print final average relative errors
        System.out.println("Number of Top-K Heavy Hitters = " + K);
        System.out.println("Avg Relative Error for Top-K Heavy Hitters with CM = " + (totalRelErrorCM / K));
        System.out.println("Avg Relative Error for Top-K Heavvy Hitters with CS = " + (totalRelErrorCS / K));

        //If K>10, additional print
        if (10 >= K) {
            System.out.println("\nTop K heavy hitters:");

            topK.sort(Comparator.comparingLong(Map.Entry::getKey));

            for (Map.Entry<Long, Long> entry : topK) {
                long item = entry.getKey();
                long trueFreq = entry.getValue();
                long estCM = frequencyCM.getOrDefault(item, 0L);
                System.out.printf("Item %-10d True Frequency = %-5d Estimated Frequency with CM = %-5d%n", item, trueFreq, estCM);
            }
            }
    }
}
