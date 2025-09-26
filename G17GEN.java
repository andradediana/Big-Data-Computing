
import java.io.*;
import java.util.*;

public class G17GEN {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Usage: java FairVsLloydDataGenerator <N> <K>");
            return;
        }

        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);

        //int N = 10000;
        //int K = 5;

        String outputFile = "gen_data.csv";

        Random rand = new Random();

        // Generate cluster centers randomly in [0,100]x[0,100]
        double[][] centers = new double[K][2];
        for (int i = 0; i < K; i++) {
            centers[i][0] = rand.nextDouble() * 100;
            centers[i][1] = rand.nextDouble() * 100;
        }

        int pointsPerCluster = N / K;
        int remainder = N % K;

        // Use a helper class to store point info with cluster
        class Point {
            double x, y;
            char clazz;
            int cluster;
            Point(double x, double y, char clazz, int cluster) {
                this.x = x; this.y = y; this.clazz = clazz; this.cluster = cluster;
            }
        }

        List<Point> points = new ArrayList<>();

        for (int i = 0; i < K; i++) {
            int clusterSize = pointsPerCluster + (i < remainder ? 1 : 0);

            // Alternate majority class A and B
            char majorityClass = (i % 2 == 0) ? 'A' : 'B';
            char minorityClass = (majorityClass == 'A') ? 'B' : 'A';

            double majorityFraction = 0.8;
            int majorityCount = (int) (clusterSize * majorityFraction);
            int minorityCount = clusterSize - majorityCount;

            for (int j = 0; j < majorityCount; j++) {
                double x = centers[i][0] + rand.nextGaussian() * 3;
                double y = centers[i][1] + rand.nextGaussian() * 3;
                points.add(new Point(x, y, majorityClass, i));
            }

            for (int j = 0; j < minorityCount; j++) {
                double x = centers[i][0] + 5 + rand.nextGaussian() * 3;
                double y = centers[i][1] + 5 + rand.nextGaussian() * 3;
                points.add(new Point(x, y, minorityClass, i));
            }
        }

        // Shuffle points so output isn't sorted by cluster
        Collections.shuffle(points);

        // Write CSV file
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            for (Point p : points) {
                writer.printf("%.3f,%.3f,%c%n", p.x, p.y, p.clazz);
            }
        }

        System.out.println("Dataset saved to " + outputFile);

        // Print points with their cluster index
        for (Point p : points) {
            System.out.printf("Point (%.3f, %.3f), Class %c%n", p.x, p.y, p.clazz);
        }
    }
}
