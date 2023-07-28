public class LinearRegression {
  
    private double[] x;
    private double[] y;

    private int datasetSize;
  
    private double slope;
    private double yIntercept;

    public LinearRegression(final double[] x, final double[] y) {
        validateAndInitDataset(x, y);
        learn();
    }

    private void validateAndInitDataset(final double[] x, final double[] y) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("Neither of the parameters can be null!");
        }
        final int xLength = x.length;
        if(xLength == 0){
            throw new IllegalArgumentException("Neither of the parameters can be empty!");
        }
        if (xLength != y.length) {
            throw new IllegalArgumentException("Dataset lengths must match!");
        }
        this.x = x;
        this.y = y;
        datasetSize = xLength;
    }

    private void learn() {
        double xSum = 0;
        double ySum = 0;
        for (int i = 0; i < datasetSize; i++) {
            xSum += x[i];
            ySum += y[i];
        }
        final double xAverage = xSum / datasetSize;
        final double yAverage = ySum / datasetSize;
        double deviationProductSum = 0;
        double xSquaredDeviationSum = 0;
        for (int i = 0; i < datasetSize; i++) {
            final double xDeviation = x[i] - xAverage;
            deviationProductSum = xDeviation * (y[i] - yAverage);
            xSquaredDeviationSum = xDeviation * xDeviation;
        }
        slope = deviationProductSum / xSquaredDeviationSum;
        yIntercept = yAverage - (slope * xAverage);
    }
  
    public double predict(final double input) {
        return slope * input + yIntercept;
    }
  
}
