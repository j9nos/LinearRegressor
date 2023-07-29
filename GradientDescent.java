public class GradientDescent {
  
    private double[] x;
    private double[] y;

    private int datasetSize;

    private double slope = Math.random();
    private double yIntercept = Math.random();

    public GradientDescent(final double[] x, final double[] y, final double learningRate, final int learningIterations) {
        validateAndInitDataset(x, y);
        learn(learningRate, learningIterations);
    }

    private void validateAndInitDataset(final double[] x, final double[] y) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("Neither of the parameters can be null!");
        }
        datasetSize = x.length;
        if (datasetSize == 0) {
            throw new IllegalArgumentException("Neither of the parameters can be empty!");
        }
        if (datasetSize != y.length) {
            throw new IllegalArgumentException("Dataset lengths must match!");
        }
        this.x = x;
        this.y = y;
    }

    private void learn(final double learningRate, final double learningIterations) {
        final double normalization = learningRate / datasetSize;
        for (int i = 0; i < learningIterations; i++) {
            double slopeGradient = 0;
            double yInterceptGradient = 0;
            for (int j = 0; j < datasetSize; j++) {
                final double yPredicted = predict(x[j]);
                final double error = y[j] - yPredicted;
                slopeGradient -= x[j] * error;
                yInterceptGradient -= error;
            }
            slope -= slopeGradient * normalization;
            yIntercept -= yInterceptGradient * normalization;
        }
    }

    public double predict(final double input) {
        return slope * input + yIntercept;
    }

}

