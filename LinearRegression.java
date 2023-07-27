import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class LinearRegression {
  
    private final List<DatasetElement> dataset;
    private final double slope;
    private final double yIntercept;

    public LinearRegression(final List<DatasetElement> dataset) {
        validateDataset(dataset);
        this.dataset = dataset;
        final double averageX = calculateAverage(DatasetElement::x);
        final double averageY = calculateAverage(DatasetElement::y);
        final List<Double> deviationX = calculateDeviation(averageX, DatasetElement::x);
        final List<Double> deviationY = calculateDeviation(averageY, DatasetElement::y);
        final double sumDeviationProduct = sumDeviationProduct(deviationX, deviationY);
        final double squaredXDeviations = sumSquaredXDeviations(deviationX);
        slope = calculateSlope(sumDeviationProduct, squaredXDeviations);
        yIntercept = calculateYIntercept(averageX, averageY);
    }

    private void validateDataset(final List<DatasetElement> dataset) {
        if (dataset == null || dataset.isEmpty()) {
            throw new IllegalArgumentException("Dataset must not be null nor empty!");
        }
    }

    private double calculateAverage(final ToDoubleFunction<DatasetElement> getter) {
        return dataset.parallelStream().mapToDouble(getter).summaryStatistics().getAverage();
    }

    private List<Double> calculateDeviation(final double average, final ToDoubleFunction<DatasetElement> getter) {
        return dataset.parallelStream().mapToDouble(e -> getter.applyAsDouble(e) - average).boxed().toList();
    }

    private double sumDeviationProduct(final List<Double> deviationX, final List<Double> deviationY) {
        return IntStream.range(0, dataset.size()).parallel().mapToDouble(i -> deviationX.get(i) * deviationY.get(i))
                .sum();
    }

    private double sumSquaredXDeviations(final List<Double> deviationX) {
        return deviationX.parallelStream().mapToDouble(e -> e * e).sum();
    }

    private double calculateSlope(final double sumDeviationProduct, final double squaredXDeviations) {
        return sumDeviationProduct / squaredXDeviations;
    }

    private double calculateYIntercept(final double averageX, final double averageY) {
        return averageY - (slope * averageX);
    }

    public double predict(final double input) {
        return slope * input + yIntercept;
    }

    public record DatasetElement(double x, double y) {
    }

}
