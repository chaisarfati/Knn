package HomeWork3;

import weka.core.Instances;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData(Instances instances) {
        // Stores the means of the values for each feature
        double[] means = new double[instances.numAttributes()-1];
        for (int i = 1; i < means.length; i++) {
            means[i] = instances.meanOrMode(i);
        }
        // Stores the standard deviations of the values for each feature
        double[] stds = new double[instances.numAttributes()-1];
        for (int i = 1; i < stds.length; i++) {
            stds[i] = Math.sqrt(instances.variance(i));
        }

        // Creates new Instances object that'll contain scaled version of instances
        Instances result = new Instances(instances, 0);
        // Scaling every instance of instances and store it in result
        for (int i = 0; i < instances.numInstances(); i++) {
            result.add(instances.instance(i));
            for (int j = 1; j < instances.numAttributes()-1; j++) {
                double normalizedValue = (instances.instance(i).value(j) - means[j]) / stds[j];
                result.instance(i).setValue(j, normalizedValue);
            }
        }
        return result;
	}


    /**
     * Returns the standard deviation of all the values attained for the ith feature
     * @param instances
     * @param i
     * @return
     */
	public static double std(Instances instances, int i){
        double mean = instances.meanOrMode(i);
        double variance = 0;
        for (int j = 0; j < instances.numInstances(); j++) {
            variance += Math.pow(instances.instance(j).value(i) - mean, 2);
        }
        variance /= instances.numInstances() - 1;
        return Math.sqrt(variance);
    }
}