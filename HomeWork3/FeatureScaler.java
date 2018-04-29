package HomeWork3;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData(Instances instances) {
        Instances scaled = new Instances(instances);
        Standardize filter = new Standardize();
        try{
            filter.setInputFormat(scaled);
        }catch (Exception e){
            e.printStackTrace();
        }

        try{
            scaled = Filter.useFilter(scaled, filter);
        }catch (Exception e){
            e.printStackTrace();
        }
        return scaled;
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