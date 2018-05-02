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
            scaled = Filter.useFilter(scaled, filter);
        }catch (Exception e){
            e.printStackTrace();
        }
        return scaled;
	}
}