package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

class Nearest{
    Instance instance;
    double distance;
    public Nearest(Instance instance, double distance) {
        this.instance = instance;
        this.distance = distance;
    }
}


class DistanceCalculator {
    double p;
    boolean efficient;
    boolean infinity;

    public DistanceCalculator(double p, boolean efficient, boolean infinity) {
        this.p = p;
        this.efficient = efficient;
        this.infinity = infinity;
    }

    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it has a class variables.
     */
    public double distance (Instance one, Instance two) {
        if(infinity){
            return lInfinityDistance(one, two);
        }else{
            return lpDistance(one, two);
        }
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
        double result = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            double x1 = one.value(i),
                    x2 = two.value(i);
            result += Math.pow(Math.abs(x1 - x2) , p);
        }
        return Math.pow(result, 1/p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double maximum = Math.abs(one.value(0) - two.value(0));
        double current;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            current = Math.abs(one.value(i) - two.value(i));
            if(maximum < current){
                maximum = current;
            }
        }
        return maximum;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two, double threshold) {
        double result = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            double x1 = one.value(i),
                    x2 = two.value(i);
            result += Math.pow(Math.abs(x1 - x2), p);

            if(result > Math.pow(threshold, p)){
                return Math.pow(result, 1/p);
            }
        }
        return Math.pow(result, 1/p);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
        double max = 0, current;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            current = Math.abs(one.value(i) - two.value(i));
            if(max > threshold) return max;
            if(max < current) max = current;
        }
        return max;
    }

    public double effDistance(Instance one, Instance two, double max){
        if(infinity){
            return efficientLInfinityDistance(one, two, max);
        }else {
            return efficientLpDistance(one, two, max);
        }
    }
}

public class Knn implements Classifier {

    public boolean efficient = false;
    private Instances m_trainingInstances;
    private DistanceCalculator calculator;
    private int k;
    public boolean weight;

    public Knn(Instances m_trainingInstances){
        this.m_trainingInstances = m_trainingInstances;
    }

    public Knn(Instances m_trainingInstances, DistanceCalculator calculator, int k, boolean weight) {
        this.m_trainingInstances = m_trainingInstances;
        this.calculator = calculator;
        this.k = k;
        this.weight = weight;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_trainingInstances = instances;
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        ArrayList<Instance> list;
        if(efficient){
            list = effNearestNeighbors(instance);
        }else {
            list = findNearestNeighbors(instance);
        }

        if(!weight){
            return getAverageValue(list);
        }else{
            return getWeightedAverageValue(list, instance);
        }
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double result = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            result += Math.abs(instances.instance(i).classValue() - regressionPrediction(instances.instance(i)));
        }
        return result / instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param insances     Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances insances, int num_of_folds) {
        double valError = 0.0;
        //insances.randomize(new Random());
        for (int i = 0; i < num_of_folds; i++) {
            valError += calcAvgError(insances.testCV(num_of_folds, i));
        }
        valError /= num_of_folds;
        return valError;
    }




    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public ArrayList<Instance> findNearestNeighbors(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Nearest[] nearests = new Nearest[m_trainingInstances.numInstances() - 1];
        int j = 0;
        for (int i = 0; i < nearests.length; i++) {
            if(!equalInstance(instance, m_trainingInstances.instance(j))) {
                nearests[i] = new Nearest(m_trainingInstances.instance(j),
                        calculator.distance(m_trainingInstances.instance(j), instance));
                j++;
            }else{
                j++;
                i--;
            }
        }
        sort(nearests, 0, nearests.length - 1);
        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(nearests[i].instance);
        }
        return list;
    }
    /* Auxiliary function of findNearestNeighbors */
    private boolean equalInstance(Instance one, Instance two){
        for (int i = 0; i < one.numAttributes(); i++) {
            if(one.value(i) != two.value(i)) return false;
        }
        return true;
    }



    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public ArrayList<Instance> effNearestNeighbors(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Nearest[] nearests = new Nearest[m_trainingInstances.numInstances() - 1];
        int j = 0;
        double max = Double.POSITIVE_INFINITY;
        for (int i = 0; i < nearests.length; i++) {

            if(i < k + 1){
                max = Double.POSITIVE_INFINITY;
            }

            if(!equalInstance(instance, m_trainingInstances.instance(j))) {

                nearests[i] = new Nearest(m_trainingInstances.instance(j),
                        calculator.effDistance(m_trainingInstances.instance(j), instance, max));

                max = getMaxValue(nearests);
                j++;
            }else{
                j++;
                i--;
            }
        }
        sort(nearests, 0, nearests.length - 1);
        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(nearests[i].instance);
        }
        return list;
    }


    /**
     * Find k nearest neighbors using the efficient distance check
     * @param instance
     * @return
     */
    public ArrayList<Instance> effFindNearestNeighbors(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Nearest[] nearests = new Nearest[k];
        int j = 0;
        for (int i = 0; i < nearests.length; i++) {
            if(!equalInstance(instance, m_trainingInstances.instance(j))) {
                nearests[i] = new Nearest(m_trainingInstances.instance(j),
                        calculator.distance(m_trainingInstances.instance(j), instance));
                j++;
            }else{
                j++;
                i--;
            }
        }
        double currentMax;

        for (int i = nearests.length; i < m_trainingInstances.numInstances(); i++) {
            currentMax = getMaxValue(nearests);
            if ( calculator.effDistance(m_trainingInstances.instance(i), instance, currentMax)
                    < currentMax){
                nearests[getMaxIndex(nearests)] = new Nearest(m_trainingInstances.instance(i), calculator.distance(
                        m_trainingInstances.instance(i), instance
                ));
            }
        }

        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(nearests[i].instance);
        }
        return list;
    }
    /* Get max elem in Nearest arr*/
    public int getMaxIndex(Nearest[] arr){
        double max = arr[0].distance;
        int index = 0;
        double current;
        for (int i = 1; i < arr.length; i++) {
            current = arr[i].distance;
            if(max < current){
                max = current;
                index = i;
            }
        }
        return index;
    }
    /* Get max value in Nearest arr*/
    public double getMaxValue(Nearest[] arr){
        double max = arr[0].distance;
        int counter = 0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] == null){
                break;
            }else{
                counter++;
            }
        }
        for (int i = 0; i < counter; i++) {
            if (arr[i].distance > max) {
                    max = arr[i].distance;
                }
        }
        return max;
    }



    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param set
     * @return the average of the targets
     */
    public double getAverageValue(List<Instance> set) {
        double result = 0;
        for (int i = 0; i < set.size(); i++) {
            result += set.get(i).classValue();
        }
        return result / set.size();
    }


    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @param list
     * @param instance
     * @return the weighted average of the targets
     */
    public double getWeightedAverageValue(List<Instance> list, Instance instance) {
        double upperSum = 0.0;
        double weightTotal = 0.0;
        double weightI;

        // Computes the denominator of the formula
        for (int i = 0; i < list.size(); i++) {
            // Checks if the distance is not zero
            if(Math.pow(calculator.distance(list.get(i), instance), 2) != 0){
                weightTotal += 1.0 / Math.pow(calculator.distance(list.get(i), instance), 2);
            }
        }
        // Compute the numerator of the formula
        for (int i = 0; i < list.size(); i++) {
            weightI = 1.0 / Math.pow(calculator.distance(list.get(i), instance), 2);
            // Checks if the distance is not zero again
            if(Math.pow(calculator.distance(list.get(i), instance), 2) != 0.0){
                upperSum += weightI * list.get(i).classValue();
            }

        }
        // If the instances in the collection all have a 0 distance from instance
        if(weightTotal == 0.0 && upperSum == 0.0){
            return getAverageValue(list);
        }

        return upperSum / weightTotal;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }

    // Setters of Knn
    public void setCalculator(DistanceCalculator calculator) {
        this.calculator = calculator;
    }
    public void setK(int k) {
        this.k = k;
    }
    public void setWeight(boolean weight) {
        this.weight = weight;
    }
    public void setM_trainingInstances(Instances m_trainingInstances) {
        this.m_trainingInstances = m_trainingInstances;
    }
    public void setEfficient(boolean efficient) {
        this.efficient = efficient;
    }

    /**
     * Quicksort implementation that sorts the array of Nearest
     * object in increasing order of the distance field
     * @param arr
     * @param low
     * @param high
     */
    public void sort(Nearest arr[], int low, int high) {
        if (low < high) {
            /* pi is partitioning index, arr[pi] is
              now at right place */
            int pi = partition(arr, low, high);
            // Recursively sort elements before
            // partition and after partition
            sort(arr, low, pi - 1);
            sort(arr, pi + 1, high);
        }
    }
    /* Partition function of quicksort */
    private int partition(Nearest[] arr, int low, int high) {
        double pivot = arr[high].distance;
        int i = (low - 1); // index of smaller element
        for (int j = low; j < high; j++) {
            // If current element is smaller than or
            // equal to pivot
            if (arr[j].distance <= pivot) {
                i++;
                // swap arr[i] and arr[j]
                Nearest temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        // swap arr[i+1] and arr[high] (or pivot)
        Nearest temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }


}


