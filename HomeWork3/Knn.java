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
        if(efficient && infinity){
            return efficientLInfinityDistance(one, two);
        }else if(efficient && !infinity){
            return efficientLpDistance(one, two);
        }else if(!efficient && infinity){
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
    private double efficientLpDistance(Instance one, Instance two) {
        return 0.0;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        return 0.0;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck {Regular, Efficient}
    private Instances m_trainingInstances;
    private DistanceCalculator calculator;
    private int k;

    public Knn(Instances m_trainingInstances, DistanceCalculator calculator, int k) {
        this.m_trainingInstances = m_trainingInstances;
        this.calculator = calculator;
        this.k = k;
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
        return 0.0;
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
        return 0.0;
    }


    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public ArrayList<Instance> findNearestNeighbors(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Nearest[] nearests = new Nearest[m_trainingInstances.numInstances()];
        for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
            nearests[i] = new Nearest(m_trainingInstances.instance(i),
                    calculator.distance(m_trainingInstances.instance(i), instance));
        }
        sort(nearests, 0, nearests.length - 1);
        for (int i = 0; i < k; i++) {
            list.add(nearests[i].instance);
        }
        return list;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
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
     * @return
     */
    public double getWeightedAverageValue(List<Instance> list, Instance instance) {
        double weightTotal = 0.0;
        for (int i = 0; i < list.size(); i++) {
            weightTotal += 1 / Math.pow(calculator.distance(list.get(i), instance), 2);
        }

        double weightI;
        double upperSum = 0.0;
        for (int i = 0; i < list.size(); i++) {
            weightI = 1 / Math.pow(calculator.distance(list.get(i), instance), 2);
            upperSum += weightI * list.get(i).classValue();
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

    public void setCalculator(DistanceCalculator calculator) {
        this.calculator = calculator;
    }


    public void setK(int k) {
        this.k = k;
    }

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


    /* The main function that implements QuickSort()
      arr[] --> Array to be sorted,
      low  --> Starting index,
      high  --> Ending index */
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

}

