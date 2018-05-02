package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
            list = effFindNearestNeighbor2(instance);
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
        insances.randomize(new Random());
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
        double upperSum = 0.0;
        double weightTotal = 0.0;
        double weightI;

        for (int i = 0; i < list.size(); i++) {
            if(Math.pow(calculator.distance(list.get(i), instance), 2) != 0){
                weightTotal += 1.0 / Math.pow(calculator.distance(list.get(i), instance), 2);
            }
        }


        for (int i = 0; i < list.size(); i++) {
            weightI = 1.0 / Math.pow(calculator.distance(list.get(i), instance), 2);

            if(Math.pow(calculator.distance(list.get(i), instance), 2) != 0.0){
                upperSum += weightI * list.get(i).classValue();
            }

        }

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

    private boolean equalInstance(Instance one, Instance two){
        for (int i = 0; i < one.numAttributes(); i++) {
            if(one.value(i) != two.value(i)) return false;
        }
        return true;
    }


    public ArrayList<Instance> effFindNearestNeighbor(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Nearest[] nearests = new Nearest[k + 1];
        Instances copy = new Instances(m_trainingInstances);
        for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
            copy.add(m_trainingInstances.instance(i));
        }

        double maximum;

        for (int i = 0; i < k; i++) {
            insertElement(nearests, copy.instance(i), instance, Double.POSITIVE_INFINITY);
        }
        maximum = insertElement(nearests, copy.instance(k), instance, Double.POSITIVE_INFINITY);
        for (int i = k + 1; i < copy.numInstances(); i++) {
            if(!equalInstance(instance, copy.instance(i))) {
                maximum = insertElement(nearests, copy.instance(i), instance, maximum);
            }
        }

        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(nearests[i].instance);
        }
        return list;
    }
    /* Helper of findNearestNeighbors */
    public double insertElement(Nearest[] arr, Instance element, Instance in, double max){
        int k = arr.length - 1;
        Nearest elem = new Nearest(element, calculator.effDistance(element, in, max));
        int counter = 0;
        for (int i = 0; i < k; i++) {
            if(arr[i] == null) counter++;
        }

        if(counter >= 1){
            for (int i = 0; i < k; i++ ) {
                if(arr[i] == null){
                    arr[i] = elem;
                    break;
                }

            }
            int i = 1;
            double maxi = arr[0].distance;
            double current;
            while (arr[i] != null){
                current = arr[i].distance;
                if(current > maxi){
                    maxi = current;
                }
                i++;
            }
            return maxi;
        }else {
            for (int i = 0; i < k; i++) {
                if (elem.distance < arr[i].distance) {
                    arr[k] = elem;
                    sort(arr, 0, arr.length - 1);
                    arr[k] = null;
                    return arr[arr.length - 2].distance;
                }
            }
        }
        return 0;
    }


    public ArrayList<Instance> effFindNearestNeighbor3(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Instance[] best = new Instance[k];

        for (int i = 0; i < k; i++) {
            best[i] = m_trainingInstances.instance(i);
        }
        for (int i = k; i < m_trainingInstances.numInstances(); i++) {
            if(calculator.distance(m_trainingInstances.instance(i), instance) < maxDistanceInArr(best, instance)){
                best[getMaxIndex(best, instance)] = m_trainingInstances.instance(i);
            }
        }

        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(best[i]);
        }
        return list;
    }


    public ArrayList<Instance> effFindNearestNeighbor2(Instance instance) {
        ArrayList<Instance> list = new ArrayList<>();
        Instance[] best = new Instance[k];
        double max = 0;

        for (int i = 0; i < k; i++) {
            best[i] = m_trainingInstances.instance(i);
        }
        for (int i = k; i < m_trainingInstances.numInstances(); i++) {
            max = maxDistanceInArr(best, instance);
            if(calculator.effDistance(m_trainingInstances.instance(i), instance, max) < max){
                best[getMaxIndex(best, instance)] = m_trainingInstances.instance(i);
            }
        }

        // Add to the list the k nearest neighbors
        for (int i = 0; i < k; i++) {
            list.add(best[i]);
        }
        return list;
    }

    public int getMaxIndex(Instance[] arr, Instance in){
        double max = calculator.distance(arr[0], in);
        int index = 0;
        double current;
        for (int i = 1; i < arr.length; i++) {
            current = calculator.distance(arr[i], in);
            if(max < current){
                max = current;
                index = i;
            }
        }
        return index;
    }

    public double maxDistanceInArr(Instance[] arr, Instance in){
        double max = calculator.distance(arr[0], in);
        double current;
        for (int i = 1; i < arr.length; i++) {
            current = calculator.distance(arr[i], in);
            if(current > max){
                max =  current;
            }
        }
        return max;
    }
}


