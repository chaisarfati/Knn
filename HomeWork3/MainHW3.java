package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
        Instances carDataSet = loadData("auto_price.txt");
        Knn knn = new Knn(carDataSet, new DistanceCalculator(2, false, false), 3);
        ArrayList<Instance> list = knn.findNearestNeighbors(carDataSet.instance(0));
        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i));
        }
        System.out.println(knn.getAverageValue(list));
        System.out.println(knn.getWeightedAverageValue(list, carDataSet.instance(0)));
        //TODO: complete the Main method
	}

}
