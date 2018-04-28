package HomeWork3;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
        // Original data set
        Instances carDataSet = loadData("auto_price.txt");
        // Scaled data set
        FeatureScaler scaler = new FeatureScaler();
        Instances scaledCarDataSet = scaler.scaleData(carDataSet);


        System.out.println("----------------------------\n" +
                "Results for original dataset:\n" +
                "----------------------------");

        boolean globWeight = false, globInfinity = false;
        double globP = 1;
        int globK = 1;

        boolean currWeight, currInfinity;
        double currError;

        Knn knn = new Knn(carDataSet);
        DistanceCalculator calc = new DistanceCalculator(globP, false, globInfinity);
        knn.setWeight(globWeight);
        knn.setCalculator(calc);
        knn.setK(globK);
        double lowestError = knn.crossValidationError(carDataSet, 10);


        for (int i = 0; i < 2; i++) {
            currWeight = (i==0) ? false : true;
            knn.setWeight(currWeight);
            for (int j = 1; j <= 4 ; j++) {
                currInfinity = (j == 4) ? true : false;
                calc = new DistanceCalculator(j, false, currInfinity);
                knn.setCalculator(calc);
                for (int k = 1; k <= 20; k++) {
                    knn.setK(k);
                    currError = knn.crossValidationError(carDataSet, 10);
                    if(lowestError > currError){
                        globK = k;
                        globP = j;
                        globInfinity = currInfinity;
                        globWeight = currWeight;
                        lowestError = currError;
                    }
                }
            }
        }

        String realLp = (globP==4) ? "infinity" : Double.toString(globP);
        String weightNormal = (!globWeight) ? "uniform" : "weighted";

        System.out.println("Cross validation error with K = " + globK + ", " +
                "lp = " + realLp + ", majority function = " + weightNormal + " for auto_price data is: " +
                lowestError + "\n");


        System.out.println("----------------------------\n" +
                "Results for scaled dataset:\n" +
                "----------------------------");


        globWeight = false;
        globInfinity = false;
        globP = 1;
        globK = 1;


        calc = new DistanceCalculator(globP, false, globInfinity);
        knn.setM_trainingInstances(scaledCarDataSet);
        knn.setWeight(globWeight);
        knn.setCalculator(calc);
        knn.setK(globK);
        lowestError = knn.crossValidationError(carDataSet, 10);

        for (int i = 0; i < 2; i++) {
            currWeight = (i==0) ? false : true;
            knn.setWeight(currWeight);
            for (int j = 1; j <= 4 ; j++) {
                currInfinity = (j == 4) ? true : false;
                calc = new DistanceCalculator(j, false, currInfinity);
                knn.setCalculator(calc);
                for (int k = 1; k <= 20; k++) {
                    knn.setK(k);
                    currError = knn.crossValidationError(carDataSet, 10);
                    if(lowestError > currError){
                        globK = k;
                        globP = j;
                        globInfinity = currInfinity;
                        globWeight = currWeight;
                        lowestError = currError;
                    }
                }
            }
        }

        realLp = (globP==4) ? "infinity" : Double.toString(globP);
        weightNormal = (!globWeight) ? "uniform" : "weighted";

        System.out.println("Cross validation error with K = " + globK + ", " +
                "lp = " + realLp + ", majority function = " + weightNormal + " for auto_price data is: " +
                lowestError + "\n");

    }

}
