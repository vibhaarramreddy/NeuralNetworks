import java.io.*;
import java.util.Scanner;

/**
 * @author Datha Arramreddy
 * @version 10/5/19
 *
 * This class creates a neural network. This network's operation to emulate a multilayer perceptron.
 * This class utilizes two different files to configure the layers of the network and to determine 
 * expected input and output values. Currently, this network is designed to have three activation layers 
 * with any number of hidden activations in the hidden activation later. In the future, the code will be 
 * designed to use an adpative learning factor to improve efficiency and it will be .
 * 
 * Methods: 
 * configuration(...), inputs(...), setWeights(),
 * randomizeWeights(), setLearningConstant(...),
 * fillInputActivations(...), fillNetwork(), thresholdFunction(...),
 * calculateIndividualError(...), calculateDeltaWeights(...), train(),
 * printWeights(), printErrors(...), printNetworkOutputs(), findMaxError(...),
 * isDescentFinished(), printOutputs(), main()
 */
public class NeuralNetwork 
{

   private double[][] activations;

   private static double[][][] weights;

   private int[] numActivationsInLayer;

   private int numLayers;

   private double[][] inputs;

   private double[] truthVals;

  //private double error;

   private double learningConstant;

  // private double lambdaMultiplier = 1.01; (Use when applying adaptive lambda)

   private double lambdaLimit = 1.5; //REMEMBER TO CONSOLIDATE ALL THE MAGIC NUMBERS
   
   private final double MIN_ERROR = 0.001;
   
   private final int MAX_ITERATIONS = 10000000;
   
   private final double MIN_LEARNING_CONSTANT = 0.0000001; 
   
   /**
    * Constructor for the Neural Network, initializing all the variables necessary for building the model based on user input
    * 
    * @param configName    the name of the file that contains all the configurations for the network
    * @param allInputs     the name of the file that contains all the inputs
    * @param learningFact  the value the learning factor will be set to.
    */
   public NeuralNetwork(String configName, String allInputs) throws FileNotFoundException
   {
      configuration(configName);
      inputs(allInputs);
      activations = new double[numLayers][];
      
      /*
       * This for loop creates a jagged array to host the activations of the network. The activations array
       * is initially filled with default double values.
       */
      for (int layer = 0; layer < numLayers; layer++)
      {
         activations[layer] = new double[numActivationsInLayer[layer]];
      }
      
      //Creates a jagged weight array
      weights = new double[numLayers - 1][][];
      for (int m = 0; m < numLayers - 1; m++)
      {
         weights[m] = new double[numActivationsInLayer[m]][numActivationsInLayer[m+1]];
      }
   }
   
   /**
    * The configuration method takes in the name of a file containing the configuration of the network.
    * @param filename the name of the file containing an integer representing the number of layers
    *                 in the network. Then it contains numbers representing the number of
    *                 activations in each layer.
    * @throws FileNotFoundException
    */
   public void configuration(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));
      numLayers = sc.nextInt();

      numActivationsInLayer = new int[numLayers];
      for(int layer = 0; layer < numLayers; layer++)
      {
         if (sc.hasNextInt())
         {
            numActivationsInLayer[layer] = sc.nextInt();
         }
      }
   }

   /**
    * The inputs method takes in the name of a file containing the inputs of the network as well as the
    * target values.
    * @param filename the name of the file containing the inputs and target values of the network
    * @throws FileNotFoundException
    */
   public void inputs(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));

      int numInputSets = sc.nextInt();
      inputs = new double[numInputSets][numActivationsInLayer[0]];
      truthVals = new double[numInputSets];

      for (int set = 0; set < numInputSets; set++)
      {
         if (sc.hasNextLine())
         {
            sc.nextLine();
         }
         for (int k = 0; k < numActivationsInLayer[0]; k++)
         {
            if (sc.hasNextDouble())
            {
               inputs[set][k] = sc.nextDouble();
            }
         }
         if (sc.hasNextDouble())
         {
            truthVals[set] = sc.nextDouble();
         }
      }

      sc.close();
   }
   /**
    * Sets the learning factor
    * @param learningFact  the value the learning factor will be set to
    */
   public void setLearningFactor(double learningFact)
   {
      learningConstant = learningFact;
   }

   /**
    * Sets all the weights in the 3-D array. For now, this is a manual implementation as 
    * the weights are from my spreadsheet and entered here.
    */
   public void setWeights()
   {
      weights[0][0][0] = 0.409;
      weights[0][0][1] = 0.409;
      weights[0][1][0] = 0.409;
      weights[0][1][1] = 0.409;
      weights[1][0][0] = 0.409;
      weights[1][1][0] = 0.409;
   }

   /**
    * Creates random values for each of the weights in the weights array. Each
    * of the weights is greater than or equal to 0.0 and less than 1.0.
    */
   public void randomizeWeights()
   {
      for (int m = 0; m < weights.length; m++)
      {
         for (int left = 0; left < weights[m].length; left++)
         {
            for (int right = 0; right < weights[m][left].length; right++)
            {
               weights[m][left][right] = 2 * Math.random(); //magic number used to enlargen range
            }
         }
      }
   }

   /**
    * Takes the inputs from a particular test case and places them into the activations array.
    * @param inputCase the input case for which to fill the inputs
    */
   public void fillInputActivations(int inputCase)
   {
      for (int k = 0; k < inputs[inputCase].length; k++)
      {
         activations[0][k] = inputs[inputCase][k];
      }
   }

   /**
    * Fills in the rest of the activations following the input layer.
    */
   public void fillNetwork()
   {
      for (int layer = 1; layer < numLayers; layer++)
      {
         for (int node = 0; node < numActivationsInLayer[layer]; node++)
         {
            double actValue = 0.0;
            for (int sourceNode = 0; sourceNode < numActivationsInLayer[layer - 1]; sourceNode++)
            {
               actValue += (activations[layer - 1][sourceNode] * weights[layer - 1][sourceNode][node]);
            }
            activations[layer][node] = thresholdFunc(actValue);
         }
      }
   }

   /**
    * Attaches a particular input to an output within a particular range.
    * @param input the input value to be bounded.
    * @return the output value assigned to the given input.
    */
   public double thresholdFunc(double input)
   {
      return (1.0 / (1 + Math.pow(Math.E, -1*input)));
   }

   /**
    * Takes a particular input case and calculates its respective error.
    * @param inputCase the input case for which the error is to be calculated
    * @return the error for the input case
    */
   public double calculateIndividualError(int testCase)
   {
      double output = activations[2][0];
      double targetOutput = truthVals[testCase];
      double err = 0.5 * (output - targetOutput) * (output - targetOutput);
      return err;
   }

   /**
    * Modifies the weights of the network using the error.
    * The method follows the steepest descent method of minimization.
    * @param testCase the test case to be used
    * @return an array of Doubles containing the modified weights
    */
   public double[][][] calculateDeltaWeights(int testCase)
   {
      double[][][] modifiedWeights = new double[numLayers - 1][][];
      for (int m = 0; m < numLayers - 1; m++)
      {
         modifiedWeights[m] = new double[numActivationsInLayer[m]][numActivationsInLayer[m+1]];
      }

      double outputInverse = activations[2][0];
      //modifies the weights in the leftmost weight layer for a N-M-1 configuration
      for (int k = 0; k < modifiedWeights[0].length; k++)
      {
         for (int j = 0; j < modifiedWeights[0][k].length; j++)
         {
            double sigmoidInverse = activations[1][j];
            double deltaWeight = learningConstant * activations[0][k] * sigmoidInverse * (1 - sigmoidInverse);
            deltaWeight *= (truthVals[testCase] - outputInverse) * outputInverse * (1 - outputInverse);
            deltaWeight *= weights[1][j][0];
            modifiedWeights[0][k][j] = weights[0][k][j] + deltaWeight;
         }
      }
      
      //modifies the weights in the rightmost weight layer for a N-M-1 configuration
      for (int j = 0; j < modifiedWeights[1].length; j++)
      {
         double deltaWeight = learningConstant * (truthVals[testCase] - outputInverse);
         deltaWeight *= (outputInverse * (1 - outputInverse) * activations[1][j]);
         modifiedWeights[1][j][0] = weights[1][j][0] + deltaWeight;
      }     

      return modifiedWeights;
   }

   /**
    * Alters weights to minimize the individual errors for test cases. Training terminates once the stopping conditions are reached.
    * Stopping conditions include: maximum individual error is below a given threshold, the iteration number is higher than 
    * a given threshold, or the learning constant is below a given threshold.
    */
   public void train()
   {
      double initialLambda = 0.1;

      randomizeWeights();
      setLearningFactor(initialLambda);

      fillInputActivations(0);
      fillNetwork();

      double[] errors = new double[inputs.length];
      errors[0] = calculateIndividualError(0);
      if (inputs.length > 1)
      {
         for (int i = 1; i < inputs.length; i++)
         {
            errors[i] = Double.MAX_VALUE;
         }
      }

      int iterations = 0;
      double maxError = findMaxError(errors);

      while (!isDescentFinished(maxError, iterations, learningConstant))
      {
         int testCase = iterations % inputs.length;
         fillInputActivations(testCase);
         fillNetwork();
         errors[testCase] = calculateIndividualError(testCase);

         double[][][] oldWeights = weights;
         double[][][] newWeights = calculateDeltaWeights(testCase);
         weights = newWeights;

         fillNetwork(); //using updated weights from the delta W step
         double newError = calculateIndividualError(testCase);

         if (newError < errors[testCase])
         {
            if (learningConstant < lambdaLimit)
            {
               //learningConstant *= lambdaMultiplier;
            }
            maxError = findMaxError(errors);
         }
         else
         {
            weights = oldWeights;
            //learningConstant /= lambdaMultiplier;
         }
         iterations++;
      }
      
      System.out.println("Number of Iterations Taken: " + iterations);
      System.out.println();
      
      System.out.println("Learning Factor: " + learningConstant);
      System.out.println();
      
      printErrors(errors);
      System.out.println();
      
      printNetworkOutputs();
      System.out.println();
      
      //printWeights();
      
   }

   /**
    * Prints all of the weights in the weight array.
    */
   public void printWeights()
   {
      for (int m = 0; m < weights.length; m++)
      {
         for (int left = 0; left < weights[m].length; left++)
         {
            for (int right = 0; right < weights[m][left].length; right++)
            {
               System.out.println("weights[" + m + "][" + left + "][" + right + "] = " + weights[m][left][right]);
            }
         }
      }
   }

   /**
    * Prints all the error values from the array.
    * @param errors the array to be printed
    */
   public void printErrors(double[] errors)
   {
      for (int i = 0; i < errors.length; i++)
      {
         String inputString = "[";
         for (int inputIndex = 0; inputIndex < inputs[i].length - 1; inputIndex++)
         {
            inputString += inputs[i][inputIndex] + ", ";
         }
         inputString += inputs[i][inputs[i].length - 1] + "]";
         System.out.println("Error for case " + inputString + ": " + errors[i]);
      }
   }

   /**
    * Print values that are looked found in the train function.
    */
   public void printNetworkOutputs()
   {
      for (int testCase = 0; testCase < inputs.length; testCase++)
      {
         fillInputActivations(testCase);
         fillNetwork();
         printOutputs(testCase);
      }
   }

   /**
    * Finds the maximum Double in an array of errors.
    * @param errors the array to be searched for a maximum
    * @return the maximum error
    */
   public double findMaxError(double[] errors)
   {
      double max = errors[0];
      for (int i = 0; i < errors.length; i++)
      {
         if (errors[i] > max)
         {
            max = errors[i];
         }
      }
      return max;
   }

   /**
    * Determines whether the training is complete based on the error, the number
    * of iterations, or the learning constant.
    * @param error the maximum error in the array
    * @param iterations the number of iterations
    * @param lambda the learning constant
    * @return whether or not the descent is complete
    */
   public boolean isDescentFinished(double error, int iterations, double lambda)
   {
      if (error < MIN_ERROR)
      {
         System.out.println();
         System.out.println("Training stopped: Max error is less than " + MIN_ERROR + ".");
         System.out.println();
         return true;
      }
      else if (iterations > MAX_ITERATIONS)
      {
         System.out.println();
         System.out.println("Training stopped: Iterations are greater than " + MAX_ITERATIONS + ".");
         System.out.println();
         return true;
      }
      else if (lambda < MIN_LEARNING_CONSTANT)
      {
         System.out.println();
         System.out.println("Training stopped: Learning constant is less than " + MIN_LEARNING_CONSTANT + ".");
         System.out.println();
         return true;
      }
      else
      {
         return false;
      }
   }

   /**
    * Prints the output value for a given test case of inputs.
    * @param testCase the input values for the output value to be printed
    */
   public void printOutputs(int testCase)
   {
      String inputString = "[";
      for (int inputIndex = 0; inputIndex < inputs[testCase].length - 1; inputIndex++)
      {
         inputString += inputs[testCase][inputIndex] + ", ";
      }
      inputString += inputs[testCase][inputs[testCase].length - 1] + "]";

      for (int output = 0; output < activations[numLayers - 1].length; output++)
      {
         System.out.println("Output " + output + " for input case " + inputString + ": " + activations[numLayers - 1][output]);
      }
   }

   /**
    * The main method creates a neural network and inputs the configuration and truthTable files outside of compile time. It sets 
    * up the weights and activations in the corresponding arrays and prints the output value of the
    * network.
    * @param args the arguments of the main method.
    * @throws IOException
    */
   public static void main(String[] args) throws IOException
   {
      NeuralNetwork tester = new NeuralNetwork("./src/Networkconfig.txt", "./src/truthTable.txt");

      tester.train();
   }
}
