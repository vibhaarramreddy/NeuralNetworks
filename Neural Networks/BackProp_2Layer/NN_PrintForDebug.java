import java.io.*;
import java.util.Scanner;

/**
 * @author Vibha Arramreddy
 * December 6th 2019
 * This class creates a simple multilayer perceptron that has three activation layers. 
 * The number of activations in each layer is user configurable.
 */ 

public class MultilayerPerceptron 
{

   private double[][] activations;   
   private double[][][] weights;
   private double[][] Θvalues;
   private int numLayers;
   private int numRows;
   private int numOutputs;
   private int numInputs;
   

   /**
    * Constructs a multilevel perception with configurable number of layers and inputs.
    * 
    * @param numInputs    number of nodes in the activation layer
    * @param numOutputs   number of nodes in the output layer
    * @param numHidden    array whose length represents the number of hidden layers and value(s) represents the number of nodes in the layer(s)
    */
   public MultilayerPerceptron(int numInputs, int numOutputs, int[] numHidden)
   {
      numLayers = 2 + numHidden.length;
      
      //finding the max number of rows or nodes
      int maxNodes = Math.max(numInputs, numOutputs);
      for (int i = 0; i < numHidden.length; i++)
      {
         if (numHidden[i] > maxNodes)
         {
            maxNodes = numHidden[i];
         }
      }
     
      numRows = maxNodes;
      
      this.numInputs = numInputs;
      this.numOutputs = numOutputs;
            
      activations = new double[numLayers][];                                 //The array containing all the node values
      weights = new double[(numLayers-1)][numRows][numRows];                 //The array containing all the weight values
         
      activations[0] = new double[numInputs];                                 //Defines the number or nodes/rows in the activation layer
      activations[numLayers-1] = new double[numOutputs];                      //Defines the number of nodes/rows in the output layer
      for (int i = 0; i < numHidden.length; i++)                              //Loop that defines all of the hidden layers
      {
         activations[i+1] = new double[numHidden[i]];
      }
     
      Θvalues = new double[numLayers][numRows];
   }


   /**
    * Sets the input values for the network.
    * 
    * @param inputValues the values for the input nodes in order
    */
   public void setInputValues(double[] inputValues)
   {
      for (int i = 0; i < inputValues.length; i++)             //Iterates over the nodes in the activation layer
      {
         activations[0][i] = inputValues[i];
      }  
   }  
   
   /**
    * Outputs a random number and has a configurable range. 
    * @param min  the min value possible
    * @param max  the max value possible 
    * @return a random values between min and max
    */
   public double randomNumberGenerator(double min, double max)
   {
      double range = max - min;
      return ((Math.random() * range) + min);
   }
     
   /**
    * Randomizes the weight values for the perceptron.
    * @param min  the min value possible
    * @param max  the max value possible
    */
   public void randomizeWeightValues(double min, double max)
   { 
      
      for (int n= 0; n < weights.length; n++)
      {
         for (int leftA= 0; leftA < weights[n].length; leftA++)
         {
            for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
            {
               weights[n][leftA][rightA] = randomNumberGenerator(min, max);
            }
         }
      }
   }

   /**
    * Hard codes the value of the weights in order to test the neural network. 
    * For this case, the weights correspond to the weights for OR in the spreadsheet.
    * Currently, this method is not being used. It is here for debugging reasons only. 
    */
   public void setWeights()
   {
      weights[0][0][0] = 0.49;
      weights[0][0][1] = 0.49;
      weights[0][1][0] = 0.49;
      weights[0][1][1] = 0.49;
      weights[1][0][0] = 0.68;
      weights[1][1][0] = 0.68;
   }

   
   /**
    * The propagation function inputs the output vector into the connectivity matrix to determine a net input for each unit. 
    * For this model, the function calculates the net input of the specified node by taking the dot product of all nodes in the previous layer. 
    * 
    * @precondition only calculates values for nodes in the hidden and output layers, not the activation layer. 
    * 
    * @param actRow    the row of the activation node
    * @param actLayer  the layer of the activation node
    * @return the net input for the specified node
    */
   public double propagationFunction(int actLayer, int actRow)
   {
      double activationValue = 0.0;
      
      //System.out.print("Debug: a[" + actLayer + "][" + actRow + "] = ");
      
      for (int i = 0; i < activations[actLayer-1].length; i++)         //Iterates over the nodes in the previous layer
      {
         activationValue += (activations[actLayer-1][i] * weights[actLayer-1][i][actRow]);
         
         //System.out.print("a[" + (actLayer-1) + "][" + i + "]w[" + (actLayer-1) + "][" + i + "][" + actRow + "] + ");  //Debug: checks for indices problems
      }
      
      //System.out.println();                                                                                            //Debug
      
      return activationValue;
   }
   
   
   /**
    * The activation function combines the inputs into the unit and 
    * the current activation state of the unit to produce a new activation level for the the unit. 
    * Since, for this model, there is no activation function, the net input is just returned.
    * 
    * @param netInput the net input into the node
    * @return the new value for the unit
    */
   public double activationFunction(double netInput)
   {
      return (1.0/(1.0 + Math.exp(-netInput)));
   }
   
   /**
    * The derivative of the activationFunction
    */
   public double derivative(double netInput)
   {
      double activationValue = activationFunction(netInput);
      return activationValue*(1.0 - activationValue);
   }
   
   /**
    * Calculates the final value of the node using the both propagation and activation function.
    * Was used in a previous version, but is not being used currently.
    * 
    * @param actLayer  the layer of the activation node
    * @param actRow    the row of the activation node
    * @return the final value of the node
    */
   public double calculateActivationValue(int actLayer, int actRow)
   {
      double netInput = propagationFunction(actLayer, actRow);
      activations[actLayer][actRow] = activationFunction(netInput);
      return activations[actLayer][actRow];
   }
   
   
   /**
    * Calculates the values of the output nodes. 
    * First, uses the propagation function to determine the value of the hidden nodes. 
    * Then, uses the propagation function and hidden nodes' values to determine the value for the output nodes. 
    * 
    * @return an array containing all the output values
    */
   public double[] calculateOutputs() 
   {
      //Iterates over the nodes in the hidden and output layers
      for (int layers = 1; layers < activations.length; layers++)                  
      {
         for (int rows = 0; rows < activations[layers].length; rows++)   
         { 
            double netInput = propagationFunction(layers, rows);
           
            Θvalues[layers][rows] = netInput;

            activations[layers][rows] = activationFunction(netInput);         //Uses the propagation function to define values for all non-input nodes
         }
      }
      
      return activations[activations.length-1];           //Returns all output values 
   }
   
   
   /**
    * Uses the error function to calculate the error of the specific case.
    * E = (1/2)(T - F)^2
    * where T is the exact/training value and F is the calculated value.
    * 
    * @param calculatedAns   the answer calculated using the neural network
    * @param trainingVal     the exact answer 
    * @return the error value for the specific case (inputs)
    */
   public double calculateIndividualError(double calculatedAns, double trainingVal)
   {
      return ((0.5) * ((trainingVal - calculatedAns) * (trainingVal - calculatedAns)));
   }
   
   
   /**
    * Calculates the total error of the network by combining all the individual errors.
    * Total Error = sqrt(E1^2 + E2^2 + ... + EN^2)
    * where N is the total number of individual errors.
    * 
    * @param calculatedErrors   an array containing all the individual errors
    * @return the total error for the neural network
    */
   public double calculateTotalError(double[] calculatedErrors)
   {
      double totalError = 0.0;
      
      for (int i = 0; i < calculatedErrors.length; i++)       //Iterates over the individual errors
      {
         totalError += (calculatedErrors[i] * calculatedErrors[i]);
      }
      
      return Math.sqrt(totalError);
   }
   
   /**
    * Resets the weights to their previous values. 
    * Not currently being used as weight rollback is not implemented.
    * 
    * @param oldWeights the values which the weights are being reset to
    */
   public void rollBackWeights(double[][][] oldWeights)
   {
      for (int n= 0; n < weights.length; n++)
      {
         for (int leftA= 0; leftA < weights[n].length; leftA++)
         {
            for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
            {
               weights[n][leftA][rightA] = oldWeights[n][leftA][rightA];
            }
         }
      }
   }
   
   /**
    * Saves the current weights in a new array. 
    * Not currently being used as weight rollback is not implemented.
    * 
    * @param oldWeights the array in which the weights are being saved. 
    */
   public void saveWeights(double[][][] oldWeights)
   {
      for (int n= 0; n < weights.length; n++)
      {
         for (int leftA= 0; leftA < weights[n].length; leftA++)
         {
            for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
            {
               oldWeights[n][leftA][rightA] = weights[n][leftA][rightA];
            }
         }
      }
   }
   
   /**
    * Uses back propagation to alter weights and minimize the individual errors for test cases. 
    * Training terminates once the maximum individual error is below a given threshold, the iteration number is higher than 
    * a given threshold, the change in error is below a given threshold, or the learning constant is below a given threshold.
    * 
    * @param errorThreshold         the max value the error can be before the network stops training
    * @param iterationThreshold     the max amount of iterations before the network stops training
    * @param changeErrorThreshold   the max value the change in error can be before the network stops training
    * @param trainingVal            the training value for the network
    * @param inputValues            the input values for the network
    */
   public void steepestDescentTraining(double errorThreshold, int iterationThreshold, double changeErrorThreshold, 
         double[][] trainingValues, double[][] inputValues, double initialError, double learningFactor, double growthFactor)
   {
      double currentError = initialError;    
      int numberIterations = 0;
      double deltaError = 1.0;
      double previousError = currentError;              //previous total error for entire network
      double lambda = learningFactor;
      double maxError = initialError;                   //the highest error for among the individual test cases
      int done = 0;
      double individualErrors[] = new double[numOutputs];
      
      while (done == 0) 
      {
         maxError = 0.0;
         
         for (int testCases = 0; testCases < trainingValues.length; testCases++)
         {
            
            deltaError = Math.abs(currentError - previousError);
         
            //Calculates the delta weights for each test case
            setInputValues(inputValues[testCases]);
            calculateOutputs();
            
            
            //double Ω[][] = new double[numLayers][numRows];
            double Ωj[] = new double[numRows];
            double Ωk[] = new double[numRows];
            
            //Calculates the delta weight values for the output layer for this test case
            for (int j = 0; j < activations[numLayers-2].length; j++)
            {
               for (int i = 0; i < numOutputs; i++)
               {
                  double Θi = Θvalues[numLayers-1][i];
                  
                  double ωi = (trainingValues[testCases][i] - activations[numLayers-1][i]);
                  
                  double ψi = ωi * derivative(Θi); 
                  
                  weights[numLayers-2][j][i] += (lambda) * activations[numLayers-2][j] * ψi;       //third-layer weights
                  
                  Ωj[j] += ψi * weights[numLayers-2][j][i];
               }
            }
            
            
            //Calculates the delta weight values for the second hidden-layer for this test case
            for (int k = 0; k < activations[1].length; k++)
            {
               for (int j = 0; j < activations[numLayers-2].length; j++)
               {
                  double Θj = Θvalues[numLayers-2][j];

                  double Ψj = Ωj[j] * derivative(Θj);
                  
                  weights[1][k][j] += (lambda) * activations[1][k] * Ψj;                  //second-layer weights 
                  
                  Ωk[k] += Ψj * weights[1][k][j];
               } 
            }
            
            //Calculates the delta weight values for the second hidden-layer for this test case
            for (int m = 0; m < numInputs; m++)
            {
               for (int k = 0; k < activations[1].length; k++)
               {
                  double Θk = Θvalues[1][k];

                  double Ψk = Ωk[k] * derivative(Θk);
                  
                  weights[0][m][k] += (lambda) * activations[0][m] * Ψk;                  //first-layer weights    
               } 
            }
            
            
            calculateOutputs();
            
            for (int i = 0; i < numOutputs; i++)
            {
               individualErrors[i] = calculateIndividualError(activations[2][i], trainingValues[testCases][i]);
            }
            
            currentError = calculateTotalError(individualErrors); 
            
            if (currentError > maxError)
            {
               maxError = currentError;
            }
            
            if (currentError >= previousError)          
            {
               lambda /= growthFactor;
            }
            else
            {
               lambda *= growthFactor;
            }
            
            previousError = currentError;
            
            numberIterations++; 
         
         } //for (int testCases = 0; testCases < trainingValues.length; testCases++)
         
         //Prints out the reason for why the network stopped training
         if (maxError < errorThreshold)
         {
            done = 1;
            System.out.println("reason: Max error is less than " + errorThreshold);
         }
         else if (numberIterations > iterationThreshold)
         {
            done = 2;
            System.out.println("reason: Iterations are greater than " + iterationThreshold);
         }
         else if (deltaError < changeErrorThreshold)
         {
            done = 3;
            System.out.println("reason: Delta error is greater than " + changeErrorThreshold);
         }
         else if (lambda == 0.0)
         {
            done = 4;
            System.out.println("reason: Learning factor is greater than " + 0);
         }
        
      } //while (currentError > errorThreshold && numberIterations < iterationThreshold && deltaError < changeErrorThreshold && lamda != 0)
      
      System.out.println("lambda " + lambda);
      System.out.println("iterations " + numberIterations);
      System.out.println("error threshold " + errorThreshold);
      System.out.println("max error " + maxError);
   }

   
   /**
    * Runs and trains the the entire perceptron. 
    * 
    * @param args the arguments passed through the main method
    */
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner sc = new Scanner(System.in);
      System.out.println("Configuration File:");                           
      String fileName = sc.next();
      
      //Reads in infromation from the Configuration File
      Scanner sc1 = new Scanner(new FileReader(fileName));
      int activations = sc1.nextInt();
      int outputs = sc1.nextInt();
      int[] hiddenLayers = new int[sc1.nextInt()];
      
      for (int i = 0; i < hiddenLayers.length; i++)
      {
         hiddenLayers[i] = sc1.nextInt();
      }
      
      sc1.nextLine();
      sc1.nextLine();
      
      int numTestCases = sc1.nextInt();
      double[][] inputValues = new double[numTestCases][activations];
      double[][] trainingValues = new double[numTestCases][outputs];
      sc1.nextLine();
      
      for (int testCase = 0; testCase < numTestCases; testCase++)          //reads in the input values
      {
         for (int k = 0; k < activations; k++)
         {
            if (sc1.hasNextDouble())
            {
               inputValues[testCase][k] = sc1.nextDouble();
            }
         }
         if (sc1.hasNextLine())
         {
            sc1.nextLine();
         }
      }
      
      for (int testCase = 0; testCase < numTestCases; testCase++)         //reads in the training values
      {
         for (int k = 0; k < outputs; k++)
         {
            if (sc1.hasNextDouble())
            {
               trainingValues[testCase][k] = sc1.nextDouble();
            }
         }
         if (sc1.hasNextLine())
         {
            sc1.nextLine();
         }
      }

      sc1.nextLine();
      
      double errorThreshold = sc1.nextDouble();
      int iterationThreshold = sc1.nextInt();
      double changeErrorThreshold = sc1.nextDouble();
      double learningFactor = sc1.nextDouble();
      double growthFactor = sc1.nextDouble();
      
      sc1.nextLine();
      sc1.nextLine();
      
      double min = sc1.nextDouble();            //reads in the min and max for weight randomization 
      double max = sc1.nextDouble();
      
      MultilayerPerceptron tempPerceptron = new MultilayerPerceptron(activations, outputs, hiddenLayers);
      
      double[][] errors = new double[trainingValues.length][tempPerceptron.numOutputs];
      
      //Sets up the network for steepest descent training
      tempPerceptron.randomizeWeightValues(min, max);
      
      for (int testCases = 0; testCases < trainingValues.length; testCases++)
      {
         for (int i = 0; i < tempPerceptron.numOutputs; i++)
         {
            tempPerceptron.setInputValues(inputValues[testCases]);
            tempPerceptron.calculateOutputs();
            errors[testCases][i] = tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][i], trainingValues[testCases][i]);
         }
      }
      
      double[] initialIndividualErrors = new double[tempPerceptron.numOutputs];
            
      for (int i = 0; i < tempPerceptron.numOutputs; i++)
      {
         initialIndividualErrors[i] = tempPerceptron.calculateTotalError(errors[i]);
      }
      
      double initialError = tempPerceptron.calculateTotalError(initialIndividualErrors);  //initial error of the network using the randomized weights
      
      //Network gets trained using the steepest descent algorithm 
      tempPerceptron.steepestDescentTraining(errorThreshold, iterationThreshold, changeErrorThreshold, trainingValues, inputValues, 
            initialError, learningFactor, growthFactor);
      
      //Prints out diagnostic information about the trained network
      for (int testCases = 0; testCases < trainingValues.length; testCases++)
      {
         for (int i = 0; i < tempPerceptron.numOutputs; i++)
         {
            tempPerceptron.setInputValues(inputValues[testCases]);
            tempPerceptron.calculateOutputs();
            System.out.println();
            System.out.println("Case " + (testCases+1) + ", Output " + (i+1) + " :");
            System.out.println("expected " + trainingValues[testCases][i]);
            System.out.println("output: " + tempPerceptron.activations[2][i]);
            System.out.println("error: " + tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][i], trainingValues[testCases][i]));
         }
      }

   }
   
   
}
