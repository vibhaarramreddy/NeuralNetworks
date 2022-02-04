import java.util.Scanner;

/**
 * @author Vibha Arramreddy
 * September 4th 2019
 * This class creates a simple multilayer perceptron that has three activation layers. 
 * The number of activations in each layer is user configurable
 */

public class MultilayerPerceptron 
{

   private double[][] activations;   
   private double[][][] weights;
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
    * Randomizes the weight values for the perceptron.
    */
   public void randomizeWeightValues(double min, double max)
   { 
      
      double range = max - min;
      
      for (int n= 0; n < weights.length; n++)
      {
         for (int leftA= 0; leftA < weights[n].length; leftA++)
         {
            for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
            {
               weights[n][leftA][rightA] = ((Math.random() * range) + min);
            }
         }
      }
   }
   
   
   
   /**
    * Hard codes the value of the weights in order to test the neural network. 
    * For this case, the weights correspond to the weights for OR in the spreadsheet. 
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
      
      //activations[actLayer][actRow] = activationValue; 
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
      return activationFunction(netInput)*(1.0 - activationFunction(netInput));
   }
   
   /**
    * Calculates the final value of the node using the both propagation and activation function.
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
            activations[layers][rows] = calculateActivationValue(layers, rows);         //Uses the propagation function to define values for all non-input nodes
         }
      }
		
      return activations[activations.length-1];                                        //Returns all output values 
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
    * Alters weights to minimize the individual errors for test cases. Training terminates once the
    * maximum individual error is below a given threshold, the iteration number is higher than 
    * a given threshold, the change in error is below a given threshold, or the learning constant is below a given threshold.
    * 
    * @param errorThreshold         the max value the error can be before the network stops training
    * @param iterationThreshold     the max amount of iterations before the network stops training
    * @param changeErrorThreshold   the max value the change in error can be before the network stops training
    * @param trainingVal            the training value for the network
    * @param inputValues            the input values for the networ
    */
   public void steepestDescentTraining(double errorThreshold, int iterationThreshold, double changeErrorThreshold, 
         double[] trainingValues, double[][] inputValues, double initialError, double learningFactor)
   {
      double currentError = initialError;    
      int numberIterations = 0;
      double deltaError = 1.0;
      double previousError = currentError;              //previous total error for entire network
      double lambda = learningFactor;
      double growthFactor = 1.0;
      double maxError = initialError;                   //the highest error for among the individual test cases
      int done = 0;
         
      while (done == 0) 
      {
         maxError = 0.0;
         
         for (int testCases = 0; testCases < trainingValues.length; testCases++)
         {
            
            deltaError = Math.abs(currentError - previousError);
 
            //Saves the old weights in an new array
            //saveWeights(oldWeights);
         
            //Calculates the delta weights for each test case
            setInputValues(inputValues[testCases]);
            calculateOutputs();
            
            /**
            System.out.println();
            System.out.println("weight values: ");
            for (int n= 0; n < weights.length; n++)
            {
               for (int leftA= 0; leftA < weights[n].length; leftA++)
               {
                  for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
                  {
                     System.out.println("w[" + n + "][" + leftA + "][" + rightA + "]: " + weights[n][leftA][rightA]);
                  }
               }
            }
            System.out.println();
            */
            
            //Calculates the delta weight values for this test case   
            for (int leftA = 0; leftA < numInputs; leftA++)
            {
               for (int rightA = 0; rightA < activations[leftA].length; rightA++)
               {
                  
                  double firstLayerSum = 0.0;
                  
                  for (int k = 0; k < activations[0].length; k++)
                  {
                     firstLayerSum += activations[0][k] * weights[1][k][rightA];
                  }
                  
                  double output = activations[2][0];
                  double secondLayerSum = 0.0;
                  
                  for (int j = 0; j < activations[1].length; j++)
                  {
                     secondLayerSum += activations[1][j] * weights[1][j][0];
                  }
                  
                  double deltaWeight = lambda * activations[0][leftA] * derivative(firstLayerSum);
                  
                  deltaWeight *= (trainingValues[testCases] - output) * derivative(secondLayerSum);
                  
                  deltaWeight *= weights[1][rightA][0];
                  
                  weights[0][leftA][rightA] += deltaWeight;
                  
               }
            }
            
            for (int leftA = 0; leftA < activations[1].length; leftA++)
            {
                  double output = activations[2][0];
                  double secondLayerSum = 0.0;
                  
                  for (int j = 0; j < activations[1].length; j++)
                  {
                     secondLayerSum += activations[1][j] * weights[1][j][0];
                  }
                  
                  double deltaWeight = lambda * (trainingValues[testCases] - output);
                  deltaWeight *= derivative(secondLayerSum) * activations[1][leftA];
                  
                  weights[1][leftA][0] += deltaWeight;
            }
            
            
            
            calculateOutputs();
            
            currentError = calculateIndividualError(activations[2][0], trainingValues[testCases]);
            
            if (currentError > maxError)
            {
               maxError = currentError;
            }
            
            if (currentError >= previousError)            //rolls back wieghts if the error increases
            {
               //rollBackWeights(oldWeights);
            
               lambda /= growthFactor;
            }
            else
            {
               lambda *= growthFactor;
            }
            
            previousError = currentError;
            
            numberIterations++; 
         
         }
         
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
   public static void main(String[] args) 
   {
      Scanner sc1 = new Scanner(System.in);
      System.out.println("Number of activation nodes: ");
      int activations = sc1.nextInt(); 
      System.out.println("Number of output nodes: ");
      int outputs = sc1.nextInt();
      
      System.out.println("Number of hidden layers: ");
      int[] hiddenLayers = new int[sc1.nextInt()];
      
      for (int i = 0; i < hiddenLayers.length; i++)
      {
         System.out.println("Number of nodes in hidden layer " + (i+1) + ": ");
         hiddenLayers[i] = sc1.nextInt();
      }
      
      MultilayerPerceptron tempPerceptron = new MultilayerPerceptron(activations, outputs, hiddenLayers);
      
      double[][] inputValues = {{0,0}, {0,1}, {1,0}, {1,1}};     
      double[] trainingValues = {0, 0, 0, 1};
      double[] errors = new double[trainingValues.length];
      
      tempPerceptron.randomizeWeightValues(-1, 1);
      
      for (int testCases = 0; testCases < trainingValues.length; testCases++)
      {
         tempPerceptron.setInputValues(inputValues[testCases]);
         tempPerceptron.calculateOutputs();
         errors[testCases] = tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][0], trainingValues[testCases]);
      }
      
      double initialError = tempPerceptron.calculateTotalError(errors);
      
      tempPerceptron.steepestDescentTraining(0.001, 10000000, 0.0, trainingValues, inputValues, initialError, 0.1);
      
      for (int testCases = 0; testCases < trainingValues.length; testCases++)
      {
         tempPerceptron.setInputValues(inputValues[testCases]);
         tempPerceptron.calculateOutputs();
         System.out.println();
         System.out.println("Case " + (testCases+1) + " :");
         System.out.println("expected " + trainingValues[testCases]);
         System.out.println("output: " + tempPerceptron.activations[2][0]);
         System.out.println("error: " + tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][0], trainingValues[testCases]));
      }

   }
   
	
}
