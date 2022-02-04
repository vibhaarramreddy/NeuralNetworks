import java.util.Scanner;

/**
 * @author Vibha Arramreddy
 * September 4th 2019
 * Building a simple, generic multilayer perceptron
 */

public class MultilayerPerceptron 
{

   private double[][] activations;   
   private double[][][] weights;
   private int numLayers;
   private int numRows;
	

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
      
            
      activations = new double[numLayers][];                                 //The array containing all the node values
      weights = new double[(numLayers-1)][numRows][numRows];                 //The array containing all the weight values
		   
      activations[0] = new double[numInputs];                                 //Defines the number or nodes/rows in the activation layer
      activations[numLayers-1] = new double[numOutputs];                      //Defines the number of nodes/rows in the output layer
      for (int i = 0; i < numHidden.length; i++)                              //Loop that defines all of the hidden layers
      {
         activations[i+1] = new double[numHidden[i]];
      }
      
      randomizeWeightValues();
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
   public void randomizeWeightValues()
   {
      weights[0][0][0] = Math.random();
      weights[0][0][1] = Math.random();
      weights[0][1][0] = Math.random();
      weights[0][1][1] = Math.random();
      weights[1][0][0] = Math.random();
      weights[1][1][0] = Math.random();
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
      return (1/(1 - Math.exp(netInput)));
   }
   
   /**
    * 
    */
   public double derivative(double netInput)
   {
      return activationFunction(netInput)*(1 - activationFunction(netInput));
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
    * 
    * @return
    */
   public double runPerceptron(double[] inputVals, double trainingVal)
   {                           
      double[][] inputValues = {inputVals};                                     //The input values for the OR function                                   
      
      setInputValues(inputValues[0]);
      double[] output = calculateOutputs();
      double error = calculateIndividualError(output[0], trainingVal);
      
      return error;
   }
   
   /**
    * 
    */
   public void steepestDescentTraining(double errorThreshold, int iterationThreshold, double changeErrorThreshold, 
         double trainingVal, double[] inputValues)
   {
      double currentError = 0;
      int numberIterations = 0;
      double deltaError = 1;
      double previousError = -1;
      double lamda = 0.1;
      double[][][] oldWeights = new double[numLayers-1][numRows][numRows];
      
      setInputValues(inputValues);
      calculateOutputs();
      currentError = calculateIndividualError(activations[2][0], trainingVal);
      
      while (currentError > errorThreshold && numberIterations < iterationThreshold && deltaError >= changeErrorThreshold && lamda != 0.0) 
      {
         deltaError = Math.abs(currentError - previousError);
 
         //Saves the old weights in an new array
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
         
         //Calculates the new weight values
         for (int n= 0; n < weights.length; n++)
         {
            for (int leftA= 0; leftA < weights[n].length; leftA++)
            {
               for (int rightA = 0; rightA < weights[n][leftA].length; rightA++)
               {
                  if (n == 1)
                  {
                     weights[1][leftA][0] += (-lamda)*(-1)*(trainingVal - activations[2][0])*(derivative(propagationFunction(2, 0)))*(activations[1][leftA]);
                  }
                  if (n == 0)
                  {
                     weights[0][leftA][rightA] += (-lamda)*(-1)*(activations[0][leftA])*(derivative(propagationFunction(1, leftA)))*(trainingVal - activations[2][0])
                           *(derivative(propagationFunction(2, 0)))*(weights[1][rightA][0]);
                  }
               }
            }
         }
         
         if (currentError < previousError)
         {
            lamda *= 2;
         }
         else
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
            
            lamda /= 2;
         }
         
         previousError = currentError;
         
         calculateOutputs();
         currentError = calculateIndividualError(activations[2][0], trainingVal);
         numberIterations++; 
         
         System.out.println("error " + currentError);
         System.out.println("lamda " + lamda);
      } //while (currentError > errorThreshold && numberIterations < iterationThreshold && deltaError < changeErrorThreshold && lamda != 0)
      
      System.out.println(numberIterations);
   }
   
   /**
    * Runs the the entire perceptron. 
    * In this case, it creates a perceptron for the OR function (from the spreadsheet).
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
		
      //tempPerceptron.setWeights();                                      //Sets the predetermined weights
      
      double[] trainingValues = new double[outputs];
      double[] inputValues = new double[activations];
      
      //for(int j = 0; j < 4; j++)                                          //runs 4 test cases
      
      for (int i = 0; i < activations; i++)
      {
         System.out.println("Input Value " + (i+1) + " (Double): ");
         inputValues[i] = sc1.nextDouble();
      }
      
      for (int i = 0; i < outputs; i++)
      {
         System.out.println("Training Value " + (i+1) + " (Double): ");
         trainingValues[i] = sc1.nextDouble();
      }
      
      //tempPerceptron.setInputValues(inputValues);
      //tempPerceptron.calculateOutputs();
      //System.out.println(tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][0], trainingValues[0])); 
      
      tempPerceptron.steepestDescentTraining(0.01, 2000, 0.0, trainingValues[0], inputValues);
      //System.out.println(tempPerceptron.activations[2][0]);
      System.out.println(tempPerceptron.calculateIndividualError(tempPerceptron.activations[2][0], trainingValues[0]));
      
      for (int n= 0; n < tempPerceptron.weights.length; n++)
      {
         for (int leftA= 0; leftA < tempPerceptron.weights[n].length; leftA++)
         {
            for (int rightA = 0; rightA < tempPerceptron.weights[n][leftA].length; rightA++)
            {
               System.out.println(tempPerceptron.weights[n][leftA][rightA]);
            }
         }
      }
   }
	
}
