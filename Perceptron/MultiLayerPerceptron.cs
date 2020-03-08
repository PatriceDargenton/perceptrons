using System;
using System.Collections.Generic;

namespace Perceptron {
	internal class MultiLayerPerceptron {
		public static Random rng = new Random();
		public Matrix weights_ih;
		public Matrix weights_ho;
		public Matrix bias_h;
		public Matrix bias_o;
        private float learningRate;

        /// <summary>
        /// Average error of the output matrix
        /// </summary>
        public float averageError;

        /// <summary>
        /// Activate function of each neuron of the MLP
        /// </summary>
        private ActivationFunctionForMatrix.IActivationFunctionForMatrix activFct;

        /// <summary>
        /// Lambda function for the activation function
        /// </summary>
        private Func<float, float> lambdaFct;

        /// <summary>
        /// Lambda function for the derivative of the activation function
        /// </summary>
        private Func<float, float> lambdaFctD;
                
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNodes"></param>
        /// <param name="hiddenNodes"></param>
        /// <param name="outputNodes"></param>
        public MultiLayerPerceptron(int inputNodes, int hiddenNodes, int outputNodes, float learningRate) {
			this.weights_ih = new Matrix(hiddenNodes, inputNodes);
			this.weights_ho = new Matrix(outputNodes, hiddenNodes);
			this.bias_h = new Matrix(hiddenNodes, 1);
			this.bias_o = new Matrix(outputNodes, 1);
			this.weights_ih.Randomize();
			this.weights_ho.Randomize();
			this.bias_h.Randomize();
			this.bias_o.Randomize();
			this.learningRate = learningRate;

            Func<float, float> lambdaFct = (x) => activFct.Activation(x, gain: 1, center: 0);
            Func<float, float> lambdaFctD = (x) => activFct.Derivative(x, gain: 1);
            SetLambdaActivationFunction(lambdaFct, lambdaFctD);
        }

        /// <summary>
        /// Set specific activation function
        /// </summary>
        public void SetLambdaActivationFunction(
            Func<float, float> lambdaFct, Func<float, float> lambdaFctD)
        {
            this.lambdaFct = lambdaFct;
            this.lambdaFctD = lambdaFctD;
        }

        /// <summary>
        /// Set registered activation function
        /// </summary>
        public void SetActivationFunction(
            ActivationFunctionForMatrix.TActivationFunction fctAct)
        {
            switch (fctAct)
            {
                case ActivationFunctionForMatrix.TActivationFunction.Sigmoid:
                    this.activFct = new ActivationFunctionForMatrix.SigmoidFunction();
                    break;
                case ActivationFunctionForMatrix.TActivationFunction.HyperbolicTangent:
                    this.activFct = new ActivationFunctionForMatrix.HyperbolicTangentFunction();
                    break;
                case ActivationFunctionForMatrix.TActivationFunction.ELU:
                    this.activFct = new ActivationFunctionForMatrix.ELUFunction();
                    break;
                case ActivationFunctionForMatrix.TActivationFunction.ReLU:
                    this.activFct = new ActivationFunctionForMatrix.ReluFunction();
                    break;
                default:
                    this.activFct = null;
                    break;
            }
        }

        /// <summary>
        /// Sigmoid activation function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        // private float sigmoid(float x) {
        //	return (float)(1 / (1 + Math.Exp(-x)));
        //}

        /// <summary>
        /// The derivative of sigmoid.
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        // private float dsigmoid(float y) {
        //	return y * (1 - y);
        //}

        public float[] FeedForward(float[] inputs_array) {

			// Generating the Hidden Outputs
			Matrix inputs = Matrix.FromArray(inputs_array);
			Matrix hidden = Matrix.Multiply(this.weights_ih, inputs);
			hidden.Add(this.bias_h);

			// activation function!
			hidden.Map(this.lambdaFct);

			// Generating the output's output!
			Matrix output = Matrix.Multiply(this.weights_ho, hidden);
			output.Add(this.bias_o);
			output.Map(this.lambdaFct);

			return output.ToArray();
		}

        public void Train(float[] inputs_array, float[] targets_array) {
			// Generating the Hidden Outputs
			Matrix inputs = Matrix.FromArray(inputs_array);
			Matrix hidden = Matrix.Multiply(this.weights_ih, inputs);
			hidden.Add(this.bias_h);

			// activation function!
			hidden.Map(this.lambdaFct);

    		// Generating the output's output!
			Matrix outputs = Matrix.Multiply(this.weights_ho, hidden);
			outputs.Add(this.bias_o);
			outputs.Map(this.lambdaFct);

    		// Convert array to matrix object
			Matrix targets = Matrix.FromArray(targets_array);

			// Calculate the error
			// ERROR = TARGETS - OUTPUTS
			Matrix output_errors = Matrix.Subtract(targets, outputs);
            //this.averageError = Math.Abs(output_errors.Average());
            // Compute first abs then average:
            this.averageError = output_errors.Abs().Average();

            // let gradient = outputs * (1 - outputs);
            // Calculate gradient
            Matrix gradients = Matrix.Map(outputs, this.lambdaFctD);
			gradients.Multiply(output_errors);
			gradients.Multiply(this.learningRate);

			// Calculate hidden -> output delta weights
			Matrix hidden_t = Matrix.Transpose(hidden);
			Matrix weight_ho_deltas = Matrix.Multiply(gradients, hidden_t);

			// Adjust the weights by deltas
			this.weights_ho.Add(weight_ho_deltas);
			// Adjust the bias by its deltas (which is just the gradients)
			this.bias_o.Add(gradients);

    		// Calculate the hidden layer errors
			Matrix weights_ho_t = Matrix.Transpose(this.weights_ho);
			Matrix hidden_errors = Matrix.Multiply(weights_ho_t, output_errors);

			// Calculate hidden gradient
			Matrix hidden_gradient = Matrix.Map(hidden, this.lambdaFctD);
			hidden_gradient.Multiply(hidden_errors);
			hidden_gradient.Multiply(this.learningRate);

			// Calculate input -> hidden delta weights
			Matrix inputs_t = Matrix.Transpose(inputs);
			Matrix weight_ih_deltas = Matrix.Multiply(hidden_gradient, inputs_t);

			this.weights_ih.Add(weight_ih_deltas);
			// Adjust the bias by its deltas (which is just the gradients)
			this.bias_h.Add(hidden_gradient);
		}

        /// <summary>
        /// This is our test function.
        /// </summary>
        /// <returns>The number of correct answers.</returns>
        public float[] Test(float[] inputs) {
			return this.FeedForward(inputs);
		}
	}
}
