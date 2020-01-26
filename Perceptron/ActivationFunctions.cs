using System;

namespace ActivationFunctionForMatrix
{
    public enum TActivationFunction
    {
        Sigmoid = 1,
        HyperbolicTangent = 2,

        /// <summary>
        /// Exponential Linear Units
        /// </summary>
        ELU = 3,

        /// <summary>
        /// Rectified Linear Units (ReLU)
        /// </summary>
        ReLU = 4
    }

    /// <summary>
    /// Interface for all activation functions
    /// </summary>
    public interface IActivationFunctionForMatrix
    {

        /// <summary>
        /// Activation function
        /// </summary>
        float Activation(float x, float gain, float center);

        float Derivative(float x, float gain);
    }

    /// <summary>
    /// Implements f(x) = Sigmoid
    /// </summary>
    public class SigmoidFunction : IActivationFunctionForMatrix
    {

        public float Activation(float x, float gain, float center)
        {
            // gain should be always 1 here, to respect the rightness of the derivative
            double y = 1 / (1 + Math.Exp(-(x - center)));
            float yf = (float)y;
            return yf;
        }

        public float Derivative(float x, float gain)
        {
            float y = x * (1 - x);
            return y;
        }
    }

    /// <summary>
    /// Implements f(x) = Hyperbolic Tangent
    /// </summary>
    public class HyperbolicTangentFunction : IActivationFunctionForMatrix
    {

        public float Activation(float x, float gain, float center)
        {
            // gain should be always 1 here, to respect the rightness of the derivative
            float xc = x - center;
            double y = 2 / (1 + Math.Exp(-2 * xc)) - 1;
            float yf = (float)y;
            return yf;
        }

        public float Derivative(float x, float gain)
        {
            float y = 1 - (x * x);
            return y;
        }
    }

    /// <summary>
    /// Implements f(x) = Exponential Linear Unit (ELU)
    /// </summary>
    public class ELUFunction : IActivationFunctionForMatrix
    {

        public float Activation(float x, float gain, float center)
        {
            float xc = (x - center);
            float yf;
            if (xc >= 0)
                yf = xc;
            else
            {
                double y = gain * (Math.Exp(xc) - 1);
                yf = (float)y;
            }

            return yf;
        }

        public float Derivative(float x, float gain)
        {
            if (gain < 0) return 0;

            float y;
            if (x >= 0)
                y = 1;
            else
                y = x + gain;

            return y;
        }
    }

    /// <summary>
    /// Implements Rectified Linear Unit (ReLU)
    /// </summary>
    public class ReluFunction : IActivationFunctionForMatrix
    {

        public float Activation(float x, float gain, float center)
        {
            float xc = x - center;
            return Math.Max(xc * gain, 0);
        }

        public float Derivative(float x, float gain)
        {
            if (x >= 0) return gain;
            return 0;
        }
    }
}