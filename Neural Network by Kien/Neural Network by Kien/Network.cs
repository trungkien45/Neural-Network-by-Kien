using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    public class Network
    {
        private class Neuron
        {
            private readonly List<Dendrite> dendrites;
            private double _value = 0;
            public double Value
            {
                get
                { return _value; }
                set
                { _value = value; }
            }
            public double delta
            {
                get; set;
            }
            public double Bias { get; set; }
            public double Sum { get; set; }
            public List<Dendrite> Dendrites { get { return dendrites; } }
            public Dendrite this[int index]
            {
                get
                {
                    return Dendrites[index];
                }
            }

            public Neuron(int nDendrites)
            {
                dendrites = new List<Dendrite>();
                for (int i = 0; i < nDendrites; i++)
                {
                    Dendrites.Add(new Dendrite());
                }
                Bias = Util.GetRandom(0, 1);
            }

            public Neuron(double v, int nDendrites = 0)
            {
                dendrites = new List<Dendrite>();
                for (int i = 0; i < nDendrites; i++)
                {
                    Dendrites.Add(new Dendrite());
                }

                _value = v;
                Bias = Util.GetRandom(0, 1);
            }
        }

        private class Layer
        {
            private List<Neuron> neurons;
            //public double Bias { get; private set; }
            public List<Neuron> Neurons { get { return neurons; } }

            public Neuron this[int index]
            {
                get
                {
                    return Neurons[index];
                }
            }
            public Layer(int nNeurons, int n)
            {
                neurons = new List<Neuron>();
                for (int i = 0; i < nNeurons; i++)
                {
                    Neurons.Add(new Neuron(n));
                }
                //Bias = bias;
            }
            public Layer(double[] v, int n)
            {
                neurons = new List<Neuron>();
                for (int i = 0; i < v.Length; i++)
                {
                    Neurons.Add(new Neuron(v[i], n));
                }
                //Bias = bias;

            }
        }

        private class Dendrite
        {
            private double weight;
            public double Weight
            {
                get { return weight; }
                set { weight = value; }
            }

            //Provide a constructor for the class.
            //It is always better to provide a constructor instead of using
            //the compiler provided constructor
            public Dendrite()
            {
                weight = getRandom(0.00000001, 1.0);
            }

            private static double getRandom(double MinValue, double MaxValue)
            {
                return Util.GetRandomD() * (MaxValue - MinValue) + MinValue;
            }
        }

        List<Layer> layers;
        public delegate double ActivationFuntionDelegate(double x);
        ActivationFuntionDelegate activationFuntion;
        public double LearnningRate { get; private set; }
        public ActivationFuntionDelegate ActivationFuntion { get { return activationFuntion; } }

        private List<Layer> Layers { get { return layers; } set { layers = value; } }
        public void Train(double[] input, double[] output)
        {
            if (input.Length != layers[0].Neurons.Count)
            {
                throw new IndexOutOfRangeException("The number of neurons in the first layer must be equal the number of input");
            }
            if (output.Length != layers[layers.Count - 1].Neurons.Count)
            {
                throw new IndexOutOfRangeException("The number of neurons in the last layer must be equal the number of output");
            }
            List<double> outputRUN = Run(input);
            AdjustNetwork(outputRUN, output);
        }

        private void AdjustNetwork(List<double> outputRUN, double[] output)
        {
            //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
            double error = 0;
            //Calc Total errors

            for (int j = 0; j < layers[layers.Count - 1].Neurons.Count; j++)
            {
                double delta = output[j] - outputRUN[j];
                error += Math.Pow(delta, 2);

            }
            error /= 2;
            ///////////////////////////////////////
            for (int i = 0; i < Layers[Layers.Count - 1].Neurons.Count; i++)
            {
                Neuron neuron = Layers[Layers.Count - 1].Neurons[i];

                neuron.delta = Util.GetDerivativeAtX(activationFuntion,neuron.Sum) * (output[i] - neuron.Value);

            }
            for (int j = Layers.Count - 2; j >= 1; j--)
            {
                for (int k = 0; k < Layers[j].Neurons.Count; k++)
                {
                    Neuron n = Layers[j].Neurons[k];
                    n.delta = 0;

                    for (int i = 0; i < Layers[j + 1].Neurons.Count; i++)
                    {

                        n.delta += Util.GetDerivativeAtX(activationFuntion, n.Sum) *
                                  Layers[j + 1].Neurons[i].Dendrites[k].Weight *
                                  Layers[j + 1].Neurons[i].delta;
                    }
                }
            }

            for (int i = Layers.Count - 1; i >= 1; i--)
            {
                for (int j = 0; j < Layers[i].Neurons.Count; j++)
                {
                    Neuron n = Layers[i].Neurons[j];
                    n.Bias = n.Bias + (LearnningRate * n.delta);

                    for (int k = 0; k < n.Dendrites.Count; k++)
                        n.Dendrites[k].Weight = n.Dendrites[k].Weight + (LearnningRate * Layers[i - 1].Neurons[k].Value * n.delta);
                }
            }
        }
        public List<double> Run(double[] input)
        {
            List<double> result = new List<double>();
            if (input.Length != layers[0].Neurons.Count)
            {
                throw new IndexOutOfRangeException("The number of neurons in the first layer must be equal the number of input");
            }

            for (int i = 0; i < input.Length; i++)
            {
                layers[0].Neurons[i].Value = input[i];
            }

            for (int k = 1; k < Layers.Count; k++)
            {
                Layer layer = layers[k];
                Layer layerbefore = layers[k - 1];
                for (int i = 0; i < layer.Neurons.Count; i++)
                {
                    Neuron neuron = layer[i];
                    for (int j = 0; j < layerbefore.Neurons.Count; j++)
                    {

                        neuron.Value += neuron.Value * neuron.Dendrites[j].Weight;
                    }
                    //adjust by activation Funtion
                    neuron.Sum = neuron.Value;
                    neuron.Value = ActivationFuntion(neuron.Sum + neuron.Bias);
                }
            }
            Layer layerOutput = Layers[Layers.Count - 1];
            for (int k = 0; k < layerOutput.Neurons.Count; k++)
            {
                result.Add(layerOutput.Neurons[k].Value);
            }
            return result;
        }

        public Network(List<int> nNeuronsInLayers, ActivationFuntionDelegate activationFuntion, double learningrate = 1)
        {
            this.activationFuntion = activationFuntion;
            layers = new List<Layer>();
            LearnningRate = learningrate;
            for (int i = 0; i < nNeuronsInLayers.Count; i++)
            {
                if (nNeuronsInLayers[i] < 1)
                    throw new IndexOutOfRangeException("The number of neurons in a layer must be greater than 0");
                int nbefore = 0;
                if (i > 0)
                    nbefore = nNeuronsInLayers[i - 1];
                Layer layer = new Layer(nNeuronsInLayers[i], nbefore);
                layers.Add(layer);
            }
        }
    }
}
