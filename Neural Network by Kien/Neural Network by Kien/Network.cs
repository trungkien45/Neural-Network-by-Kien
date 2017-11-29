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
            private List<Dendrite> dendrites;
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
                get;set;
            }
            public double Sum { get; set; }
            public List<Dendrite> Dendrites { get { return dendrites; } }
            public Dendrite this[int index]
            {
                get
                {
                    return Dendrites[index];
                }
            }

            public Neuron(int nDendritesToAfter)
            {
                dendrites = new List<Dendrite>();
                for (int i = 0; i < nDendritesToAfter; i++)
                {
                    Dendrites.Add(new Dendrite());
                }

            }

            public Neuron(double v, int nDendritesToAfter = 0)
            {
                dendrites = new List<Dendrite>();
                for (int i = 0; i < nDendritesToAfter; i++)
                {
                    Dendrites.Add(new Dendrite());
                }

                _value = v;
            }
        }

        private class Layer
        {
            private List<Neuron> neurons;
            public double Bias { get; private set; }
            public List<Neuron> Neurons { get { return neurons; } }

            public Neuron this[int index]
            {
                get
                {
                    return Neurons[index];
                }
            }
            /// <summary>
            /// Value for each neuron = 0
            /// </summary>
            /// <param name="nNeurons"></param>
            /// <param name="nafter"></param>
            public Layer(int nNeurons, int nafter, double bias)
            {
                neurons = new List<Neuron>();
                for (int i = 0; i < nNeurons; i++)
                {
                    Neurons.Add(new Neuron(nafter));
                }
                Bias = bias;
            }
            public Layer(double[] v, int nafter, double bias)
            {
                neurons = new List<Neuron>();
                for (int i = 0; i < v.Length; i++)
                {
                    Neurons.Add(new Neuron(v[i], nafter));
                }
                Bias = bias;

            }
            private static double getRandom(double MinValue, double MaxValue)
            {
                return Util.GetRandomD() * (MaxValue - MinValue) + MinValue;
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
            AdjustNetwork(outputRUN,output);    
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
            double deltaoutnet = 0;
            //Output Layer
            for (int j = 0; j < layers[layers.Count - 1].Neurons.Count; j++)
            {
                //calc delta
                deltaoutnet = Util.GetDerivativeAtX(activationFuntion, layers[layers.Count - 1].Neurons[j].Sum);
                layers[layers.Count - 1].Neurons[j].delta = deltaoutnet;
                for (int i = 0; i < layers[layers.Count - 2].Neurons.Count; i++)
                {
                    double DeltaW = -LearnningRate * layers[layers.Count - 2].Neurons[i].Value * layers[layers.Count - 1].Neurons[j].delta;
                    layers[layers.Count - 2].Neurons[i].Dendrites[j].Weight += DeltaW; 
                }
            }
            for (int k = layers.Count-2; k > 1; k--)
            {
                for (int j = 0; j < layers[k].Neurons.Count; j++)
                {
                    double deltaL = 0;
                    //calc delta at layerL
                    for (int i = 0; i < Layers[k+1].Neurons.Count; i++)
                    {
                        deltaL += layers[k + 1].Neurons[i].delta*layers[k].Neurons[j].Dendrites[k+1].Weight;
                    }
                    deltaL *= Util.GetDerivativeAtX(activationFuntion, layers[k].Neurons[j].Sum);
                    layers[k].Neurons[j].delta = deltaL;
                    //update Weight
                    for (int i = 0; i < layers[k-1].Neurons.Count; i++)
                    {
                        double DeltaW = -LearnningRate * layers[k-1].Neurons[i].Value * layers[k].Neurons[j].delta;
                        layers[k-2].Neurons[j].Dendrites[i].Weight += DeltaW;
                    }
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

            for (int k = 0; k < Layers.Count - 1; k++)
            {
                Layer layerbefore = layers[k];
                Layer layerafter = layers[k + 1];
                for (int j = 0; j < layerafter.Neurons.Count; j++)
                {
                    for (int i = 0; i < layerbefore.Neurons.Count; i++)
                    {
                        layerafter.Neurons[j].Value += layerbefore.Neurons[i].Value * layerbefore.Neurons[i].Dendrites[j].Weight;
                    }
                    //adjust by activation Funtion
                    layerafter.Neurons[j].Sum = layerafter.Neurons[j].Value;
                    layerafter.Neurons[j].Value = ActivationFuntion(layerafter.Neurons[j].Sum + layerafter.Bias);
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
            this.LearnningRate = learningrate;
            for (int i = 0; i < nNeuronsInLayers.Count; i++)
            {
                if (nNeuronsInLayers[i] < 1)
                    throw new IndexOutOfRangeException("The number of neurons in a layer must be greater than 0");
                int nafter = 0;
                if (i != nNeuronsInLayers.Count - 1)
                    nafter = nNeuronsInLayers[i + 1];
                Layer layer = new Layer(nNeuronsInLayers[i], nafter, Util.GetRandom(0.00000001, 1.0));
                layers.Add(layer);
            }
        }
    }
}
