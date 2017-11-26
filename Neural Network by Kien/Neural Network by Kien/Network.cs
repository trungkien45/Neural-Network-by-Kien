using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    public class Network
    {
        List<Layer> layers;
        public delegate double ActivationFuntionDelegate(double x);
        ActivationFuntionDelegate activationFuntion;

        public ActivationFuntionDelegate ActivationFuntion { get { return activationFuntion; } }

        public List<Layer> Layers { get { return layers; } set { layers = value; } }
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
            for (int i = layers.Count-1; i > 0; i--)
            {
                if (i == layers.Count - 1)
                {
                    double[] value = new double[layers[i].Neurons.Count];
                    double[] delta = new double[layers[i].Neurons.Count];
                    for (int j = 0; j < layers[i].Neurons.Count; j++)
                    {
                        value[j] = layers[i].Neurons[j].Value;
                        double error = output[j] - outputRUN[j];
                        delta[j] = Util.GetDerivativeAtX(activationFuntion,layers[i].Neurons[j].Value) *(error);
                        double[] w = new double[layers[i-1].Neurons.Count];
                        for (int k = 0; k < layers[i-1].Neurons.Count; k++)
                        {
                            w[k] = layers[i-1].Neurons[k].Dendrites[j].Weight;
                            layers[i - 1].Neurons[k].Dendrites[j].Weight += delta[j] / layers[i - 1].Neurons[k].Dendrites[j].Weight;
                            //Calc delta use value to store delta
                            layers[i - 1].Neurons[k].Value += delta[j] / w[k] * Util.GetDerivativeAtX(activationFuntion,w[k]);
                        }
                    }

                }
                else
                {
                    double[] delta = new double[layers[i].Neurons.Count];

                    for (int j = 0; j < layers[i].Neurons.Count; j++)
                    {
                        double[] w = new double[layers[i - 1].Neurons.Count];
                        for (int k = 0; k < layers[i-1].Neurons.Count; k++)
                        {
                            w[k] = layers[i - 1].Neurons[k].Dendrites[j].Weight;
                            layers[i - 1].Neurons[k].Dendrites[j].Weight += layers[i].Neurons[k].Value / layers[i - 1].Neurons[k].Dendrites[j].Weight;
                            //Calc delta use value to store delta
                            layers[i - 1].Neurons[k].Value += layers[i].Neurons[k].Value / w[k] * Util.GetDerivativeAtX(activationFuntion, w[k]);
                        }
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
                    layerafter.Neurons[j].Value = ActivationFuntion(layerafter.Neurons[j].Value);
                }
            }
            Layer layerOutput = Layers[Layers.Count - 1];
            for (int k = 0; k < layerOutput.Neurons.Count; k++)
            {
                result.Add(layerOutput.Neurons[k].Value);
            }
            return result;
        }

        public Network(List<int> nNeuronsInLayers, ActivationFuntionDelegate activationFuntion)
        {
            this.activationFuntion = activationFuntion;
            layers = new List<Layer>();
            for (int i = 0; i < nNeuronsInLayers.Count; i++)
            {
                if (nNeuronsInLayers[i] < 1)
                    throw new IndexOutOfRangeException("The number of neurons in a layer must be greater than 0");
                int nafter = 0;
                if (i != nNeuronsInLayers.Count - 1)
                    nafter = nNeuronsInLayers[i + 1];
                Layer layer = new Layer(nNeuronsInLayers[i], nafter);
                layers.Add(layer);
            }
        }
    }
}
