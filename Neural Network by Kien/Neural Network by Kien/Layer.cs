using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    public class Layer
    {
        private List<Neuron> neurons;
        
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
        public Layer(int nNeurons, int nafter)
        {
            neurons = new List<Neuron>();
            for (int i = 0; i < nNeurons; i++)
            {
                Neurons.Add(new Neuron(nafter));
            }
        }
        public Layer(double[] v, int nafter)
        {
            neurons = new List<Neuron>();
            for (int i = 0; i < v.Length; i++)
            {
                Neurons.Add(new Neuron(v[i], nafter));
            }
        }
    }
}
