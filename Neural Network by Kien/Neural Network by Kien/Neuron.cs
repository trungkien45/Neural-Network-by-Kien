using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    public class Neuron
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

        public List<Dendrite> Dendrites { get { return dendrites; }}

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
            for(int i=0;i< nDendritesToAfter; i++)
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
}
