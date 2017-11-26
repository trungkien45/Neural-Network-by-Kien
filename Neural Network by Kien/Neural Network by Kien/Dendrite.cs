using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    /// <summary>
    /// a short branched extension of a nerve cell, along which impulses received from other cells at synapses are transmitted to the cell body.
    /// một nhánh ngắn của tế bào thần kinh, dọc theo đó các xung điện nhận được từ các tế bào khác ở khớp thần kinh được truyền đến thân tế bào.
    /// </summary>
    public class Dendrite
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

        private double getRandom(double MinValue, double MaxValue)
        {
            return Util.GetRandomD() * (MaxValue - MinValue) + MinValue;
        }
    }
}
