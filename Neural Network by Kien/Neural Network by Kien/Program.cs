using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> list = new List<int>();
            list.Add(1);
            list.Add(3);
            list.Add(1);
            Network network = new Network(list, new Network.ActivationFuntionDelegate(x=>x),0.1);
            network.Train(new double[] { 3 }, new double[] { 3 });
            network.Run(new double[] { 3 });
        }
    }
}
