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
            Network network = new Network(list, new Network.ActivationFuntionDelegate(x => 1 / (1 + Math.Exp(-x))), 0.1);
            Random random = new Random();
            for (int i = 0; i < 20000; i++)
            {
                if (random.Next() % 2 == 0)
                    //n.Train(ins, ots);
                    network.Train(new double[] { 0 }, new double[] { 0 });
                else
                    //nn.Train(ins1, ots1);
                    network.Train(new double[] { 1 }, new double[] { 1 });
            }

            Console.WriteLine(network.Run(new double[] { 1 })[0]);
            Console.WriteLine(network.Run(new double[] { 0 })[0]);
            Console.ReadKey();
        }
    }
}
