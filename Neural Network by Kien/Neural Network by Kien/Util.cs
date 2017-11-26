using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_by_Kien
{
    public static class Util
    {
        const double delta = 10e-6;

        private static Random rnd = new Random();
        public static int GetRandomI()
        {
            return rnd.Next();
        }
        public static double GetRandomD()
        {
            return rnd.NextDouble();
        }
        public static double GetDerivativeAtX(Network.ActivationFuntionDelegate activationFuntion, double x)
        {
            return activationFuntion(x + delta) / (x + delta);
        }
    }
}
