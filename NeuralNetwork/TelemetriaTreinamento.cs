using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class TelemetriaTreinamento
    {
        public int Iteracao { get; set; }
        public double[][,] Pesos { get; set; }
        public double[][] Vies { get; set; }
        public double[] Erro { get; set; }

    }
}
