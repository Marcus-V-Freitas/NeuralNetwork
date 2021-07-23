using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class RedeNeural
    {
        /// <summary>
        /// L Camadas * N Neuronios
        /// </summary>
        private Neuronio[][] Neuronios { get; set; }

        /// <summary>
        /// L-1 Camada * Node * Node Pesos
        /// </summary>
        private double[][,] Pesos { get; set; }

        /// <summary>
        /// Número de iterações de aprendizado
        /// </summary>
        public int Iteracoes { get; set; } = 5000;

        /// <summary>
        /// Alterna para controlar a aplicação de regularização L2 com pesos já treinados
        /// </summary>
        public bool L2_Regularizacao { get; set; }

        /// <summary>
        /// A força da regularização L2
        /// </summary>
        public double Lambda { get; set; } = 0.00003;

        /// <summary>
        /// Controla a taxa de aprendizado. Aumente para saltos maiores, mas com cautela, pesos muito altos, pode impedir a convergência.
        /// </summary>
        public double Alpha { get; set; } = 5.5;

        private int UltimaCamada { get; set; }

        public Random Rnd { get; set; } = new Random();

        /// <summary>
        /// Delegate opcional para monitorar o aprendizado da rede neural
        /// </summary>
        public Action<TelemetriaTreinamento> Monitor { get; set; }


        /// <summary>
        /// Constroi uma nova rede artifical
        /// </summary>
        /// <param name="camadas"></param>
        public RedeNeural(int[] camadas)
        {
            UltimaCamada = camadas.Length - 1;
            Neuronios = new Neuronio[camadas.Length][];

            for (int camada = 0; camada < camadas.Length; camada++)
            {
                Neuronios[camada] = new Neuronio[camadas[camada]];
                for (int neuronio = 0; neuronio < camadas[camada]; neuronio++)
                {
                    //Inicializa cada node[
                    Neuronios[camada][neuronio] = new Neuronio();
                }
            }
        }

        /// <summary>
        /// Inicializar os pesos de forma aleatória*
        /// </summary>
        private void InicializarPesos()
        {
            // Recomenda-se inicializar pesos aleatoriamente, com média 0.
            // A variância dessas ponderações também deve diminuir à medida que você itera para
            // os neurônios de saída. A ideia é que não queremos pesos mais rasos
            // para aprender mais rápido do que as camadas mais profundas.

            // A intuição é que a complexidade da rede é determinada pelo número de
            // neurônios e neurônios com peso zero desaparecem efetivamente. Os pesos distribuem
            // correções no gradiente e se forem grandes, eles aprenderão efetivamente mais rápido
            // do que outros neurônios.

            // O tamanho da matriz é camada (i) * camada (i + 1)
            int camadas = Neuronios.GetLength(0);
            Pesos = new double[camadas - 1][,];
            for (int camada = 0; camada < camadas - 1; camada++)
            {
                Pesos[camada] = new double[Neuronios[camada].Length, Neuronios[camada + 1].Length];  //2x3, 3x1
                for (int neuronio = 0; neuronio < Neuronios[camada].Length; neuronio++)
                {
                    for (int peso = 0; peso < Neuronios[camada + 1].Length; peso++)
                        Pesos[camada][neuronio, peso] = FuncaoPesos(camada + 1);
                }
            }
        }

        /// <summary>
        /// Amostra de uma distribuição aleatória teoricamente ótima de pesos de neurônios
        /// </summary>
        /// <param name="camada"> Atual camada à se atribuir os pesos </param>
        private double FuncaoPesos(int camada)
        {
            // Faça uma amostra aleatória de uma distribuição uniforme no intervalo [-b, b] onde b é:
            // onde fanIn é o número de unidades de entrada nos pesos e
            // fanOut é o número de unidades de saída nestes pesos
            var fanIn = (camada > 0) ? Neuronios[camada - 1].Length : 0;
            var fanOut = Neuronios[camada].Length;
            var b = Math.Sqrt(6) / Math.Sqrt(fanIn + fanOut);
            return Rnd.NextDouble() * 2 * b - b;
        }

        /// <summary>
        /// Initialize Vieses to zero
        /// </summary>
        private void InicializarVieses()
        {
            int layers = Neuronios.GetLength(0);
            for (int camada = 1; camada < layers; camada++)
            {
                for (int neuronio = 0; neuronio < Neuronios[camada].Length; neuronio++)
                {
                    Neuronios[camada][neuronio].Vies = 0.0;
                }
            }
        }


        /// <summary>
        /// O treinamento permite que a rede aprenda como prever Saida y com base nas entradas.
        /// </summary>
        /// <param name="Entrada"> entradas de treinamento </param>
        /// <param name="y"> saidas de treinamento </param>
        public void Treinamento(double[][] Entrada, double[] y)
        {
            InicializarPesos();
            InicializarVieses();

            int iteracao = Iteracoes;

            var custo = new double[Neuronios[UltimaCamada].Length];

            while (iteracao-- > 0)
            {
                //Faça um loop em cada entrada e calcule a previsão
                for (int entrada = 0; entrada < Entrada.GetLength(0); entrada++)
                {

                    //FeedForward
                    var Saida = Prever(Entrada[entrada]);

                    //Realize o treinamento atualizando o viés e os pesos após cada iteração

                    //Calcula o erro na camada de saída
                    for (int neuronio = 0; neuronio < Neuronios[UltimaCamada].Length; neuronio++)
                    {
                        custo[neuronio] = Neuronios[UltimaCamada][neuronio].Saida - y[entrada];

                        //Atribuir à variavel do erro da camada seu respectivos erro 
                        Neuronios[UltimaCamada][neuronio].Erro = (custo[neuronio] * SigmoidPrime(Neuronios[UltimaCamada][neuronio].Entrada + Neuronios[UltimaCamada][neuronio].Vies));
                    }

                    //Retropropaga o erro pela rede
                    BackPropagate();

                    //Ajuste o Vies pelo valor deste Erro
                    for (int camada = 1; camada <= UltimaCamada; camada++)
                        for (int neuronio = 0; neuronio < Neuronios[camada].Length; neuronio++)
                            // Ajuste o Vies pela derivada do Erro (que é o próprio Erro)
                            Neuronios[camada][neuronio].Vies -= (Alpha * Neuronios[camada][neuronio].Erro);

                    //Ajuste os Pesos pelo valor deste Erro
                    for (int camada = 0; camada <= UltimaCamada - 1; camada++)
                    {
                        for (int j = 0; j < Neuronios[camada].Length; j++)
                            for (int k = 0; k < Neuronios[camada + 1].Length; k++)
                            {
                                Pesos[camada][j, k] -= (Alpha * Neuronios[camada][j].Saida * Neuronios[camada + 1][k].Erro);

                                //Adicionar regularização L2 para evitar overfitting, desencorajando pesos elevados
                                if (L2_Regularizacao)
                                    Pesos[camada][j, k] -= (Lambda * Pesos[camada][j, k]);
                            }
                    }
                }

                if (Monitor != null)
                {
                    //Produz uma matriz jagged de Vies
                    var Vies = new double[Neuronios.Length][];
                    for (int i = 0; i < Vies.GetLength(0); i++)
                    {
                        Vies[i] = new double[Neuronios[i].Length];
                        for (int j = 0; j < Vies[i].Length; j++)
                            Vies[i][j] = Neuronios[i][j].Vies;
                    }

                    var telemetry = new TelemetriaTreinamento()
                    {
                        Iteracao = iteracao,
                        Pesos = Pesos,
                        Vies = Vies,
                        Erro = custo
                    };
                    Monitor(telemetry);
                }
            }
        }

        /// <summary>
        ///  Use a rede neural para prever a Saida dada alguma entrada.
        /// </summary>
        /// <param name="Entrada">Valores a serem avaliados</param>
        public double[] Prever(double[] Entrada)
        {
            //Propagação Forward
            //Preencher as primeiras camadas de neurônios Saida com dados de entrada
            for (int d = 0; d < Neuronios[0].Length; d++)
                Neuronios[0][d].Saida = Entrada[d];

            //Fase de feed forward
            for (int l = 1; l < Neuronios.GetLength(0); l++)
            {
                //Agora calcular cada neuronio da camada
                for (int neuronio = 0; neuronio < Neuronios[l].Length; neuronio++)
                {
                    //Calcula o neurônio na camada
                    double soma = 0;

                    //Iterar sobre as saídas e pesos das camadas anteriores                 
                    for (int neuronioAnterior = 0; neuronioAnterior < Neuronios[l - 1].Length; neuronioAnterior++)
                        soma += (Neuronios[l - 1][neuronioAnterior].Saida * Pesos[l - 1][neuronioAnterior, neuronio]);

                    //Armazena as entradas ponderadas na Entrada.
                    Neuronios[l][neuronio].Entrada = soma;

                    //A Saida é o sigmóide da Entrada ponderada mais o Vies
                    Neuronios[l][neuronio].Saida = Sigmoid(Neuronios[l][neuronio].Entrada + Neuronios[l][neuronio].Vies);
                }
            }

            //prepara um vetor de saídas para retornar
            var outputlayer = Neuronios.GetLength(0) - 1;
            var Saida = new double[Neuronios[outputlayer].Length];
            for (int n = 0; n < Saida.Length; n++)
                Saida[n] = Neuronios[outputlayer][n].Saida;

            return Saida;
        }

        /// <summary>
        /// Retropropague o Erro proporcionalmente a todos os neurônios por sua contribuição para a Saida.
        /// </summary>
        public void BackPropagate()
        {
            //Da direita para a esquerda (Saida para Entrada)
            for (int camada = UltimaCamada - 1; camada > 0; camada--)
            {
                for (int neuronio = 0; neuronio < Neuronios[camada].Length; neuronio++)
                {
                    //Soma do produto do peso * Erro na camada + 1
                    double soma = 0.0;
                    for (int node = 0; node < Neuronios[camada + 1].Length; node++)
                    {
                        //Pesos da camada atual é na verdade da camada + 1
                        soma += (Pesos[camada][neuronio, node] * Neuronios[camada + 1][node].Erro);
                    }

                    Neuronios[camada][neuronio].Erro = soma * SigmoidPrime(Neuronios[camada][neuronio].Entrada + Neuronios[camada][neuronio].Vies);
                }
            }
        }

        /// <summary>
        /// Calcula a função de transformação sigmóide (x).
        /// </summary>
        /// <param name="x"> valor de entrada </param>
        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Calcula a derivada do sigmóide(x)
        /// </summary>
        /// <param name="x"> valor de entrada </param>

        private static double SigmoidPrime(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }
    }
}
