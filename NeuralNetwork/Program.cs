using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //Rede Neural Artificial Simples
            //Vamos construir uma ANN de 3 camadas para aprender a função XOR.

            //Defina as camadas de rede
            //a camada 0 é a camada de entrada, 1 é a camada oculta e 2 é a camada de saída.
            int[] camadas = new[] { 2, 2, 1 };
            var rede = new RedeNeural(camadas)
            {
                Iteracoes = 1000,              //iterações de treinamento
                Alpha = 3.5,                    //taxa de aprendizagem, menor é mais lento, muito alto pode não convergir.
                L2_Regularizacao = true,       //definir a regularização L2 para evitar overfitting
                Lambda = 0.0003,                //força de L2
                Rnd = new Random(12345)         //fornecer uma semente para resultados repetíveis
            };

            //Defina as entradas e as saídas.
            //O último valor nesse conjunto de treinamento é a saída esperada.

            //XOR
            var X_treinamento = new double[][]
            {
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
            };

            //Normalize a entrada para -1, + 1 quando necessário
            //Queremos uma média de 0 e 1 stdev.

            //Pegue as 2 primeiras colunas como entrada e a última 1 coluna como destino y_treinamento (o rótulo esperado)
            var entradas = new double[X_treinamento.GetLength(0)][];
            for (int entrada = 0; entrada < X_treinamento.GetLength(0); entrada++)
            {
                entradas[entrada] = new double[camadas[0]];
                for (int j = 0; j < camadas[0]; j++)
                    entradas[entrada][j] = X_treinamento[entrada][j];
            }

            //Crie a matriz de rótulo esperada
            var y_treinamento = new double[X_treinamento.GetLength(0)];
            for (int entrada = 0; entrada < X_treinamento.GetLength(0); entrada++)
                y_treinamento[entrada] = X_treinamento[entrada][camadas[0]];


            //Também vamos monitorar o X_treinamento fornecendo uma função delegate
            rede.Monitor = delegate (TelemetriaTreinamento telemetria)
            {
                Console.CursorLeft = 0;
                Console.CursorTop = 0;

                //Mostra algumas informações sobre seu aprendizado em cada iteração
                Console.WriteLine($"Iteração {telemetria.Iteracao}");

                //Exibir alguns dados de amostra
                Console.WriteLine($"{rede.Prever(new[] { 0.0, 0.0 })[0]} -> 0");
                Console.WriteLine($"{rede.Prever(new[] { 0.0, 1.0 })[0]} -> 1");
                Console.WriteLine($"{rede.Prever(new[] { 1.0, 0.0 })[0]} -> 1");
                Console.WriteLine($"{rede.Prever(new[] { 1.0, 1.0 })[0]} -> 0");

                //Imprimir os pesos e Vieses
                Console.WriteLine("\nPesos:");
                for (int camada = 0; camada < telemetria.Pesos.Length; camada++)
                {
                    Console.WriteLine($"  Camada {camada}");
                    Console.WriteLine("  --------------------------");
                    for (int j = 0; j < telemetria.Pesos[camada].GetLength(0); j++)
                    {
                        for (int k = 0; k < telemetria.Pesos[camada].GetLength(1); k++)
                            Console.Write("  {0:#.##}\t", telemetria.Pesos[camada][j, k]);

                        Console.WriteLine();
                    }
                }

                Console.WriteLine("\nVieses:");
                Console.WriteLine("--------------------------");
                for (int camada = 1; camada < telemetria.Vies.Length; camada++)
                {
                    for (int neuronio = 0; neuronio < telemetria.Vies[camada].Length; neuronio++)
                        Console.Write("  {0:#.##}\t", telemetria.Vies[camada][neuronio]);
                    Console.WriteLine();
                }

                //Exibir erro médio
                var absCost = (double)telemetria.Erro.Sum(v => Math.Abs(v)) / telemetria.Erro.Length;
                Console.WriteLine("\nErro {0:#.#####}", absCost);
            };

            //Comece o treinamento da rede para aprender a função que corresponde aos nossos dados.
            rede.Treinamento(entradas, y_treinamento);

            //Confirme se funcionou
            Console.WriteLine($"A rede aprendeu XOR(1,0)={rede.Prever(new[] { 1.0, 0.0 })[0]}");
            Console.WriteLine($"A rede aprendeu XOR(1,1)={rede.Prever(new[] { 1.0, 1.0 })[0]}");
            Console.ReadKey(true);
        }
    }
}
