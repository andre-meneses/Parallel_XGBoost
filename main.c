#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "tree.h"
#include "utils.h"
#include "xgb.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number>\n", argv[0]);
        return 1;
    }

    // Obter o número do argumento de entrada
    char *number = argv[1];

    // Criar o nome do arquivo usando o número fornecido
    char filename[256];
    snprintf(filename, sizeof(filename), "sintetic-datasets/cate_mushroom_%s.csv", number);

    // Ler o CSV
    Data *Xy = readCSV(filename, ",", 0);
    if (Xy == NULL) {
        fprintf(stderr, "Error reading CSV file: %s\n", filename);
        return 1;
    }

    // Criar e ajustar o modelo
    XGBoostModel *m = createXGBoostModel(MultiClassification);
    fitModel(Xy, m);

    // Alocar memória para as previsões e fazer a predição
    double *outy = mallocOrDie(sizeof(double) * Xy->n_example);
    predictModel(Xy, outy, m);

    // Imprimir a matriz de confusão
    printConfusionMatrix(Xy, outy);

    return 0;
}
