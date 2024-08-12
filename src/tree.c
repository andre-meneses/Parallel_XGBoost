#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "data.h"
#include "utils.h"

#define score(G, H) (((G) * (G)) / ((H) + tree->lambda))
#define weight(G, H) (-(G) / ((H) + tree->lambda))


typedef struct {
    double score;
    double split;
    int feature;
    Subset *subset;
} BestSplit;

TreeNode *splitNode(Data *Xy, Subset *index_subset, int cur_depth, GradientPair *gpair, XGBoostTree *tree) {
    double G = 0, H = 1e-16;
    TreeNode *ret = mallocOrDie(sizeof(TreeNode));

    // Redução para calcular G e H
    #pragma omp parallel for reduction(+:G,H)
    for (int i = 0; i < Xy->n_example; i++) {
        if (inSubset(i, index_subset)) {
            G += gpair[i].g;
            H += gpair[i].h;
        }
    }

    if (cur_depth == tree->max_depth) {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
        return ret;
    }

    double before_score = score(G, H);

    // Inicializa o melhor split global
    BestSplit global_best_split;
    global_best_split.score = 0.0;
    global_best_split.subset = initSubset(index_subset->size, 0);

    // Cada thread terá sua própria estrutura BestSplit
    #pragma omp parallel
    {
        BestSplit local_best_split;
        local_best_split.score = 0.0;
        local_best_split.subset = initSubset(index_subset->size, 0);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < Xy->n_feature; i++) {
            double G_L = 0, H_L = 0, G_R, H_R;
            Subset *local_next_subset = initSubset(index_subset->size, 0);
            resetSubset(local_next_subset);

            for (int j = 0; j < Xy->n_example; j++) {
                if (local_next_subset->cnt == index_subset->cnt - 1) break;

                int example_index = Xy->feature_blocks[i][j];

                if (!inSubset(example_index, index_subset)) continue;

                addToSubset(example_index, local_next_subset);

                G_L += gpair[example_index].g;
                H_L += gpair[example_index].h;
                G_R = G - G_L;
                H_R = H - H_L;
                double after_score = score(G_L, H_L) + score(G_R, H_R);

                if (after_score > local_best_split.score) {
                    local_best_split.score = after_score;
                    local_best_split.feature = i;
                    local_best_split.split = Xy->X[example_index][i];
                    memcpy(local_best_split.subset->bitset, local_next_subset->bitset, sizeof(char) * BITNSLOTS(local_next_subset->size));
                    local_best_split.subset->cnt = local_next_subset->cnt;
                }
            }
            freeSubset(local_next_subset);
        }

        // Comparação global fora da seção crítica
        #pragma omp critical
        {
            if (local_best_split.score > global_best_split.score) {
                global_best_split.score = local_best_split.score;
                global_best_split.feature = local_best_split.feature;
                global_best_split.split = local_best_split.split;
                memcpy(global_best_split.subset->bitset, local_best_split.subset->bitset, sizeof(char) * BITNSLOTS(local_best_split.subset->size));
                global_best_split.subset->cnt = local_best_split.subset->cnt;
            }
        }

        freeSubset(local_best_split.subset);
    }

    if (global_best_split.score > before_score + tree->gamma) {
        ret->feature_id = global_best_split.feature;
        ret->info.split_cond = global_best_split.split;
        ret->left = splitNode(Xy, global_best_split.subset, cur_depth + 1, gpair, tree);
        global_best_split.subset->cnt = index_subset->cnt - global_best_split.subset->cnt;

        for (int i = 0; i < BITNSLOTS(global_best_split.subset->size); i++) {
            global_best_split.subset->bitset[i] = index_subset->bitset[i] & ~(global_best_split.subset->bitset[i]);
        }
        ret->right = splitNode(Xy, global_best_split.subset, cur_depth + 1, gpair, tree);
    } else {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
    }

    freeSubset(global_best_split.subset);
    return ret;
}


void fitTree(Data *Xy, GradientPair *gpair, XGBoostTree *tree) {
    Subset *ss = initSubset(Xy->n_example, 1);
    tree->root = splitNode(Xy, ss, 0, gpair, tree);

    freeSubset(ss);
}

void predictTree(Data *Xy, double *outy, XGBoostTree *tree) {
    #pragma omp parallel for
    for (int i = 0; i < Xy->n_example; i++) {
        TreeNode *node = tree->root;
        while (!isLeaf(node)) {
            double t = Xy->X[i][node->feature_id];
            if (t <= node->info.split_cond)
                node = node->left;
            else
                node = node->right;
        }
        outy[i] = node->info.leaf_value;
    }
}

void printNode(TreeNode *node, int depth) {
    for (int i = 0; i < depth; i++) printf("\t");
    if (isLeaf(node))
        printf("leaf value: %f\n", node->info.leaf_value);
    else {
        printf("feature: %d, split: %f\n", node->feature_id, node->info.split_cond);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("LEFT\n");
        printNode(node->left, depth + 1);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("RIGHT\n");
        printNode(node->right, depth + 1);
    }
}

void printTree(XGBoostTree *tree) { printNode(tree->root, 0); }