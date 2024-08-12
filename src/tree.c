#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "data.h"
#include "utils.h"
#define score(G, H) (((G) * (G)) / ((H) + tree->lambda))
#define weight(G, H) (-(G) / ((H) + tree->lambda))

// 1º FUNÇÃO CRÍTICA
#include <omp.h>

typedef struct {
    double score;
    int feature;
    double split;
    Subset *subset;
} BestData;

#pragma omp declare reduction(maximum : BestData : omp_out = omp_in.score > omp_out.score ? omp_in : omp_out)

TreeNode *splitNode(Data *Xy, Subset *index_subset, int cur_depth, GradientPair *gpair, XGBoostTree *tree) {
    double G = 0, H = 1e-16;
    TreeNode *ret = (TreeNode *)malloc(sizeof(TreeNode));

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
    Subset *next_subset = initSubset(index_subset->size, 0);
    Subset *best_next_subset = initSubset(index_subset->size, 0);
    BestData best = {0, -1, 0.0, best_next_subset};

    // Parallel for each feature
    #pragma omp parallel for schedule(dynamic) private(next_subset)
    for (int i = 0; i < Xy->n_feature; i++) {
        double G_L = 0, H_L = 0, G_R, H_R;
        Subset *local_next_subset = initSubset(index_subset->size, 0);
        resetSubset(local_next_subset);

        for (int j = 0; j < Xy->n_example; j++) {
            if (local_next_subset->cnt == index_subset->cnt - 1)
                break;

            int example_index = Xy->feature_blocks[i][j];

            if (!inSubset(example_index, index_subset))
                continue;

            addToSubset(example_index, local_next_subset);
            G_L = gpair[example_index].g;
            H_L = gpair[example_index].h;
            G_R = G - G_L;
            H_R = H - H_L;
            double after_score = score(G_L, H_L) + score(G_R, H_R);

            #pragma omp critical
            {
                if (after_score > best.score) {
                    best.score = after_score;
                    best.feature = i;
                    best.split = Xy->X[example_index][i];
                    memcpy(best.subset->bitset, local_next_subset->bitset, sizeof(char) * BITNSLOTS(local_next_subset->size));
                    best.subset->cnt = local_next_subset->cnt;
                }
            }
        }
        freeSubset(local_next_subset);
    }

    if (best.score > before_score + tree->gamma) {
        ret->feature_id = best.feature;
        ret->info.split_cond = best.split;
        ret->left = splitNode(Xy, best.subset, cur_depth + 1, gpair, tree);
        best.subset->cnt = index_subset->cnt - best.subset->cnt;
        for (int i = 0; i < BITNSLOTS(best.subset->size); i++)
            best.subset->bitset[i] = index_subset->bitset[i] & ~(best.subset->bitset[i]);
        ret->right = splitNode(Xy, best.subset, cur_depth + 1, gpair, tree);
    } else {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
    }
    freeSubset(next_subset);
    freeSubset(best.subset);
    return ret;
}

void fitTree(Data *Xy, GradientPair *gpair, XGBoostTree *tree) {
    Subset *ss = initSubset(Xy->n_example, 1);
    tree->root = splitNode(Xy, ss, 0, gpair, tree);
    freeSubset(ss);
}

// 4º FUNÇÃO CRÍTICA
void predictTree(Data *Xy, double *outy, XGBoostTree *tree) {
    //TreeNode *node = NULL;
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
        printf("feature: %d, split: %f\n", node->feature_id,
               node->info.split_cond);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("LEFT\n");
        printNode(node->left, depth + 1);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("RIGHT\n");
        printNode(node->right, depth + 1);
    }
}

void printTree(XGBoostTree *tree) { printNode(tree->root, 0); }
