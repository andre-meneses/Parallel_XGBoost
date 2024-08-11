#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "data.h"
#include "utils.h"
#define score(G, H) (((G) * (G)) / ((H) + tree->lambda))
#define weight(G, H) (-(G) / ((H) + tree->lambda))

TreeNode *splitNode(Data *Xy, Subset *index_subset, int cur_depth,
                    GradientPair *gpair, XGBoostTree *tree) {
    double G = 0, H = 1e-16;
    TreeNode *ret = mallocOrDie(sizeof(TreeNode));

    #pragma omp parallel for reduction(+:G,H)
    for (int i = 0; i < Xy->n_example; i++)
        if (inSubset(i, index_subset)) {
            G += gpair[i].g;
            H += gpair[i].h;
        }
    
    if (cur_depth == tree->max_depth) {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
        return ret;
    }

    double before_score = score(G, H);
    Subset *best_next_subset = initSubset(index_subset->size, 0);
    double best_score = 0;
    int best_feature = -1;
    double best_split = 0;

    #pragma omp parallel
    {
        double local_best_score = 0;
        double local_best_split = 0;
        int local_best_feature = -1;
        Subset *local_best_next_subset = initSubset(index_subset->size, 0);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < Xy->n_feature; i++) {
            double G_L = 0, H_L = 0, G_R, H_R;
            Subset *local_next_subset = initSubset(index_subset->size, 0);
            resetSubset(local_next_subset);

            for (int j = 0; j < Xy->n_example; j++) {
                int example_index = Xy->feature_blocks[i][j];
                if (!inSubset(example_index, index_subset))
                    continue;
                
                addToSubset(example_index, local_next_subset);
                G_L += gpair[example_index].g;
                H_L += gpair[example_index].h;
                G_R = G - G_L;
                H_R = H - H_L;
                double after_score = score(G_L, H_L) + score(G_R, H_R);

                if (after_score > local_best_score) {
                    local_best_score = after_score;
                    local_best_feature = i;
                    local_best_split = Xy->X[example_index][i];
                    memcpy(local_best_next_subset->bitset, local_next_subset->bitset,
                           sizeof(char) * BITNSLOTS(local_next_subset->size));
                    local_best_next_subset->cnt = local_next_subset->cnt;
                }
            }
            freeSubset(local_next_subset);
        }

        #pragma omp critical
        {
            if (local_best_score > best_score) {
                best_score = local_best_score;
                best_feature = local_best_feature;
                best_split = local_best_split;
                memcpy(best_next_subset->bitset, local_best_next_subset->bitset,
                       sizeof(char) * BITNSLOTS(local_best_next_subset->size));
                best_next_subset->cnt = local_best_next_subset->cnt;
            }
        }
        freeSubset(local_best_next_subset);
    }

    if (best_score > before_score + tree->gamma) {
        ret->feature_id = best_feature;
        ret->info.split_cond = best_split;
        ret->left = splitNode(Xy, best_next_subset, cur_depth + 1, gpair, tree);
        best_next_subset->cnt = index_subset->cnt - best_next_subset->cnt;
        for (int i = 0; i < BITNSLOTS(best_next_subset->size); i++)
            best_next_subset->bitset[i] = index_subset->bitset[i] & ~(best_next_subset->bitset[i]);
        ret->right = splitNode(Xy, best_next_subset, cur_depth + 1, gpair, tree);
    } else {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
    }
    freeSubset(best_next_subset);
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
